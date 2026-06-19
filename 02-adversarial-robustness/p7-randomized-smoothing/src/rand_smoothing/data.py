"""Data loaders for the smoothed classifier.

DEFAULT (offline) path: a deterministic SYNTHETIC "digit-like" dataset generated
with NumPy/torch only — no download, works with the always-installed libs. Each
class is a distinct low-frequency 2D pattern plus noise, so a tiny CNN learns it
in one epoch yet it still exercises the full smoothing / certification pipeline.

OPTIONAL path: real MNIST via torchvision (`--dataset mnist`). torchvision is
imported lazily so this module imports fine without a network or the dataset.

Pixels are kept in [0, 1] (no normalization) so the smoothing noise sigma and the
certified L2 radius are expressed directly in input (pixel) units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
IMG = 28


def _class_prototypes(num_classes: int = 10, size: int = IMG) -> torch.Tensor:
    """One smooth, distinct grayscale prototype per class (deterministic)."""
    yy, xx = np.meshgrid(
        np.linspace(0, np.pi, size), np.linspace(0, np.pi, size), indexing="ij"
    )
    protos = []
    for k in range(num_classes):
        fx, fy = 1 + k % 4, 1 + (k // 4)
        phase = (k * 0.7) % np.pi
        p = np.sin(fx * xx + phase) * np.cos(fy * yy + 0.3 * k)
        p = (p - p.min()) / (p.max() - p.min() + 1e-8)  # -> [0, 1]
        # pull prototypes toward a common gray mean so classes partially overlap;
        # this makes per-point confidence (and thus certified radii) genuinely vary.
        p = 0.5 + 0.45 * (p - 0.5)
        protos.append(p)
    return torch.tensor(np.stack(protos), dtype=torch.float32)  # (K, 28, 28)


def make_synthetic(
    n: int, num_classes: int = 10, noise: float = 0.35, seed: int = 0
) -> TensorDataset:
    """Generate n labeled synthetic images in [0, 1] (prototype + Gaussian noise)."""
    g = torch.Generator().manual_seed(seed)
    protos = _class_prototypes(num_classes)
    labels = torch.randint(0, num_classes, (n,), generator=g)
    base = protos[labels]  # (n, 28, 28)
    imgs = base + noise * torch.randn(n, IMG, IMG, generator=g)
    imgs = imgs.clamp(0.0, 1.0).unsqueeze(1)  # (n, 1, 28, 28)
    return TensorDataset(imgs, labels)


def get_loaders(
    batch_size: int = 128,
    dataset: str = "synthetic",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    train_subset: int | None = None,
    test_subset: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader).

    dataset="synthetic" (default, offline) or "mnist" (optional, downloads).
    `*_subset` caps the number of examples.
    """
    if dataset == "synthetic":
        train_ds: object = make_synthetic(train_subset or 6000, seed=1)
        test_ds: object = make_synthetic(test_subset or 1000, seed=2)
    elif dataset == "mnist":
        from torchvision import datasets, transforms  # lazy: optional dependency

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        tfm = transforms.ToTensor()  # -> float tensor in [0, 1]
        train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)
        if train_subset is not None:
            train_ds = Subset(train_ds, range(min(train_subset, len(train_ds))))
        if test_subset is not None:
            test_ds = Subset(test_ds, range(min(test_subset, len(test_ds))))
    else:
        raise ValueError(f"unknown dataset {dataset!r} (use 'synthetic' or 'mnist')")

    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
