"""Data loaders for the standard-vs-adversarial-training comparison.

DEFAULT (offline) path: deterministic SYNTHETIC 28x28 "digit-like" images,
generated with torch only — NO download, NO torchvision required. Each of 10
classes is a fixed random per-class template (a smooth blob) plus per-sample
noise, kept in [0, 1] so the L-inf epsilon maps directly to a fraction of full
pixel intensity (exactly like MNIST in the FGSM project). The classes are
linearly-ish separable but far from trivial, so a standard CNN is clearly
vulnerable to PGD and adversarial training clearly helps.

OPTIONAL (enhanced) path: real MNIST via torchvision, enabled with `--real`.
torchvision is imported lazily so this module imports fine without it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_IMG = 28
_NUM_CLASSES = 10


def _class_templates(num_classes: int = _NUM_CLASSES, seed: int = 1234) -> torch.Tensor:
    """One fixed smooth template per class, shape (C, 1, 28, 28), values in [0, 1].

    Each template is a sum of a few Gaussian blobs at class-specific locations,
    giving visually distinct, smoothly-varying patterns. Deterministic in `seed`.
    """
    g = torch.Generator().manual_seed(seed)
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, _IMG), torch.linspace(0, 1, _IMG), indexing="ij"
    )
    templates = torch.zeros(num_classes, 1, _IMG, _IMG)
    for c in range(num_classes):
        img = torch.zeros(_IMG, _IMG)
        n_blobs = 3
        for _ in range(n_blobs):
            cx, cy = torch.rand(2, generator=g).tolist()
            sigma = 0.10 + 0.15 * torch.rand(1, generator=g).item()
            amp = 0.6 + 0.6 * torch.rand(1, generator=g).item()
            img += amp * torch.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)))
        img = img / img.max().clamp(min=1e-6)
        templates[c, 0] = img
    return templates.clamp(0, 1)


def make_synthetic(
    n_per_class: int,
    *,
    noise: float = 0.25,
    seed: int = 0,
) -> TensorDataset:
    """Build a balanced synthetic dataset: templates + per-sample noise, in [0, 1]."""
    templates = _class_templates()
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in range(_NUM_CLASSES):
        base = templates[c].expand(n_per_class, 1, _IMG, _IMG)
        eps = noise * torch.randn(n_per_class, 1, _IMG, _IMG, generator=g)
        x = (base + eps).clamp(0, 1)
        xs.append(x)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    X = torch.cat(xs)
    Y = torch.cat(ys)
    # shuffle once, deterministically
    perm = torch.randperm(len(X), generator=g)
    return TensorDataset(X[perm], Y[perm])


def get_loaders(
    batch_size: int = 128,
    *,
    real: bool = False,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    train_subset: int | None = None,
    test_subset: int | None = None,
    train_per_class: int = 600,
    test_per_class: int = 200,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader).

    Default: synthetic (offline, torch-only). `real=True`: download MNIST.
    `*_subset` caps total examples (used for fast smoke runs).
    """
    if real:
        train_ds, test_ds = _real_mnist(data_dir)
    else:
        train_ds = make_synthetic(train_per_class, seed=0)
        test_ds = make_synthetic(test_per_class, seed=999)

    if train_subset is not None:
        train_ds = Subset(train_ds, range(min(train_subset, len(train_ds))))
    if test_subset is not None:
        test_ds = Subset(test_ds, range(min(test_subset, len(test_ds))))

    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _real_mnist(data_dir: str | Path):
    """Optional path. Imported lazily so the module loads without torchvision."""
    try:
        from torchvision import datasets, transforms
    except ImportError as e:  # pragma: no cover - optional path
        raise ImportError(
            "real=True needs torchvision (pip install torchvision). "
            "The default offline path uses synthetic data and needs only torch."
        ) from e

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()  # -> float tensor in [0, 1]
    train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)
    return train_ds, test_ds
