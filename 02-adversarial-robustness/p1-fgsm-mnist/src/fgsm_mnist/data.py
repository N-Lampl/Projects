"""MNIST data loaders.

IMPORTANT: we use ToTensor only (pixels in [0, 1]) and NO normalization. FGSM
clips perturbed images back to the valid pixel range [0, 1], so keeping the data
in that range is what makes the epsilon values directly interpretable as a
fraction of full pixel intensity.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def get_loaders(
    batch_size: int = 128,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    train_subset: int | None = None,
    test_subset: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader). Downloads MNIST on first run.

    `*_subset` caps the number of examples (used for fast smoke runs).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()  # -> float tensor in [0, 1]

    train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)

    if train_subset is not None:
        train_ds = Subset(train_ds, range(min(train_subset, len(train_ds))))
    if test_subset is not None:
        test_ds = Subset(test_ds, range(min(test_subset, len(test_ds))))

    # generator pinned for deterministic shuffling
    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
