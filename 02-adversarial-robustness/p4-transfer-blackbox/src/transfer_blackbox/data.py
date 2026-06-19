"""Data for the transfer / black-box study.

DEFAULT (offline) path: a SYNTHETIC MNIST-like dataset of 8x8 grayscale digit
glyphs rendered procedurally. No download, no network — `make attack` works out
of the box with only torch/numpy installed. The glyphs are simple stroked shapes
for digits 0-9 with per-sample jitter/noise, which is enough to (a) train two
distinct small classifiers to high accuracy and (b) study adversarial transfer.

OPTIONAL (real) path: pass --real to the scripts to use torchvision MNIST
(downloaded to data/, git-ignored). Imported lazily so the module imports fine
without torchvision data on disk.

All pixels live in [0, 1] (ToTensor convention) so epsilon values are directly
interpretable as a fraction of full pixel intensity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
IMG_SIZE = 8  # synthetic glyphs are 8x8

# 7-segment-ish stroke maps on an 8x8 grid. Each entry lists (row, col) pixels
# that are "on" for that digit. Hand-drawn once; jitter/noise is added per sample.
_STROKES: dict[int, list[tuple[int, int]]] = {
    0: [(1, 2), (1, 3), (1, 4), (2, 1), (2, 5), (3, 1), (3, 5), (4, 1), (4, 5),
        (5, 1), (5, 5), (6, 2), (6, 3), (6, 4)],
    1: [(1, 4), (2, 3), (2, 4), (3, 4), (4, 4), (5, 4), (6, 3), (6, 4), (6, 5)],
    2: [(1, 2), (1, 3), (1, 4), (2, 5), (3, 4), (4, 3), (5, 2), (6, 2), (6, 3),
        (6, 4), (6, 5)],
    3: [(1, 2), (1, 3), (1, 4), (2, 5), (3, 3), (3, 4), (4, 5), (5, 5),
        (6, 2), (6, 3), (6, 4)],
    4: [(1, 4), (2, 3), (2, 4), (3, 2), (3, 4), (4, 1), (4, 4), (4, 5),
        (5, 4), (6, 4)],
    5: [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (3, 2), (3, 3),
        (4, 5), (5, 5), (6, 1), (6, 2), (6, 3), (6, 4)],
    6: [(1, 3), (1, 4), (2, 2), (3, 1), (4, 1), (4, 2), (4, 3), (5, 1), (5, 5),
        (6, 2), (6, 3), (6, 4)],
    7: [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 4), (4, 3),
        (5, 3), (6, 3)],
    8: [(1, 2), (1, 3), (1, 4), (2, 1), (2, 5), (3, 2), (3, 3), (3, 4),
        (4, 1), (4, 5), (5, 1), (5, 5), (6, 2), (6, 3), (6, 4)],
    9: [(1, 2), (1, 3), (1, 4), (2, 1), (2, 5), (3, 1), (3, 5), (4, 2), (4, 3),
        (4, 4), (4, 5), (5, 5), (6, 2), (6, 3)],
}


def _render_glyph(digit: int, rng: np.random.Generator) -> np.ndarray:
    """Render one 8x8 glyph in [0,1] with intensity jitter + Gaussian noise."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for r, c in _STROKES[digit]:
        img[r, c] = rng.uniform(0.7, 1.0)
        # bleed into a random neighbour so strokes have some width/variety
        if rng.random() < 0.35:
            dr, dc = rng.integers(-1, 2), rng.integers(-1, 2)
            rr, cc = np.clip(r + dr, 0, IMG_SIZE - 1), np.clip(c + dc, 0, IMG_SIZE - 1)
            img[rr, cc] = max(img[rr, cc], rng.uniform(0.4, 0.8))
    img += rng.normal(0.0, 0.06, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def make_synthetic(n_per_class: int = 600, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a synthetic digit tensor dataset. Returns (X in [0,1], y)."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for digit in range(10):
        for _ in range(n_per_class):
            xs.append(_render_glyph(digit, rng))
            ys.append(digit)
    x = torch.from_numpy(np.stack(xs)).unsqueeze(1)  # (N, 1, 8, 8)
    y = torch.tensor(ys, dtype=torch.long)
    perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed))
    return x[perm], y[perm]


def get_synthetic_loaders(
    n_per_class: int = 600,
    batch_size: int = 128,
    test_frac: float = 0.2,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Train/test DataLoaders over the synthetic glyph dataset (offline default)."""
    x, y = make_synthetic(n_per_class=n_per_class, seed=seed)
    n_test = int(len(x) * test_frac)
    train_ds = TensorDataset(x[n_test:], y[n_test:])
    test_ds = TensorDataset(x[:n_test], y[:n_test])
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_real_loaders(
    batch_size: int = 128,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    train_subset: int | None = 6000,
    test_subset: int | None = 1000,
) -> tuple[DataLoader, DataLoader]:
    """OPTIONAL: real MNIST via torchvision (downloaded, git-ignored). Lazy import."""
    from torch.utils.data import Subset  # noqa: PLC0415
    from torchvision import datasets, transforms  # noqa: PLC0415

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)
    if train_subset is not None:
        train_ds = Subset(train_ds, range(min(train_subset, len(train_ds))))
    if test_subset is not None:
        test_ds = Subset(test_ds, range(min(test_subset, len(test_ds))))
    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_loaders(real: bool = False, **kwargs) -> tuple[DataLoader, DataLoader]:
    """Dispatch to synthetic (default, offline) or real MNIST loaders."""
    return get_real_loaders(**kwargs) if real else get_synthetic_loaders(**kwargs)
