"""Data for the adversarial-input detector.

DEFAULT (offline) path: a deterministic SYNTHETIC 28x28 "digit-like" dataset, so
the whole project runs with zero network access and zero downloads. The shapes
are simple line/curve strokes per class — enough for the SmallCNN to reach high
accuracy and for FGSM to meaningfully fool it, which is all the detector needs.

OPTIONAL (enhanced) path: real MNIST via torchvision, enabled with
`--dataset mnist`. MNIST is ~11 MB and auto-downloads on first use; it is
git-ignored and never committed. If the download fails (offline), we fall back
to the synthetic set automatically.

In both cases pixels live in [0, 1] (no normalization) so FGSM's epsilon maps
directly to a fraction of full pixel intensity and feature-squeezing bit-depth
reduction is interpretable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _draw_digit(canvas: np.ndarray, cls: int, rng: np.random.Generator) -> None:
    """Render a crude class-dependent stroke pattern into a 28x28 canvas in-place."""
    jitter = lambda: int(rng.integers(-2, 3))  # noqa: E731
    t = 2  # stroke half-thickness

    def hline(r: int, c0: int, c1: int) -> None:
        r = np.clip(r + jitter(), t, 27 - t)
        canvas[r - t : r + t, c0:c1] = 1.0

    def vline(c: int, r0: int, r1: int) -> None:
        c = np.clip(c + jitter(), t, 27 - t)
        canvas[r0:r1, c - t : c + t] = 1.0

    def diag() -> None:
        for i in range(4, 24):
            r = i
            c = np.clip(i + jitter(), t, 27 - t)
            canvas[r - t : r + t, c - t : c + t] = 1.0

    # Each class gets a distinct, learnable signature.
    if cls == 0:
        vline(7, 5, 23); vline(20, 5, 23); hline(5, 7, 21); hline(22, 7, 21)
    elif cls == 1:
        vline(14, 4, 24)
    elif cls == 2:
        hline(5, 6, 22); vline(20, 5, 14); hline(14, 6, 22); vline(7, 14, 23); hline(22, 6, 22)
    elif cls == 3:
        hline(5, 6, 22); hline(14, 6, 22); hline(22, 6, 22); vline(20, 5, 23)
    elif cls == 4:
        vline(7, 4, 15); vline(20, 4, 24); hline(14, 6, 22)
    elif cls == 5:
        hline(5, 6, 22); vline(7, 5, 14); hline(14, 6, 22); vline(20, 14, 23); hline(22, 6, 22)
    elif cls == 6:
        vline(7, 5, 23); hline(5, 7, 21); hline(14, 7, 21); hline(22, 7, 21); vline(20, 14, 23)
    elif cls == 7:
        hline(5, 6, 22); diag()
    elif cls == 8:
        vline(7, 5, 23); vline(20, 5, 23); hline(5, 7, 21); hline(14, 7, 21); hline(22, 7, 21)
    else:  # 9
        vline(7, 5, 14); vline(20, 5, 23); hline(5, 7, 21); hline(14, 7, 21)


def synthetic_digits(n: int, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x[n,1,28,28] in [0,1], y[n]) of class-balanced synthetic digits."""
    rng = np.random.default_rng(seed)
    xs = np.zeros((n, 1, 28, 28), dtype=np.float32)
    ys = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cls = int(rng.integers(0, 10))
        ys[i] = cls
        _draw_digit(xs[i, 0], cls, rng)
        # light speckle noise + global intensity wobble for realism
        xs[i, 0] += rng.normal(0, 0.05, size=(28, 28)).astype(np.float32)
        xs[i, 0] *= float(rng.uniform(0.85, 1.0))
    np.clip(xs, 0.0, 1.0, out=xs)
    return torch.from_numpy(xs), torch.from_numpy(ys)


def _synthetic_loaders(
    batch_size: int, n_train: int, n_test: int
) -> tuple[DataLoader, DataLoader]:
    xtr, ytr = synthetic_digits(n_train, seed=1)
    xte, yte = synthetic_digits(n_test, seed=2)
    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True, generator=g
    )
    test_loader = DataLoader(TensorDataset(xte, yte), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_loaders(
    batch_size: int = 128,
    dataset: str = "synthetic",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    train_subset: int | None = 6000,
    test_subset: int | None = 2000,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader).

    dataset="synthetic" (default) -> offline synthetic digits, no download.
    dataset="mnist"               -> real MNIST via torchvision (optional path).
    Falls back to synthetic if MNIST cannot be loaded.
    """
    if dataset == "synthetic":
        return _synthetic_loaders(batch_size, train_subset or 6000, test_subset or 2000)

    # ---- optional MNIST path (lazy import; degrade gracefully) ----
    try:
        from torchvision import datasets, transforms

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        tfm = transforms.ToTensor()  # -> float tensor in [0, 1]
        train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)
        if train_subset is not None:
            train_ds = Subset(train_ds, range(min(train_subset, len(train_ds))))
        if test_subset is not None:
            test_ds = Subset(test_ds, range(min(test_subset, len(test_ds))))
        g = torch.Generator().manual_seed(42)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, generator=g
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    except Exception as exc:  # noqa: BLE001 - any download/import failure -> fallback
        print(f"[data] MNIST unavailable ({exc}); falling back to synthetic digits.")
        return _synthetic_loaders(batch_size, train_subset or 6000, test_subset or 2000)
