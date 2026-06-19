"""Data for the model-inversion target.

Default path (OFFLINE, no download): a synthetic 28x28 "digit-like" dataset of
10 classes. Each class has a fixed *prototype* image (a distinct shape rendered
on the grid); samples are the prototype plus per-pixel noise and a small random
shift. Because every class has a clean, consistent visual signature, a tiny CNN
learns it perfectly and model inversion can recover a recognisable prototype --
which is exactly the privacy story we want to demonstrate, with zero downloads.

Optional path: pass `use_mnist=True` to train on real MNIST instead (downloaded
lazily via torchvision). The inversion code is identical either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
IMG = 28
N_CLASSES = 10


def _draw(canvas: np.ndarray, pts: list[tuple[int, int]], val: float = 1.0) -> None:
    for r, c in pts:
        if 0 <= r < IMG and 0 <= c < IMG:
            canvas[r, c] = val


def _line(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    n = max(abs(r1 - r0), abs(c1 - c0)) + 1
    rs = np.linspace(r0, r1, n).round().astype(int)
    cs = np.linspace(c0, c1, n).round().astype(int)
    return list(zip(rs.tolist(), cs.tolist()))


def _thicken(canvas: np.ndarray, k: int = 1) -> np.ndarray:
    """Cheap dilation so strokes are a couple of pixels wide (more digit-like)."""
    out = canvas.copy()
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            out = np.maximum(out, np.roll(np.roll(canvas, dr, axis=0), dc, axis=1))
    return out


def class_prototypes() -> np.ndarray:
    """Return (10, 28, 28) float32 prototypes in [0, 1] -- a distinct shape per class.

    These are the "secret" class signatures the target model memorises; model
    inversion should reconstruct something resembling them.
    """
    protos = np.zeros((N_CLASSES, IMG, IMG), dtype=np.float32)
    cx = IMG // 2

    # 0: ring
    rr, cc = np.mgrid[0:IMG, 0:IMG]
    ring = (np.abs(np.sqrt((rr - cx) ** 2 + (cc - cx) ** 2) - 8) < 1.5).astype(np.float32)
    protos[0] = ring
    # 1: vertical bar
    _draw(protos[1], _line(4, cx, 23, cx))
    # 2: horizontal top + diagonal + horizontal bottom (a "Z"/2 shape)
    _draw(protos[2], _line(5, 6, 5, 21) + _line(5, 21, 22, 6) + _line(22, 6, 22, 21))
    # 3: two stacked right-facing arcs (E-ish bars)
    _draw(protos[3], _line(5, 6, 5, 20) + _line(13, 6, 13, 20) + _line(22, 6, 22, 20)
          + _line(5, 20, 22, 20))
    # 4: a cross / plus
    _draw(protos[4], _line(4, cx, 23, cx) + _line(cx, 5, cx, 22))
    # 5: top bar, left down, mid bar, right down, bottom bar (S-ish)
    _draw(protos[5], _line(5, 6, 5, 21) + _line(5, 6, 13, 6) + _line(13, 6, 13, 21)
          + _line(13, 21, 22, 21) + _line(22, 6, 22, 21))
    # 6: left bar + bottom loop
    _draw(protos[6], _line(5, 8, 22, 8) + _line(22, 8, 22, 19) + _line(13, 8, 13, 19)
          + _line(13, 19, 22, 19))
    # 7: top bar + long diagonal
    _draw(protos[7], _line(5, 6, 5, 21) + _line(5, 21, 22, 9))
    # 8: full box with mid bar
    _draw(protos[8], _line(5, 6, 5, 21) + _line(22, 6, 22, 21) + _line(5, 6, 22, 6)
          + _line(5, 21, 22, 21) + _line(13, 6, 13, 21))
    # 9: X shape
    _draw(protos[9], _line(5, 6, 22, 21) + _line(5, 21, 22, 6))

    for k in range(N_CLASSES):
        protos[k] = _thicken(protos[k], k=1)
    return protos


def make_synthetic(
    n_per_class: int = 400,
    noise: float = 0.18,
    max_shift: int = 1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X, y): X is (N, 1, 28, 28) in [0, 1], y is (N,) class labels."""
    rng = np.random.default_rng(seed)
    protos = class_prototypes()
    xs, ys = [], []
    for k in range(N_CLASSES):
        for _ in range(n_per_class):
            img = protos[k].copy()
            if max_shift:
                dr = int(rng.integers(-max_shift, max_shift + 1))
                dc = int(rng.integers(-max_shift, max_shift + 1))
                img = np.roll(np.roll(img, dr, axis=0), dc, axis=1)
            img = img + rng.normal(0, noise, size=img.shape).astype(np.float32)
            xs.append(np.clip(img, 0.0, 1.0))
            ys.append(k)
    X = torch.tensor(np.stack(xs)[:, None, :, :], dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    return X[perm], y[perm]


def get_loaders(
    batch_size: int = 128,
    n_per_class: int = 400,
    test_frac: float = 0.2,
    use_mnist: bool = False,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader).

    Default: synthetic offline data. `use_mnist=True` -> real MNIST (downloads).
    """
    if use_mnist:
        return _mnist_loaders(batch_size, data_dir)

    X, y = make_synthetic(n_per_class=n_per_class, seed=seed)
    n_test = int(len(y) * test_frac)
    test_ds = TensorDataset(X[:n_test], y[:n_test])
    train_ds = TensorDataset(X[n_test:], y[n_test:])
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _mnist_loaders(batch_size: int, data_dir: str | Path) -> tuple[DataLoader, DataLoader]:
    """Optional enhanced path: real MNIST. Imported lazily so the module still
    imports without torchvision data on disk / network access."""
    from torchvision import datasets, transforms  # lazy

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)
    g = torch.Generator().manual_seed(42)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
