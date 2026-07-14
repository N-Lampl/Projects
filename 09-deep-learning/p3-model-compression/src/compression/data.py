"""Data for the compression study - offline synthetic by default.

The default path draws a deterministic multi-class Gaussian-blob problem in
moderate dimension, so the fast tests need no network. ``load_mnist`` is an
optional torchvision path used only by the ``@slow`` test; torchvision is
imported lazily inside it so the fast suite never touches it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ClassDataset:
    """A classification dataset split into train / test tensors."""

    x_train: torch.Tensor  # (n_train, dim) float32 features
    y_train: torch.Tensor  # (n_train,) int64 labels
    x_test: torch.Tensor  # (n_test, dim) float32 features
    y_test: torch.Tensor  # (n_test,) int64 labels
    n_features: int
    n_classes: int
    source: str


def make_blobs(
    n_samples: int = 4000,
    n_features: int = 64,
    n_classes: int = 6,
    class_sep: float = 2.2,
    noise: float = 1.0,
    test_frac: float = 0.25,
    seed: int = 0,
) -> ClassDataset:
    """Draw a deterministic multi-class Gaussian-blob classification problem.

    Each class gets a random centroid on a shared scale; points are the centroid
    plus Gaussian noise. ``class_sep`` sets how far apart the centroids are, so
    the problem is hard enough that a small student can under-fit but a trained
    teacher clears chance by a wide margin.
    """
    rng = np.random.default_rng(seed)
    centroids = class_sep * rng.standard_normal((n_classes, n_features))

    per_class = n_samples // n_classes
    xs, ys = [], []
    for c in range(n_classes):
        pts = centroids[c] + noise * rng.standard_normal((per_class, n_features))
        xs.append(pts)
        ys.append(np.full(per_class, c, dtype=np.int64))
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)

    # Deterministic shuffle + split.
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    n_test = int(len(x) * test_frac)
    x_test, y_test = x[:n_test], y[:n_test]
    x_train, y_train = x[n_test:], y[n_test:]

    return ClassDataset(
        x_train=torch.from_numpy(x_train),
        y_train=torch.from_numpy(y_train),
        x_test=torch.from_numpy(x_test),
        y_test=torch.from_numpy(y_test),
        n_features=n_features,
        n_classes=n_classes,
        source=f"synthetic blobs (d={n_features}, k={n_classes})",
    )


def load_mnist(n_train: int = 8000, n_test: int = 2000, seed: int = 0) -> ClassDataset:
    """Load MNIST as flat float32 vectors via torchvision (optional, online).

    Imported lazily so the fast suite never depends on torchvision. Raises on any
    failure (missing package / download error); the ``@slow`` test catches that
    and skips.
    """
    import torchvision  # noqa: F401  (lazy: proves the dep is present)
    from torchvision import datasets, transforms

    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

    def _flatten(ds, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.arange(min(n, len(ds)))
        x = torch.stack([ds[i][0].reshape(-1) for i in idx])
        y = torch.tensor([ds[i][1] for i in idx], dtype=torch.int64)
        return x, y

    torch.manual_seed(seed)
    x_train, y_train = _flatten(train, n_train)
    x_test, y_test = _flatten(test, n_test)
    return ClassDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        n_features=x_train.shape[1],
        n_classes=10,
        source="MNIST (torchvision, flattened)",
    )
