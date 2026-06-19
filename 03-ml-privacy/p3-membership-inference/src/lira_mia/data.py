"""Datasets for the membership-inference experiment.

DEFAULT (offline, no download): a synthetic tabular classification problem built
with scikit-learn's `make_classification`. It is deliberately *over-parameterised
relative to its size* (small N, several informative + redundant features, a few
classes) so the target model memorises some training points -- which is exactly
the leakage LiRA exploits. Everything runs in seconds on a CPU.

OPTIONAL (enhanced, needs a download): Fashion-MNIST via torchvision. Imported
lazily so this module imports fine without torchvision installed / without data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


@dataclass
class Dataset:
    """A flat feature matrix + integer labels. `n_classes` is cached for models."""

    X: np.ndarray  # (N, D) float32
    y: np.ndarray  # (N,)   int64
    n_classes: int
    n_features: int


def make_synthetic_pool(
    n_samples: int = 3000,
    n_features: int = 30,
    n_informative: int = 12,
    n_classes: int = 6,
    seed: int = 42,
) -> Dataset:
    """A single large 'population' pool we later split into target/shadow worlds.

    Membership inference needs a population from which both the target's training
    set and the shadow datasets are drawn i.i.d. We standardise features once on
    the whole pool (a fixed, dataset-level transform -- not per-split fitting).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative - 2,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=0.6,
        flip_y=0.05,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = y.astype(np.int64)
    return Dataset(X=X, y=y, n_classes=n_classes, n_features=n_features)


def load_fashion_mnist_pool(root: str = "data", n_samples: int = 4000) -> Dataset:
    """OPTIONAL enhanced path: flattened Fashion-MNIST as a tabular pool.

    Lazily imports torchvision so the module loads without it. Returns the same
    `Dataset` shape so the rest of the pipeline is dataset-agnostic.
    """
    try:
        import torchvision  # noqa: F401
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - optional path
        raise RuntimeError(
            "Fashion-MNIST path needs torchvision. `pip install torchvision` "
            "or use the default synthetic pool."
        ) from exc

    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=tfm)
    idx = np.random.RandomState(0).permutation(len(ds))[:n_samples]
    X = np.stack([ds[i][0].numpy().reshape(-1) for i in idx]).astype(np.float32)
    y = np.array([int(ds[i][1]) for i in idx], dtype=np.int64)
    return Dataset(X=X, y=y, n_classes=10, n_features=X.shape[1])
