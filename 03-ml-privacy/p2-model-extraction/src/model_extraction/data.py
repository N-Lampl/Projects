"""Data for the model-extraction experiment.

DEFAULT (offline) path: a synthetic 16x16 "digit-like" dataset generated with
scikit-learn's `make_classification`, reshaped into 1-channel images. This needs
NO download and runs anywhere torch+sklearn are installed.

OPTIONAL (enhanced) path: real MNIST via torchvision, enabled with
`--dataset mnist`. torchvision is imported lazily so this module still imports
without it.

Both paths return tensors with pixels in [0, 1] and integer labels.

The KEY split for model stealing:
    * `victim`  -- the data the victim was trained on (private to the defender).
    * `attack`  -- the (unlabelled) pool the THIEF is allowed to query the victim
                   with. The thief NEVER sees true labels for these; it only sees
                   the victim's predicted labels.
    * `test`    -- held-out data used to measure both models' accuracy and to
                   measure THIEF<->VICTIM agreement (fidelity).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
IMG_SIZE = 16  # synthetic images are 16x16 = 256 features
N_CLASSES = 6


@dataclass
class Splits:
    """Tensor splits. Images are (N, 1, H, W) in [0, 1]; labels are (N,) int64."""

    victim_x: torch.Tensor
    victim_y: torch.Tensor
    attack_x: torch.Tensor  # thief query pool (labels withheld)
    attack_y: torch.Tensor  # true labels, kept ONLY to report an oracle upper bound
    test_x: torch.Tensor
    test_y: torch.Tensor
    img_size: int
    n_classes: int

    @property
    def n_attack(self) -> int:
        return self.attack_x.shape[0]


def _to_images(x: np.ndarray, img_size: int) -> torch.Tensor:
    """Min-max scale features to [0, 1] and reshape to (N, 1, H, W)."""
    x = x.astype(np.float32)
    lo, hi = x.min(0, keepdims=True), x.max(0, keepdims=True)
    x = (x - lo) / np.clip(hi - lo, 1e-8, None)
    n = x.shape[0]
    return torch.from_numpy(x).reshape(n, 1, img_size, img_size)


def make_synthetic(
    *,
    n_victim: int = 4000,
    n_attack: int = 8000,
    n_test: int = 2000,
    img_size: int = IMG_SIZE,
    n_classes: int = N_CLASSES,
    seed: int = 42,
) -> Splits:
    """Generate a synthetic image-classification dataset with sklearn.

    We make one big informative-feature dataset and slice it into the three
    splits, so victim/attack/test come from the same distribution (the realistic
    case where the thief can sample in-distribution inputs).
    """
    from sklearn.datasets import make_classification

    n_features = img_size * img_size
    total = n_victim + n_attack + n_test
    x, y = make_classification(
        n_samples=total,
        n_features=n_features,
        n_informative=60,
        n_redundant=20,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=2.5,
        flip_y=0.01,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    x, y = x[perm], y[perm]

    imgs = _to_images(x, img_size)
    labels = torch.from_numpy(y.astype(np.int64))

    a, b = n_victim, n_victim + n_attack
    return Splits(
        victim_x=imgs[:a],
        victim_y=labels[:a],
        attack_x=imgs[a:b],
        attack_y=labels[a:b],
        test_x=imgs[b:],
        test_y=labels[b:],
        img_size=img_size,
        n_classes=n_classes,
    )


def make_mnist(
    *,
    n_victim: int = 4000,
    n_attack: int = 8000,
    n_test: int = 2000,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    seed: int = 42,
) -> Splits:
    """OPTIONAL real-MNIST path (downloads ~11 MB via torchvision on first run)."""
    from torchvision import datasets, transforms  # lazy: optional dependency

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()
    train = datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
    test = datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)

    def stack(ds, idx):
        xs = torch.stack([ds[i][0] for i in idx])
        ys = torch.tensor([ds[i][1] for i in idx], dtype=torch.int64)
        return xs, ys

    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(train))
    test_idx = rng.permutation(len(test))[:n_test]

    vx, vy = stack(train, train_idx[:n_victim])
    ax, ay = stack(train, train_idx[n_victim : n_victim + n_attack])
    tx, ty = stack(test, test_idx)
    return Splits(vx, vy, ax, ay, tx, ty, img_size=28, n_classes=10)


def get_splits(dataset: str = "synthetic", **kwargs) -> Splits:
    if dataset == "synthetic":
        return make_synthetic(**kwargs)
    if dataset == "mnist":
        return make_mnist(**kwargs)
    raise ValueError(f"unknown dataset {dataset!r} (use 'synthetic' or 'mnist')")


def loader(x: torch.Tensor, y: torch.Tensor, batch_size: int = 128, shuffle: bool = False):
    g = torch.Generator().manual_seed(42)
    return DataLoader(
        TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, generator=g
    )
