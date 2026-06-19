"""Data loaders for the attack zoo.

DEFAULT (offline) path: a SYNTHETIC, linearly-separable-ish image dataset so the
project runs with zero downloads and stays fast on CPU. Each class gets a fixed
random "prototype" image plus per-sample noise, which a SmallCNN learns to ~95%+
in one or two epochs — enough to make the attacks visibly succeed.

OPTIONAL path: real CIFAR-10 (a 3-class subset, few epochs) or MNIST via
torchvision, selected with `source="cifar10"` / `source="mnist"`. These download
data (CIFAR-10 ~170MB) and are documented in the README as an enhanced path.

All pixels are kept in [0, 1] (no normalization) so the perturbation budgets
(epsilon for L-inf, the L2 norms reported) map directly to pixel intensity.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def make_synthetic(
    n_per_class: int,
    num_classes: int = 3,
    channels: int = 3,
    size: int = 32,
    noise: float = 0.22,
    seed: int = 0,
) -> TensorDataset:
    """Build a synthetic image classification dataset in [0, 1].

    Each class has a fixed random prototype (a smooth low-rank pattern). Samples
    are prototype + Gaussian noise, clamped to [0, 1]. Classes are separable
    enough for a small CNN to learn to high accuracy, while leaving room for the
    attacks to flip the predictions.

    NOTE: the prototype grid is generated from a FIXED seed independent of `seed`
    so the train and test splits share the same class prototypes (only their
    per-sample noise differs).
    """
    g = torch.Generator().manual_seed(seed)
    gp = torch.Generator().manual_seed(1234)  # shared prototypes across splits
    # Low-frequency prototypes: random small grid upsampled -> smooth structure.
    base = torch.rand(num_classes, channels, size // 4, size // 4, generator=gp)
    protos = torch.nn.functional.interpolate(
        base, size=(size, size), mode="bilinear", align_corners=False
    )
    protos = (protos - protos.amin(dim=(2, 3), keepdim=True)) / (
        protos.amax(dim=(2, 3), keepdim=True) - protos.amin(dim=(2, 3), keepdim=True) + 1e-8
    )

    xs, ys = [], []
    for c in range(num_classes):
        n = n_per_class
        eps = torch.randn(n, channels, size, size, generator=g) * noise
        x = (protos[c].unsqueeze(0) + eps).clamp(0.0, 1.0)
        xs.append(x)
        ys.append(torch.full((n,), c, dtype=torch.long))
    X = torch.cat(xs)
    Y = torch.cat(ys)
    # deterministic shuffle
    perm = torch.randperm(X.shape[0], generator=g)
    return TensorDataset(X[perm], Y[perm])


def get_loaders(
    source: str = "synthetic",
    batch_size: int = 128,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    train_subset: int | None = None,
    test_subset: int | None = None,
    num_classes: int = 3,
) -> tuple[DataLoader, DataLoader, dict]:
    """Return (train_loader, test_loader, meta).

    meta = {"in_channels", "num_classes", "size", "class_names"}.

    source:
      "synthetic" (default, offline) -- no download, fast.
      "cifar10"   (optional)         -- real CIFAR-10, first `num_classes` classes.
      "mnist"     (optional)         -- real MNIST, first `num_classes` classes.
    """
    if source == "synthetic":
        train_ds = make_synthetic(
            n_per_class=400, num_classes=num_classes, seed=1
        )
        test_ds = make_synthetic(
            n_per_class=120, num_classes=num_classes, seed=2
        )
        meta = {
            "in_channels": 3,
            "num_classes": num_classes,
            "size": 32,
            "class_names": [f"class{c}" for c in range(num_classes)],
        }
    elif source in ("cifar10", "mnist"):
        from torchvision import datasets, transforms  # lazy: only if requested

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        tfm = transforms.ToTensor()
        if source == "cifar10":
            full_train = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=tfm)
            full_test = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=tfm)
            channels, size = 3, 32
            names = full_train.classes
            train_labels = torch.tensor(full_train.targets)
            test_labels = torch.tensor(full_test.targets)
        else:
            full_train = datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
            full_test = datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)
            channels, size = 1, 28
            names = [str(i) for i in range(10)]
            train_labels = full_train.targets
            test_labels = full_test.targets

        keep = set(range(num_classes))
        tr_idx = [i for i, y in enumerate(train_labels.tolist()) if y in keep]
        te_idx = [i for i, y in enumerate(test_labels.tolist()) if y in keep]
        train_ds = Subset(full_train, tr_idx)
        test_ds = Subset(full_test, te_idx)
        meta = {
            "in_channels": channels,
            "num_classes": num_classes,
            "size": size,
            "class_names": list(names[:num_classes]),
        }
    else:
        raise ValueError(f"unknown source {source!r}")

    if train_subset is not None:
        train_ds = Subset(train_ds, range(min(train_subset, len(train_ds))))
    if test_subset is not None:
        test_ds = Subset(test_ds, range(min(test_subset, len(test_ds))))

    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, meta
