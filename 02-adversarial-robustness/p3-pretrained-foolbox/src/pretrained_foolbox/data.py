"""Sample images for the inference-time attack.

DEFAULT (offline): deterministic SYNTHETIC 32x32 RGB images with 4 visually
distinct classes (color blobs / patterns). No downloads. Pixels in [0, 1].

OPTIONAL (online): `get_cifar_samples()` pulls a few CIFAR-10 test images via
torchvision. Documented in data/README.md; never required for `make attack`.
"""

from __future__ import annotations

import numpy as np
import torch

# Human-readable names for the 4 synthetic classes.
SYNTH_CLASSES = ["red_disk", "green_bars", "blue_grad", "checker"]


def _make_class_image(cls: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a (3, size, size) float image in [0, 1] for a given class.

    Each class has a distinctive, learnable structure plus a little noise, so a
    small CNN reaches high accuracy and there is real signal for a gradient to
    push against.
    """
    # Start from a mid-gray base so every class shares a lot of pixels (low
    # margin) -- this keeps the classifier accurate but NOT trivially robust,
    # so a small L-inf perturbation can actually flip predictions.
    img = np.full((3, size, size), 0.45, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    amp = 0.18  # weak signal -> small margins -> attackable

    if cls == 0:  # faint red disk
        mask = r < size * 0.35
        img[0][mask] += amp
    elif cls == 1:  # faint vertical green bars
        bars = (xx // 4) % 2 == 0
        img[1][bars] += amp
    elif cls == 2:  # faint blue horizontal gradient
        img[2] += (xx / (size - 1)).astype(np.float32) * amp
    else:  # cls == 3: faint grayscale checkerboard
        check = ((xx // 4) % 2) ^ ((yy // 4) % 2)
        val = (check.astype(np.float32) - 0.5) * amp
        img[0] += val
        img[1] += val
        img[2] += val

    # substantial noise relative to the signal -> realistic, attackable margins
    img += rng.normal(0.0, 0.10, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def make_synthetic(
    n_per_class: int = 64,
    size: int = 32,
    num_classes: int = 4,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic synthetic dataset: (N, 3, size, size) in [0, 1], plus labels."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for cls in range(num_classes):
        for _ in range(n_per_class):
            xs.append(_make_class_image(cls, size, rng))
            ys.append(cls)
    x = torch.from_numpy(np.stack(xs)).float()
    y = torch.tensor(ys, dtype=torch.long)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    return x[perm], y[perm]


def get_cifar_samples(n: int = 8, root: str = "data") -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """OPTIONAL online path: a few CIFAR-10 test images as [0, 1] tensors."""
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())
    xs = torch.stack([ds[i][0] for i in range(n)])
    ys = torch.tensor([ds[i][1] for i in range(n)])
    return xs, ys, list(ds.classes)
