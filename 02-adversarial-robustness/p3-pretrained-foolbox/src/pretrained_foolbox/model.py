"""Target models for the inference-time attack.

Two paths share ONE convention: every model accepts images with pixels in [0, 1]
and applies its own normalization internally. That way the attack's epsilon is a
fraction of full pixel intensity (raw image space), which is the only place a
human eye can judge "how big is this perturbation" — exactly like FGSM-on-MNIST.

- DEFAULT (offline): `SmallCNN`, a tiny 3-channel CNN we train ourselves on
  synthetic 32x32 data. No downloads, no real dataset.
- OPTIONAL (online): `load_pretrained_resnet18()` downloads torchvision's
  ImageNet-pretrained ResNet-18 weights (~45 MB) and wraps them so they also
  consume [0, 1] images. This is the "attack a PRETRAINED model" path.
"""

from __future__ import annotations

import torch
from torch import nn

# ImageNet normalization (used by the optional pretrained path).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalize(nn.Module):
    """Fold dataset normalization into the model so attacks see [0, 1] inputs."""

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class SmallCNN(nn.Module):
    """A small 3-channel CNN target for the offline default path.

    Two conv blocks + a linear head, sized for 32x32 RGB images. Trained on
    synthetic data, it is just a stand-in classifier whose own input gradients
    we exploit — the attack code is identical to what you'd run on ResNet-18.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        # Normalize with simple 0.5 mean/std so inputs are roughly centered.
        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # -> 4x4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.norm(x)))


def load_pretrained_resnet18() -> nn.Module:
    """Optional online path: ImageNet-pretrained ResNet-18 that eats [0, 1] images.

    Downloads ~45 MB of weights on first call. Raises if offline; the scripts
    catch that and fall back to the self-trained SmallCNN.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.eval()
    model = nn.Sequential(Normalize(IMAGENET_MEAN, IMAGENET_STD), backbone)
    model.eval()
    return model
