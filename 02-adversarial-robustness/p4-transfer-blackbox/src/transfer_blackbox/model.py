"""Two DELIBERATELY DIFFERENT small classifiers.

Transferability is interesting precisely when the surrogate and the target are
*not* the same architecture. We provide:

  - SmallCNN : a 2-conv-layer convolutional net (the surrogate by default)
  - SmallMLP : a plain fully-connected net with a different nonlinearity (target)

Both use global adaptive pooling / flatten so they work for any square input
size (8x8 synthetic glyphs *and* 28x28 real MNIST) with no shape edits.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Convolutional surrogate. ReLU + max-pool + adaptive avg-pool head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),  # -> (B, 32, 2, 2), size-agnostic
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SmallMLP(nn.Module):
    """Fully-connected target with Tanh activations — a different decision
    surface from the conv surrogate, which is the whole point of a transfer study.
    Uses adaptive pooling first so the input dimension is fixed regardless of
    image size."""

    def __init__(self, num_classes: int = 10, hidden: int = 128) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(8)  # any input -> (B, 1, 8, 8)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.pool(x))


def build_model(kind: str) -> nn.Module:
    """Factory: 'cnn' -> SmallCNN, 'mlp' -> SmallMLP."""
    kind = kind.lower()
    if kind == "cnn":
        return SmallCNN()
    if kind == "mlp":
        return SmallMLP()
    raise ValueError(f"unknown model kind {kind!r} (expected 'cnn' or 'mlp')")
