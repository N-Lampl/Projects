"""The teacher and student networks (plain MLPs, CPU-friendly).

The teacher is a wide two-hidden-layer MLP; the student is a much narrower,
single-hidden-layer MLP with far fewer parameters. Keeping both as stacks of
``nn.Linear`` layers is deliberate - it is exactly what magnitude pruning and
dynamic quantization operate on.
"""

from __future__ import annotations

import torch
from torch import nn


class Teacher(nn.Module):
    """Wide two-hidden-layer MLP classifier."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Student(nn.Module):
    """Narrow single-hidden-layer MLP with far fewer parameters than the teacher."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_params(model: nn.Module) -> int:
    """Total number of parameters in a module."""
    return sum(p.numel() for p in model.parameters())
