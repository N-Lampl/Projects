"""Small classifiers for the victim and the thief.

Both are tiny MLPs so the whole sweep trains in a couple of minutes on CPU. The
thief is intentionally given a SLIGHTLY different architecture than the victim
(different hidden width) to make the point that extraction does not require
knowing the victim's exact architecture -- only its input/output interface.
"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, img_size: int, n_classes: int = 10, hidden: int = 256) -> None:
        super().__init__()
        in_features = img_size * img_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_victim(img_size: int, n_classes: int = 10) -> MLP:
    return MLP(img_size, n_classes, hidden=256)


def make_thief(img_size: int, n_classes: int = 10) -> MLP:
    # different capacity than the victim on purpose
    return MLP(img_size, n_classes, hidden=192)
