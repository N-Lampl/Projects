"""A tiny MLP classifier + a synthetic dataset, so training is fast and offline.

No real dataset is required: we generate a deterministic linearly-separable-ish
blob with scikit-learn. The whole point of this project is the *supply chain*
around the model artifact, not the model's accuracy.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TinyMLP(nn.Module):
    """2-layer MLP. Small enough to train in <1s on CPU."""

    def __init__(self, in_features: int = 20, hidden: int = 32, n_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_synthetic_data(
    n: int = 2000,
    in_features: int = 20,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic synthetic classification data (no download, fully offline)."""
    from sklearn.datasets import make_classification

    x, y = make_classification(
        n_samples=n,
        n_features=in_features,
        n_informative=in_features // 2,
        n_classes=n_classes,
        random_state=seed,
    )
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.int64))
