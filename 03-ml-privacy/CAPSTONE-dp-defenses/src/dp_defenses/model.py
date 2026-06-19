"""The shared target/shadow/thief MLP + the LiRA confidence signal.

The same architecture is used for the DP target, every shadow model, and the
extraction thief, so the privacy-utility comparison is apples-to-apples: only the
*training procedure* (plain SGD vs DP-SGD) changes between the epsilon=inf and
finite-epsilon targets. We keep it tiny (CPU, seconds per model) but capacity-rich
enough to memorise, which is what creates the membership signal DP must suppress.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallMLP(nn.Module):
    """2-hidden-layer MLP. Works for tabular synthetic data or flattened images."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 128) -> None:
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


@torch.no_grad()
def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray, device=None) -> float:
    device = device or torch.device("cpu")
    model.eval().to(device)
    logits = model(torch.from_numpy(X).to(device))
    return float((logits.argmax(1).cpu().numpy() == y).mean())


@torch.no_grad()
def predict_labels(model: nn.Module, X: np.ndarray, device=None) -> np.ndarray:
    """Hard top-1 labels -- the (label-only) signal the extraction thief sees."""
    device = device or torch.device("cpu")
    model.eval().to(device)
    logits = model(torch.from_numpy(X).to(device))
    return logits.argmax(1).cpu().numpy().astype(np.int64)


@torch.no_grad()
def logit_confidence(model: nn.Module, X: np.ndarray, y: np.ndarray, device=None) -> np.ndarray:
    """The LiRA per-example signal: the *logit-scaled* confidence on the TRUE label.

        phi = log( p_y / (1 - p_y) )

    Carlini et al. (2022) show the model's confidence on its true label is roughly
    Gaussian *after* this logit transform, which makes the per-example
    likelihood-ratio test well-calibrated. Computed with a numerically stable
    log-softmax and clamped to avoid +/-inf.
    """
    device = device or torch.device("cpu")
    model.eval().to(device)
    logp = F.log_softmax(model(torch.from_numpy(X).to(device)), dim=1)
    logp_y = logp[torch.arange(len(y)), torch.from_numpy(y).to(device)]
    p_y = logp_y.exp().clamp(1e-6, 1 - 1e-6)
    phi = (p_y.log() - (1.0 - p_y).log()).cpu().numpy()
    return phi.astype(np.float64)
