"""A small MLP target/shadow model + training and the LiRA confidence signal.

The same architecture is used for the target model and every shadow model -- LiRA
assumes the attacker can train shadow models that mimic the target's training
procedure. We keep it tiny (CPU, seconds per model) but capacity-rich enough to
memorise, which is what creates the membership signal.
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


def train_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    device: torch.device | None = None,
    warm_start: dict | None = None,
) -> nn.Module:
    """Train (optionally warm-started from `warm_start` weights).

    Warm-starting every shadow from a shared, cheaply-pretrained checkpoint is the
    "warm-started shadows" trick: shadows converge in far fewer epochs, so we can
    afford 8-16 of them on a CPU while keeping each one well-fit.
    """
    device = device or torch.device("cpu")
    if warm_start is not None:
        model.load_state_dict(warm_start)
    model.to(device).train()

    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = Xt.shape[0]
    g = torch.Generator().manual_seed(0)

    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray, device=None) -> float:
    device = device or torch.device("cpu")
    model.eval().to(device)
    logits = model(torch.from_numpy(X).to(device))
    return float((logits.argmax(1).cpu().numpy() == y).mean())


@torch.no_grad()
def logit_confidence(model: nn.Module, X: np.ndarray, y: np.ndarray, device=None) -> np.ndarray:
    """The LiRA per-example signal: the *logit-scaled* confidence on the TRUE label.

        phi = log( p_y / (1 - p_y) )

    Carlini et al. (2022) show the model's confidence on its true label is roughly
    Gaussian *after* this logit transform, which is what makes the per-example
    likelihood-ratio test in `attack.py` well-calibrated. We compute p_y with a
    numerically stable log-softmax and clamp to avoid +/-inf.
    """
    device = device or torch.device("cpu")
    model.eval().to(device)
    logp = F.log_softmax(model(torch.from_numpy(X).to(device)), dim=1)
    logp_y = logp[torch.arange(len(y)), torch.from_numpy(y).to(device)]
    p_y = logp_y.exp().clamp(1e-6, 1 - 1e-6)
    phi = (p_y.log() - (1.0 - p_y).log()).cpu().numpy()
    return phi.astype(np.float64)
