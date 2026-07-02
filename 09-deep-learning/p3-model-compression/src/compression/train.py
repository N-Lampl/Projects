"""Short, deterministic CPU training loops used by the baseline and the study."""

from __future__ import annotations

import torch
from torch import nn

from .data import ClassDataset


def train_classifier(
    model: nn.Module,
    data: ClassDataset,
    epochs: int = 12,
    lr: float = 1e-2,
    batch_size: int = 256,
    seed: int = 0,
) -> nn.Module:
    """Train ``model`` with Adam + cross-entropy on the train split (in place)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    x, y = data.x_train, data.y_train
    n = len(x)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model
