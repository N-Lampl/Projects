"""Train and evaluate the SmallCNN target classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import SmallCNN


def train(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int = 2,
    lr: float = 1e-3,
    device: torch.device | None = None,
    log_every: int = 100,
) -> nn.Module:
    device = device or torch.device("cpu")
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            if log_every and i % log_every == 0:
                print(f"epoch {epoch}/{epochs}  batch {i:>4}  loss {loss.item():.4f}")
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device | None = None) -> float:
    """Clean top-1 accuracy in [0, 1]."""
    device = device or torch.device("cpu")
    model.to(device).eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def save_model(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str | Path, device: torch.device | None = None) -> SmallCNN:
    device = device or torch.device("cpu")
    model = SmallCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()
