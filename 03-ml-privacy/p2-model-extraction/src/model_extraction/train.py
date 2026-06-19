"""Training + evaluation helpers used by both the victim and the thief."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import MLP


def train(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int = 8,
    lr: float = 1e-3,
    device: torch.device | None = None,
    log_every: int = 0,
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
    return model.eval()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device | None = None) -> float:
    """Top-1 accuracy in [0, 1] against the true labels."""
    device = device or torch.device("cpu")
    model.to(device).eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def agreement(
    a: nn.Module, b: nn.Module, loader: DataLoader, device: torch.device | None = None
) -> float:
    """Fraction of inputs on which models `a` and `b` predict the SAME class.

    This is the model-stealing notion of FIDELITY: how often the thief reproduces
    the victim's decision, regardless of whether that decision is correct.
    """
    device = device or torch.device("cpu")
    a.to(device).eval()
    b.to(device).eval()
    same = total = 0
    for x, _ in loader:
        x = x.to(device)
        same += (a(x).argmax(1) == b(x).argmax(1)).sum().item()
        total += x.shape[0]
    return same / max(total, 1)


def save_model(model: nn.Module, path: str | Path, img_size: int, n_classes: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "img_size": img_size, "n_classes": n_classes}, path
    )


def load_victim(path: str | Path, device: torch.device | None = None) -> MLP:
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    model = MLP(ckpt["img_size"], ckpt["n_classes"], hidden=256)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
