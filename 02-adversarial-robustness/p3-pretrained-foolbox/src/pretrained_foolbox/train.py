"""Train / load the offline SmallCNN target (so we have *something* to attack).

The pretrained ResNet-18 path needs no training; this exists only for the
fully-offline default, where we self-train a small classifier on synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import make_synthetic
from .model import SmallCNN


def train(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
    log_every: int = 1,
) -> nn.Module:
    """Quick supervised training loop. CPU, a handful of epochs."""
    device = device or torch.device("cpu")
    model.to(device).train()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        total = correct = 0
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            running += loss.item() * yb.size(0)
            correct += int((out.argmax(1) == yb).sum())
            total += yb.size(0)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  epoch {ep + 1}/{epochs}  loss={running / total:.4f}  acc={correct / total:.3f}")
    return model.eval()


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float((model(x).argmax(1) == y).float().mean())


def save_model(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path, device: torch.device, num_classes: int = 4) -> SmallCNN:
    model = SmallCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def train_offline_target(
    path: Path, device: torch.device, epochs: int = 6, seed: int = 42
) -> SmallCNN:
    """Build + train the SmallCNN on synthetic data and save it."""
    x, y = make_synthetic(n_per_class=96, seed=seed)
    model = SmallCNN(num_classes=4)
    train(model, x, y, epochs=epochs, device=device)
    save_model(model, path)
    return model.eval()
