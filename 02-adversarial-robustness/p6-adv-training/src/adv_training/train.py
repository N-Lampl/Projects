"""Standard and PGD-adversarial training of the SmallCNN.

Adversarial training (Madry et al. 2018) solves the saddle-point problem

    min_theta  E_(x,y)[ max_{||d||_inf <= eps}  L(f_theta(x + d), y) ]

by, on every minibatch, first approximating the inner max with PGD and then
taking the usual SGD/Adam step on the *adversarial* examples. Set `adv_epsilon=0`
to recover ordinary (standard) training.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .attack import pgd_perturb
from .model import SmallCNN


def train(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    adv_epsilon: float = 0.0,
    adv_steps: int = 7,
    device: torch.device | None = None,
    log_every: int = 0,
) -> nn.Module:
    """Train `model`. If `adv_epsilon > 0`, do PGD adversarial training."""
    device = device or torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    adversarial = adv_epsilon > 0

    for epoch in range(1, epochs + 1):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if adversarial:
                # Inner max: craft PGD examples against the CURRENT model.
                # pgd_perturb toggles eval/grad internally; restore train mode.
                model.eval()
                x = pgd_perturb(model, x, y, adv_epsilon, steps=adv_steps)
                model.train()
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            if log_every and i % log_every == 0:
                tag = f"adv(eps={adv_epsilon})" if adversarial else "std"
                print(f"[{tag}] epoch {epoch}/{epochs}  batch {i:>4}  loss {loss.item():.4f}")
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
