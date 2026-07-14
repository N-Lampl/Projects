"""The Fast Gradient Sign Method (FGSM), from scratch - no attack library.

FGSM (Goodfellow, Shlens, Szegedy 2015, arXiv:1412.6572) is a single-step,
L-infinity evasion attack. The whole idea is one line:

    x_adv = clip( x + epsilon * sign( grad_x L(model(x), y) ), 0, 1 )

We nudge every pixel by a fixed step `epsilon` in the direction that *increases*
the model's loss - i.e. toward being wrong. `sign(...)` makes it an L-infinity
attack (each pixel moves by at most epsilon); `clip(..., 0, 1)` keeps the result
a valid image. That's it: the model's own gradient is the weapon.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

_loss_fn = nn.CrossEntropyLoss()


def fgsm_perturb(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    loss_fn: nn.Module = _loss_fn,
) -> torch.Tensor:
    """Return the FGSM adversarial version of a batch `x` with true labels `y`.

    The ~10 lines that matter:
    """
    x = x.clone().detach().requires_grad_(True)  # track grad w.r.t. the INPUT
    loss = loss_fn(model(x), y)
    model.zero_grad(set_to_none=True)
    loss.backward()  # populates x.grad
    x_adv = x + epsilon * x.grad.sign()  # step along the gradient's sign
    x_adv = torch.clamp(x_adv, 0.0, 1.0)  # keep it a valid image
    return x_adv.detach()


@torch.no_grad()
def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    return int((model(x).argmax(1) == y).sum().item())


def accuracy_under_attack(
    model: nn.Module,
    loader: DataLoader,
    epsilons: list[float],
    device: torch.device | None = None,
) -> dict[float, float]:
    """Top-1 accuracy at each epsilon (epsilon=0 is the clean baseline).

    Note we must NOT use torch.no_grad() around the attack itself - FGSM needs the
    input gradient. We only disable grad for the final accuracy count.
    """
    device = device or torch.device("cpu")
    model.to(device).eval()
    results: dict[float, float] = {}
    for eps in epsilons:
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_eval = x if eps == 0 else fgsm_perturb(model, x, y, eps)
            correct += _accuracy(model, x_eval, y)
            total += y.numel()
        results[eps] = correct / max(total, 1)
    return results
