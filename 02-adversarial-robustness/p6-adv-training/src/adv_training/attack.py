"""Projected Gradient Descent (PGD), from scratch — no attack library.

PGD (Madry et al. 2018, arXiv:1706.06083) is the multi-step extension of FGSM
and the standard threat model for L-inf adversarial training. Starting from a
random point inside the epsilon-ball, take `steps` small gradient-sign steps of
size `alpha`, projecting back into the ball (and the valid [0, 1] image range)
after each one:

    x_0   = clip( x + Uniform(-eps, eps), 0, 1 )            # random start
    x_t+1 = clip( x_t + alpha * sign( grad_x L(f(x_t), y) ) )   # ascent step
    x_t+1 = clip( x + clip(x_t+1 - x, -eps, +eps), 0, 1 )   # project to ball & image

The result is a strong first-order adversary. We *attack* with it to measure
robustness and *train against it* (the inner max of Madry's saddle-point) to
build a robust model.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

_loss_fn = nn.CrossEntropyLoss()


def pgd_perturb(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    *,
    alpha: float | None = None,
    steps: int = 7,
    random_start: bool = True,
    loss_fn: nn.Module = _loss_fn,
) -> torch.Tensor:
    """Return the PGD adversarial version of a batch `x` with true labels `y`.

    `alpha` defaults to a sensible 2.5*eps/steps (Madry's rule of thumb).
    Returns a detached tensor (no graph), safe to use as training input.
    """
    if epsilon == 0:
        return x.detach()
    if alpha is None:
        alpha = 2.5 * epsilon / max(steps, 1)

    x_orig = x.detach()
    if random_start:
        noise = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_orig + noise, 0.0, 1.0).detach()
    else:
        x_adv = x_orig.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = loss_fn(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()  # ascent step
            x_adv = x_orig + torch.clamp(x_adv - x_orig, -epsilon, epsilon)  # project to ball
            x_adv = torch.clamp(x_adv, 0.0, 1.0)  # project to image range
        x_adv = x_adv.detach()
    return x_adv


@torch.no_grad()
def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    return int((model(x).argmax(1) == y).sum().item())


def accuracy_under_attack(
    model: nn.Module,
    loader: DataLoader,
    epsilons: list[float],
    *,
    steps: int = 7,
    device: torch.device | None = None,
) -> dict[float, float]:
    """Top-1 accuracy under PGD at each epsilon (epsilon=0 is the clean baseline).

    We must NOT wrap the attack in torch.no_grad() — PGD needs input gradients.
    Only the final accuracy count is grad-free.
    """
    device = device or torch.device("cpu")
    model.to(device).eval()
    results: dict[float, float] = {}
    for eps in epsilons:
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_eval = x if eps == 0 else pgd_perturb(model, x, y, eps, steps=steps)
            correct += _accuracy(model, x_eval, y)
            total += y.numel()
        results[eps] = correct / max(total, 1)
    return results
