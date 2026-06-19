"""FGSM, from scratch — the source of the adversarial examples we try to detect.

FGSM (Goodfellow, Shlens, Szegedy 2015, arXiv:1412.6572) is a single-step,
L-infinity evasion attack:

    x_adv = clip( x + epsilon * sign( grad_x L(model(x), y) ), 0, 1 )

We only need it here to manufacture a pile of adversarial inputs to train and
evaluate the runtime detector on.
"""

from __future__ import annotations

import torch
from torch import nn

_loss_fn = nn.CrossEntropyLoss()


def fgsm_perturb(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    loss_fn: nn.Module = _loss_fn,
) -> torch.Tensor:
    """Return the FGSM adversarial version of batch `x` with true labels `y`."""
    x = x.clone().detach().requires_grad_(True)  # track grad w.r.t. the INPUT
    loss = loss_fn(model(x), y)
    model.zero_grad(set_to_none=True)
    loss.backward()  # populates x.grad
    x_adv = x + epsilon * x.grad.sign()  # step along the gradient's sign
    x_adv = torch.clamp(x_adv, 0.0, 1.0)  # keep it a valid image
    return x_adv.detach()
