"""Inference-time evasion attacks, from scratch — no attack library required.

Both attacks are L-infinity, white-box, and operate on images in [0, 1] (the
target model folds its own normalization in, see model.py). They are identical
in spirit whether the target is our SmallCNN or a pretrained ResNet-18:

    FGSM (single step):
        x_adv = clip( x + eps * sign( grad_x L(f(x), y) ), 0, 1 )

    PGD (multi-step, the iterative cousin):
        repeat:  x <- clip_[x0-eps, x0+eps]( x + alpha * sign(grad_x L) )
                 x <- clip(x, 0, 1)

An OPTIONAL foolbox v3 backend (`fgsm_foolbox`) is provided to cross-check the
hand-rolled result; it is imported lazily so the module loads without foolbox.
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
    """Single-step FGSM adversarial example for batch `x` with true labels `y`."""
    x = x.clone().detach().requires_grad_(True)  # grad w.r.t. the INPUT
    loss = loss_fn(model(x), y)
    model.zero_grad(set_to_none=True)
    loss.backward()  # populates x.grad
    x_adv = x + epsilon * x.grad.sign()  # step along the gradient's sign
    return torch.clamp(x_adv, 0.0, 1.0).detach()  # keep a valid image


def pgd_perturb(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float | None = None,
    steps: int = 10,
    loss_fn: nn.Module = _loss_fn,
) -> torch.Tensor:
    """Iterative PGD (L-inf) — FGSM applied repeatedly, projected back into the
    epsilon-ball around the original image. Usually a much stronger attack."""
    if epsilon == 0:
        return x.clone().detach()
    alpha = alpha if alpha is not None else max(epsilon / 4.0, 1e-3)
    x0 = x.clone().detach()
    # random start inside the eps-ball (the "R" in PGD) for a stronger attack
    x_adv = (x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(model(x_adv), y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x0 + epsilon), x0 - epsilon)  # project
            x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv.detach()


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (predicted_label, softmax_confidence_of_that_label) per image."""
    probs = model(x).softmax(dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf


@torch.no_grad()
def true_label_confidence(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Softmax probability the model assigns to the TRUE class (per image)."""
    probs = model(x).softmax(dim=1)
    return probs.gather(1, y.view(-1, 1)).squeeze(1)


def fgsm_foolbox(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    """OPTIONAL cross-check using foolbox v3. Lazy import; needs `pip install foolbox`."""
    import foolbox as fb  # noqa: F401

    fmodel = fb.PyTorchModel(model.eval(), bounds=(0, 1))
    attack = fb.attacks.LinfFastGradientAttack()
    _, advs, _ = attack(fmodel, x, y, epsilons=epsilon)
    return advs.detach()
