"""An attack zoo, implemented from scratch — no attack library required.

Three classic white-box evasion attacks, each one function:

  pgd      -- Projected Gradient Descent (Madry et al. 2018), the multi-step
              L-infinity attack. Iterated FGSM with projection back into the
              epsilon-ball plus a random start.  arXiv:1706.06083

  cw_l2    -- A Carlini & Wagner L2 attack (2017), the gold-standard L2 attack.
              Minimizes ||delta||_2 + c * f(x+delta) over a change-of-variables
              that keeps the image in [0,1] without clipping.  arXiv:1608.04644

  deepfool -- DeepFool (Moosavi-Dezfooli et al. 2016), a minimal-L2 attack that
              walks the input to the nearest decision boundary via a local linear
              approximation.  arXiv:1511.04599

All operate on a batch x in [0,1] with true labels y and return x_adv in [0,1].
An OPTIONAL torchattacks backend is available via `pgd(..., backend="torchattacks")`
(imported lazily; the module imports fine without torchattacks installed).
"""

from __future__ import annotations

import torch
from torch import nn

_loss_fn = nn.CrossEntropyLoss()


# --------------------------------------------------------------------------- #
# PGD  (L-infinity)                                                            #
# --------------------------------------------------------------------------- #
def pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float = 0.03,
    alpha: float | None = None,
    steps: int = 20,
    random_start: bool = True,
    loss_fn: nn.Module = _loss_fn,
    backend: str = "scratch",
) -> torch.Tensor:
    """Projected Gradient Descent, L-infinity. Iterated FGSM with projection.

    delta_{t+1} = clip_eps( delta_t + alpha * sign(grad_x L) ),  x+delta in [0,1]
    """
    if backend == "torchattacks":  # optional enhanced path
        import torchattacks  # lazy import

        atk = torchattacks.PGD(
            model, eps=epsilon, alpha=alpha or epsilon / 4, steps=steps,
            random_start=random_start,
        )
        return atk(x, y).detach()

    if alpha is None:
        alpha = max(epsilon / 4, 1e-3)
    model.eval()
    x = x.detach()
    x_adv = x.clone()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = x_adv.clamp(0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = loss_fn(model(x_adv), y)
        model.zero_grad(set_to_none=True)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # project back into the epsilon L-inf ball, then into the image range
        delta = (x_adv - x).clamp(-epsilon, epsilon)
        x_adv = (x + delta).clamp(0.0, 1.0)
    return x_adv.detach()


# --------------------------------------------------------------------------- #
# Carlini & Wagner  (L2)                                                       #
# --------------------------------------------------------------------------- #
def cw_l2(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    c: float = 5.0,
    kappa: float = 0.0,
    steps: int = 100,
    lr: float = 0.01,
    loss_fn: nn.Module | None = None,  # unused; kept for a uniform signature
) -> torch.Tensor:
    """Carlini & Wagner L2 attack (untargeted).

    Optimize w (unconstrained) with x_adv = 0.5*(tanh(w)+1) so x_adv is always a
    valid image, minimizing:  ||x_adv - x||_2^2  +  c * f(x_adv)
    where f pushes the true-class logit below the best other logit by margin kappa.
    """
    model.eval()
    x = x.detach()
    # inverse-tanh of the clean image gives a w whose initial x_adv == x
    x_clamped = x.clamp(1e-6, 1 - 1e-6)
    w = torch.atanh(2 * x_clamped - 1).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    one_hot = torch.zeros(x.shape[0], model(x).shape[1], device=x.device)
    one_hot.scatter_(1, y.view(-1, 1), 1.0)

    best_adv = x.clone()
    best_l2 = torch.full((x.shape[0],), float("inf"), device=x.device)

    for _ in range(steps):
        x_adv = 0.5 * (torch.tanh(w) + 1)
        logits = model(x_adv)
        real = (one_hot * logits).sum(1)
        other = ((1 - one_hot) * logits - one_hot * 1e4).max(1).values
        # untargeted: want other > real  =>  penalize real - other
        f = torch.clamp(real - other + kappa, min=0.0)
        l2 = ((x_adv - x) ** 2).flatten(1).sum(1)
        loss = (l2 + c * f).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            succeeded = logits.argmax(1) != y
            improved = succeeded & (l2 < best_l2)
            best_l2 = torch.where(improved, l2, best_l2)
            best_adv[improved] = x_adv[improved].detach()
    return best_adv.detach()


# --------------------------------------------------------------------------- #
# DeepFool  (L2, minimal perturbation)                                         #
# --------------------------------------------------------------------------- #
def deepfool(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int = 50,
    overshoot: float = 0.02,
    num_classes: int | None = None,
    loss_fn: nn.Module | None = None,  # unused; uniform signature
) -> torch.Tensor:
    """DeepFool (untargeted), per-sample, vectorized over the batch.

    At each step we linearize every class boundary around the current point and
    step toward the nearest one. `overshoot` nudges past the boundary so the
    flip sticks under float error.
    """
    model.eval()
    x = x.detach()
    n_classes = num_classes or model(x).shape[1]
    x_adv = x.clone()
    active = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    for _ in range(steps):
        if not active.any():
            break
        x_cur = x_adv.detach().requires_grad_(True)
        logits = model(x_cur)
        preds = logits.argmax(1)
        active = preds == y
        if not active.any():
            break

        # gradient of each logit w.r.t. input, for the active samples
        idx = active.nonzero(as_tuple=True)[0]
        xi = x_cur[idx]  # already requires grad via x_cur
        # Recompute logits restricted to active rows for clean per-row grads.
        grads = []
        for k in range(n_classes):
            g = torch.autograd.grad(
                logits[idx, k].sum(), x_cur, retain_graph=True, create_graph=False
            )[0][idx]
            grads.append(g)
        grads = torch.stack(grads, dim=1)  # [B, K, C, H, W]

        f = logits[idx]  # [B, K]
        y_act = y[idx]
        f_orig = f.gather(1, y_act.view(-1, 1))  # [B, 1]
        g_orig = grads[torch.arange(idx.shape[0]), y_act]  # [B, C,H,W]

        w = grads - g_orig.unsqueeze(1)  # [B, K, ...]
        fk = f - f_orig  # [B, K]
        w_flat = w.flatten(2)  # [B, K, D]
        w_norm = w_flat.norm(dim=2) + 1e-8  # [B, K]
        dist = fk.abs() / w_norm  # [B, K]
        dist[torch.arange(idx.shape[0]), y_act] = float("inf")  # skip true class
        l = dist.argmin(1)  # nearest boundary

        rows = torch.arange(idx.shape[0])
        w_l = w[rows, l]  # [B, ...]
        w_l_norm = w_norm[rows, l].view(-1, *([1] * (x.dim() - 1)))
        fk_l = fk[rows, l].view(-1, *([1] * (x.dim() - 1)))
        r = (fk_l.abs() / (w_l_norm ** 2)) * w_l  # minimal step toward boundary

        x_adv = x_adv.detach()
        x_adv[idx] = (x_adv[idx] + (1 + overshoot) * r).clamp(0.0, 1.0)
        _ = xi  # keep reference; silence linters

    return x_adv.detach()
