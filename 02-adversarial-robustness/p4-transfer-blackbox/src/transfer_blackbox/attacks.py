"""Transferability + query-based black-box attacks, hand-rolled.

Two threat models live here:

1. TRANSFER (no target queries).  We have white-box access to a *surrogate*
   model and craft L-infinity PGD adversarials on it, then fire them blind at the
   *target*. If the two models share decision boundaries the examples "transfer".
   PGD (Madry et al. 2018) is iterated FGSM with a projection back into the
   epsilon-ball:

       x_{t+1} = clip_{[0,1]} ( proj_eps ( x_t + alpha * sign(grad_x L) ) )

2. QUERY-BASED BLACK-BOX (target queries, capped budget).  We can only call the
   target and read its output. Two classic score/decision-based attacks, written
   from scratch in numpy-style torch (no ART / foolbox / torchattacks):

   - Square Attack (Andriushchenko et al. 2020): a *score-based* L-infinity attack
     that, each step, flips a random square patch to +-epsilon and keeps the change
     only if the target's margin loss improves. Random search, no gradients.
   - Boundary Attack (Brendel et al. 2018): a *decision-based* L2 attack that needs
     only the predicted label. Start from an already-adversarial point and random-
     walk along the decision boundary, shrinking the distance to the original.

Every query-based attack takes a `query_budget` and stops when it is exhausted.
The "target" is accessed only through a `QueryOracle` that counts calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

_loss_fn = nn.CrossEntropyLoss()


# --------------------------------------------------------------------------- #
# 1. White-box craft on the surrogate (PGD / FGSM) + transfer evaluation       #
# --------------------------------------------------------------------------- #
def pgd_perturb(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    *,
    alpha: float | None = None,
    steps: int = 10,
    loss_fn: nn.Module = _loss_fn,
) -> torch.Tensor:
    """L-infinity PGD (set steps=1, alpha=epsilon for plain FGSM)."""
    if alpha is None:
        alpha = max(epsilon / 4.0, 1e-3) if steps > 1 else epsilon
    x0 = x.clone().detach()
    x_adv = x0.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = loss_fn(model(x_adv), y)
        model.zero_grad(set_to_none=True)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x0 + epsilon), x0 - epsilon)  # project to ball
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = x_adv.detach()
    return x_adv


@torch.no_grad()
def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    return int((model(x).argmax(1) == y).sum().item())


def transfer_accuracy(
    surrogate: nn.Module,
    target: nn.Module,
    loader: DataLoader,
    epsilons: list[float],
    *,
    steps: int = 10,
    device: torch.device | None = None,
) -> dict[str, dict[float, float]]:
    """For each epsilon, craft PGD adversarials on `surrogate` and measure accuracy
    on BOTH the surrogate (white-box upper bound) and the target (transfer).

    Returns {"surrogate": {eps: acc}, "target": {eps: acc}}.
    """
    device = device or torch.device("cpu")
    surrogate.to(device).eval()
    target.to(device).eval()
    out: dict[str, dict[float, float]] = {"surrogate": {}, "target": {}}
    for eps in epsilons:
        s_correct = t_correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_adv = x if eps == 0 else pgd_perturb(surrogate, x, y, eps, steps=steps)
            s_correct += _accuracy(surrogate, x_adv, y)
            t_correct += _accuracy(target, x_adv, y)
            total += y.numel()
        out["surrogate"][eps] = s_correct / max(total, 1)
        out["target"][eps] = t_correct / max(total, 1)
    return out


# --------------------------------------------------------------------------- #
# 2. Query oracle (the only way the black-box attacks touch the target)        #
# --------------------------------------------------------------------------- #
@dataclass
class QueryOracle:
    """Wraps the target so black-box attacks can ONLY query it, with counting.

    `scores(x)` returns logits (score-based access); `labels(x)` returns argmax
    (decision-based access). Both count toward `n_queries`.
    """

    model: nn.Module
    n_queries: int = 0

    @torch.no_grad()
    def scores(self, x: torch.Tensor) -> torch.Tensor:
        self.n_queries += int(x.shape[0])
        return self.model(x)

    @torch.no_grad()
    def labels(self, x: torch.Tensor) -> torch.Tensor:
        return self.scores(x).argmax(1)


def _margin_loss(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Untargeted margin: correct-class logit minus best-other logit. Lower (more
    negative) is better for the attacker; <0 means misclassified."""
    true = scores.gather(1, y[:, None]).squeeze(1)
    tmp = scores.clone()
    tmp.scatter_(1, y[:, None], float("-inf"))
    other = tmp.max(1).values
    return true - other


# --------------------------------------------------------------------------- #
# 3. Square Attack — score-based, L-infinity, random search                    #
# --------------------------------------------------------------------------- #
@dataclass
class AttackResult:
    x_adv: torch.Tensor
    success: torch.Tensor          # bool per sample
    queries_used: int
    queries_per_sample: list[int] = field(default_factory=list)


def square_attack(
    oracle: QueryOracle,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float = 0.3,
    query_budget: int = 200,
    p_init: float = 0.3,
    seed: int = 0,
) -> AttackResult:
    """Hand-rolled L-infinity Square Attack (single image per call, batched over the
    leading dim). Each step perturbs a random square to +-epsilon and keeps it iff
    the margin loss does not increase. Stops at `query_budget` queries per sample.
    """
    g = torch.Generator().manual_seed(seed)
    n, c, h, w = x.shape
    x0 = x.clone()
    # init with vertical-stripe +-epsilon noise (the paper's L-inf init), clipped
    init = (torch.randint(0, 2, (n, c, 1, w), generator=g).float() * 2 - 1) * epsilon
    x_adv = torch.clamp(x0 + init, 0.0, 1.0)

    cur = _margin_loss(oracle.scores(x_adv), y)  # 1 query/sample
    success = cur < 0
    per_sample = [1] * n

    step = 0
    while oracle.n_queries < query_budget * n:
        step += 1
        # schedule the square side length down over the budget (paper-style)
        frac = min(step / 50.0, 1.0)
        p = p_init * (1 - 0.8 * frac)
        s = max(1, int(round((p * h * w) ** 0.5)))
        s = min(s, h, w)

        cand = x_adv.clone()
        for i in range(n):
            if success[i]:
                continue
            r = int(torch.randint(0, h - s + 1, (1,), generator=g))
            cc = int(torch.randint(0, w - s + 1, (1,), generator=g))
            sign = (torch.randint(0, 2, (c, 1, 1), generator=g).float() * 2 - 1) * epsilon
            patch = x0[i, :, r:r + s, cc:cc + s] + sign
            cand[i, :, r:r + s, cc:cc + s] = torch.clamp(patch, 0.0, 1.0)
            # enforce the L-inf ball around x0
            cand[i] = torch.max(torch.min(cand[i], x0[i] + epsilon), x0[i] - epsilon)
            cand[i] = torch.clamp(cand[i], 0.0, 1.0)

        new = _margin_loss(oracle.scores(cand), y)  # 1 query/sample
        improved = new < cur
        for i in range(n):
            if success[i]:
                continue
            per_sample[i] += 1
            if improved[i]:
                x_adv[i] = cand[i]
                cur[i] = new[i]
        success = cur < 0
        if bool(success.all()):
            break

    return AttackResult(x_adv.detach(), success, oracle.n_queries, per_sample)


# --------------------------------------------------------------------------- #
# 4. Boundary Attack — decision-based, L2, random walk                         #
# --------------------------------------------------------------------------- #
def boundary_attack(
    oracle: QueryOracle,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    query_budget: int = 200,
    spherical_step: float = 0.1,
    source_step: float = 0.1,
    seed: int = 0,
) -> AttackResult:
    """Hand-rolled decision-based Boundary Attack (per sample). Needs ONLY the
    predicted label. Starts from random adversarial noise (already misclassified),
    then alternates an orthogonal step (stay adversarial) and a step toward the
    original (shrink L2). Stops at `query_budget` queries per sample.
    """
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    x0 = x.clone()
    x_adv = x0.clone()
    success = torch.zeros(n, dtype=torch.bool)
    per_sample = [0] * n

    # 1) find an adversarial starting point with uniform noise
    for i in range(n):
        for _ in range(50):
            if oracle.n_queries >= query_budget * n:
                break
            cand = torch.clamp(torch.rand(x0[i].shape, generator=g), 0.0, 1.0).unsqueeze(0)
            per_sample[i] += 1
            if int(oracle.labels(cand)) != int(y[i]):
                x_adv[i] = cand[0]
                success[i] = True
                break

    # 2) walk along the boundary, contracting toward x0
    while oracle.n_queries < query_budget * n:
        progressed = False
        for i in range(n):
            if not success[i] or oracle.n_queries >= query_budget * n:
                continue
            progressed = True
            delta = x0[i] - x_adv[i]
            dist = delta.norm() + 1e-12
            # orthogonal perturbation on a sphere around x_adv
            noise = torch.randn(x_adv[i].shape, generator=g)
            noise = noise - (noise * delta).sum() / (dist**2) * delta  # remove radial comp
            noise = noise / (noise.norm() + 1e-12) * spherical_step * dist
            cand = x_adv[i] + noise + source_step * delta  # nudge toward original
            cand = torch.clamp(cand, 0.0, 1.0).unsqueeze(0)
            per_sample[i] += 1
            if int(oracle.labels(cand)) != int(y[i]):
                x_adv[i] = cand[0]  # accept: still adversarial, closer to x0
        if not progressed:
            break

    return AttackResult(x_adv.detach(), success, oracle.n_queries, per_sample)
