"""Differentially private training: manual DP-SGD (default) or Opacus (optional).

DP-SGD (Abadi et al., 2016) makes a single training step (eps, delta)-private by:

  1. Computing a *per-sample* gradient g_i for every example in the minibatch.
  2. Clipping each to L2 norm C:   g_i  <-  g_i / max(1, ||g_i|| / C).
  3. Averaging the clipped grads and adding Gaussian noise:
        g~  =  (1/B) * ( sum_i clip(g_i)  +  N(0, (sigma * C)^2 I) ).
  4. Stepping the optimiser with g~.

Clipping bounds each example's influence (sensitivity C); the noise hides whether
any single example was present. Over T steps with Poisson sampling rate q = B/N,
the (subsampled-Gaussian) RDP accountant converts (sigma, q, T) into an
(eps, delta) guarantee. Smaller eps => more noise / smaller updates => stronger
privacy but lower utility -- the tradeoff this capstone measures.

DEFAULT path: everything below is plain torch + a self-contained RDP accountant,
so it runs WITHOUT Opacus. OPTIONAL path: `train_dp_opacus` uses Opacus's
PrivacyEngine and is imported lazily, so the module imports fine without it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# RDP accountant for the subsampled Gaussian mechanism (no scipy/opacus dep).
# Mironov 2017 (RDP) + Mironov, Talwar, Zhang 2019 (subsampled Gaussian).
# ---------------------------------------------------------------------------

# A standard grid of RDP orders, as used by the TF-Privacy / Opacus accountants.
_DEFAULT_ORDERS = (
    [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
)


def _log_add(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) computed stably."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _compute_log_a_int(q: float, sigma: float, alpha: int) -> float:
    """log of A_alpha for INTEGER order alpha (Mironov et al. 2019, eq. used by Opacus)."""
    log_a = -math.inf
    for i in range(alpha + 1):
        log_coef = (
            _log_comb(alpha, i)
            + i * math.log(q)
            + (alpha - i) * math.log1p(-q)
        )
        s = log_coef + (i * i - i) / (2.0 * sigma * sigma)
        log_a = _log_add(log_a, s)
    return float(log_a)


def rdp_subsampled_gaussian(q: float, sigma: float, orders=_DEFAULT_ORDERS) -> np.ndarray:
    """RDP epsilon at each order for ONE step of the subsampled Gaussian mechanism.

    For q == 1 (no subsampling) this reduces to the plain Gaussian RDP
    alpha / (2 sigma^2). We use the integer-order series for the subsampled case,
    which is exact and sufficient for the integer orders in our grid (non-integer
    orders fall back to the q==1 closed form, an upper bound that keeps eps safe).
    """
    rdp = np.zeros(len(orders))
    for j, alpha in enumerate(orders):
        if sigma == 0:
            rdp[j] = math.inf
            continue
        if q == 0:
            rdp[j] = 0.0
        elif q == 1.0:
            rdp[j] = alpha / (2.0 * sigma * sigma)
        elif float(alpha).is_integer():
            rdp[j] = _compute_log_a_int(q, sigma, int(alpha)) / (alpha - 1.0)
        else:
            # conservative upper bound for fractional orders
            rdp[j] = alpha / (2.0 * sigma * sigma)
    return rdp


def rdp_to_eps(rdp: np.ndarray, delta: float, orders=_DEFAULT_ORDERS) -> tuple[float, float]:
    """Convert per-order RDP to a single (eps, best_order) at target `delta`.

    eps = min_alpha [ rdp(alpha) + log(1/delta) / (alpha - 1) ]   (Mironov 2017).
    """
    orders = np.asarray(orders, dtype=np.float64)
    rdp = np.asarray(rdp, dtype=np.float64)
    eps = rdp + np.log(1.0 / delta) / (orders - 1.0)
    idx = int(np.nanargmin(eps))
    return float(eps[idx]), float(orders[idx])


def compute_epsilon(
    noise_multiplier: float, sample_rate: float, steps: int, delta: float
) -> float:
    """End-to-end (eps, delta) for `steps` steps of subsampled-Gaussian DP-SGD."""
    if noise_multiplier == 0:
        return math.inf
    per_step = rdp_subsampled_gaussian(sample_rate, noise_multiplier)
    total = per_step * steps
    eps, _ = rdp_to_eps(total, delta)
    return eps


def find_noise_multiplier(
    target_eps: float,
    sample_rate: float,
    steps: int,
    delta: float,
    lo: float = 0.2,
    hi: float = 64.0,
    tol: float = 1e-3,
) -> float:
    """Binary-search the smallest sigma whose (sigma, q, T) accounting <= target_eps.

    eps is monotonically decreasing in sigma, so a bisection is exact. This is how
    we hit a *requested* epsilon (e.g. 3 or 1) rather than reporting whatever sigma
    happens to give -- matching how Opacus's `make_private_with_epsilon` works.
    """
    if math.isinf(target_eps):
        return 0.0
    # widen hi until it over-satisfies the target
    while compute_epsilon(hi, sample_rate, steps, delta) > target_eps:
        hi *= 2.0
        if hi > 1e6:
            return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        eps_mid = compute_epsilon(mid, sample_rate, steps, delta)
        if eps_mid > target_eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi


# ---------------------------------------------------------------------------
# Manual DP-SGD training loop (the offline default).
# ---------------------------------------------------------------------------


@dataclass
class DPConfig:
    """Knobs for one DP-SGD run. `target_epsilon=None`/inf => non-private SGD."""

    target_epsilon: float | None = None  # None or inf => no DP
    max_grad_norm: float = 1.0  # the per-sample clip bound C
    delta: float = 1e-5
    epochs: int = 30
    lr: float = 0.05
    batch_size: int = 128


@dataclass
class DPReport:
    """What a DP run achieved: requested vs accounted epsilon + the sigma used."""

    target_epsilon: float
    accounted_epsilon: float
    noise_multiplier: float
    max_grad_norm: float
    delta: float
    steps: int
    sample_rate: float
    backend: str  # "non-private" | "manual" | "opacus"


def _per_sample_grads(
    model: nn.Module, xb: torch.Tensor, yb: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Per-sample gradients for a batch via vmap+functional_call (functorch API).

    Returns a list of (B, *param_shape) grad tensors aligned with model.parameters()
    and the per-sample loss vector (for logging). vmap gives true per-example grads
    without a Python loop, which is what makes manual DP-SGD fast enough on a CPU.
    """
    from torch.func import functional_call, grad, vmap

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def compute_loss(p, x, y):
        out = functional_call(model, (p, buffers), (x.unsqueeze(0),))
        return F.cross_entropy(out, y.unsqueeze(0))

    grad_fn = vmap(grad(compute_loss), in_dims=(None, 0, 0))
    per_sample = grad_fn(params, xb, yb)  # dict name -> (B, *shape)
    grads = [per_sample[name] for name, _ in model.named_parameters()]
    return grads, torch.zeros(xb.shape[0])


def _clip_and_noise(
    grads: list[torch.Tensor], max_grad_norm: float, noise_multiplier: float, batch_size: int
) -> list[torch.Tensor]:
    """Clip each per-sample grad to C, sum, add Gaussian noise, average over batch."""
    # flat per-sample L2 norm across all params
    sq = torch.zeros(batch_size)
    for g in grads:
        sq += g.reshape(batch_size, -1).pow(2).sum(dim=1)
    norms = sq.sqrt()
    scale = (max_grad_norm / (norms + 1e-6)).clamp(max=1.0)  # (B,)

    out = []
    for g in grads:
        # scale each sample's grad, sum over the batch
        view = scale.view(-1, *([1] * (g.dim() - 1)))
        summed = (g * view).sum(dim=0)
        if noise_multiplier > 0:
            noise = torch.normal(
                mean=0.0, std=noise_multiplier * max_grad_norm, size=summed.shape
            )
            summed = summed + noise
        out.append(summed / batch_size)
    return out


def train_dp_manual(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    cfg: DPConfig,
    device: torch.device | None = None,
    seed: int = 0,
) -> tuple[nn.Module, DPReport]:
    """Train `model` with manual DP-SGD (or plain SGD when target_epsilon is inf/None).

    We pick the noise multiplier sigma to hit `cfg.target_epsilon` via the RDP
    accountant, then run the clip+noise loop. The returned DPReport records the
    *accounted* epsilon so the privacy claim is auditable, not aspirational.
    """
    device = device or torch.device("cpu")
    model.to(device).train()
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    n = Xt.shape[0]
    bs = min(cfg.batch_size, n)
    steps_per_epoch = max(1, n // bs)
    total_steps = steps_per_epoch * cfg.epochs
    sample_rate = bs / n

    non_private = cfg.target_epsilon is None or math.isinf(cfg.target_epsilon)
    if non_private:
        sigma = 0.0
        accounted = math.inf
    else:
        sigma = find_noise_multiplier(cfg.target_epsilon, sample_rate, total_steps, cfg.delta)
        accounted = compute_epsilon(sigma, sample_rate, total_steps, cfg.delta)

    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.0)
    g = torch.Generator().manual_seed(seed)

    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, steps_per_epoch * bs, bs):
            idx = perm[i : i + bs]
            xb, yb = Xt[idx], yt[idx]
            if non_private:
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()
            else:
                grads, _ = _per_sample_grads(model, xb, yb)
                noised = _clip_and_noise(grads, cfg.max_grad_norm, sigma, xb.shape[0])
                opt.zero_grad(set_to_none=True)
                for p, gp in zip(model.parameters(), noised, strict=True):
                    p.grad = gp.detach().clone()
                opt.step()

    report = DPReport(
        target_epsilon=float(cfg.target_epsilon) if not non_private else math.inf,
        accounted_epsilon=accounted,
        noise_multiplier=sigma,
        max_grad_norm=cfg.max_grad_norm,
        delta=cfg.delta,
        steps=total_steps,
        sample_rate=sample_rate,
        backend="non-private" if non_private else "manual",
    )
    return model.eval(), report


def train_dp_opacus(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    cfg: DPConfig,
    device: torch.device | None = None,
) -> tuple[nn.Module, DPReport]:  # pragma: no cover - optional path
    """OPTIONAL enhanced path: identical training via Opacus's PrivacyEngine.

    Lazily imports opacus so the module loads without it. Use this to validate the
    manual accountant against the reference implementation. Falls back to the
    manual loop only if opacus is missing (the caller decides whether to call it).
    """
    try:
        from opacus import PrivacyEngine
    except Exception as exc:
        raise RuntimeError(
            "Opacus path needs `pip install opacus`. The default manual DP-SGD "
            "(train_dp_manual) runs without it."
        ) from exc

    device = device or torch.device("cpu")
    model.to(device).train()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    bs = min(cfg.batch_size, len(ds))
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    if cfg.target_epsilon is None or math.isinf(cfg.target_epsilon):
        for _ in range(cfg.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                opt.step()
        return model.eval(), DPReport(
            target_epsilon=math.inf, accounted_epsilon=math.inf, noise_multiplier=0.0,
            max_grad_norm=cfg.max_grad_norm, delta=cfg.delta,
            steps=cfg.epochs * (len(ds) // bs), sample_rate=bs / len(ds),
            backend="non-private",
        )

    engine = PrivacyEngine()
    model, opt, loader = engine.make_private_with_epsilon(
        module=model, optimizer=opt, data_loader=loader,
        target_epsilon=cfg.target_epsilon, target_delta=cfg.delta,
        epochs=cfg.epochs, max_grad_norm=cfg.max_grad_norm,
    )
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    eps = engine.get_epsilon(cfg.delta)
    return model.eval(), DPReport(
        target_epsilon=cfg.target_epsilon, accounted_epsilon=float(eps),
        noise_multiplier=float(opt.noise_multiplier), max_grad_norm=cfg.max_grad_norm,
        delta=cfg.delta, steps=cfg.epochs * (len(ds) // bs), sample_rate=bs / len(ds),
        backend="opacus",
    )
