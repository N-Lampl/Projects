"""Randomized smoothing certification (Cohen, Rosenfeld, Kolter, 2019).

A base classifier f is wrapped into a *smoothed* classifier

    g(x) = argmax_c  P_{e ~ N(0, sigma^2 I)} [ f(x + e) = c ].

CERTIFY (Algorithm 1 of the paper):
  1. Draw n0 noise samples, take the majority-vote class cA ("selection").
  2. Draw n fresh noise samples, count nA = #{ f(x+e) = cA } ("estimation").
  3. Get a Clopper-Pearson lower confidence bound pA_bar on P(f(x+e)=cA)
     at level (1 - alpha). If pA_bar > 1/2, CERTIFY cA with radius
            R = sigma * Phi^{-1}(pA_bar),
     where Phi^{-1} is the standard-normal inverse CDF (quantile). Otherwise ABSTAIN.

Guarantee: g(x') = cA for every x' with ||x' - x||_2 < R. The radius is exact for
the Gaussian smoothing measure and tight in the worst case.

Only torch + numpy are required. scipy is used (if present) for the Clopper-Pearson
Beta quantile and the normal quantile, with a pure-numpy fallback otherwise.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

ABSTAIN = -1

try:  # optional: scipy gives exact Beta / Normal quantiles
    from scipy.stats import beta as _beta
    from scipy.stats import norm as _norm

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only when scipy is absent
    _HAVE_SCIPY = False


# ----------------------------------------------------------------------------- #
# Statistics: Clopper-Pearson lower bound + standard-normal inverse CDF.
# ----------------------------------------------------------------------------- #
def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson lower (1 - alpha) confidence bound for a binomial
    proportion given k successes in n trials.

    Lower bound = Beta(alpha; k, n - k + 1) inverse-CDF (0 when k == 0).
    """
    if n <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    if _HAVE_SCIPY:
        return float(_beta.ppf(alpha, k, n - k + 1))
    return _beta_ppf_bisect(alpha, k, n - k + 1)


def norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (quantile), Phi^{-1}(p)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    if _HAVE_SCIPY:
        return float(_norm.ppf(p))
    return _norm_ppf_acklam(p)


# --- pure-numpy fallbacks (used only when scipy is unavailable) -------------- #
def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 1e-12) -> float:
    """Continued-fraction expansion for the regularized incomplete beta (NR 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _beta_ppf_bisect(q: float, a: float, b: float, tol: float = 1e-10) -> float:
    """Inverse of the regularized incomplete beta via bisection (Beta quantile)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _betai(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _norm_ppf_acklam(p: float) -> float:
    """Acklam's rational approximation to the standard-normal quantile."""
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ----------------------------------------------------------------------------- #
# The smoothed classifier.
# ----------------------------------------------------------------------------- #
class SmoothedClassifier:
    """Wrap a base nn.Module into a randomized-smoothing certifier."""

    def __init__(
        self,
        base: nn.Module,
        sigma: float,
        num_classes: int = 10,
        device: torch.device | None = None,
    ) -> None:
        self.base = base.eval()
        self.sigma = float(sigma)
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")
        self.base.to(self.device)

    @torch.no_grad()
    def _sample_counts(self, x: torch.Tensor, num: int, batch: int) -> np.ndarray:
        """Monte-Carlo class counts of f(x + e), e ~ N(0, sigma^2 I), over `num` draws."""
        counts = np.zeros(self.num_classes, dtype=np.int64)
        remaining = num
        x = x.to(self.device)
        while remaining > 0:
            this = min(batch, remaining)
            remaining -= this
            rep = x.repeat(this, 1, 1, 1)
            noisy = (rep + self.sigma * torch.randn_like(rep)).clamp(0.0, 1.0)
            preds = self.base(noisy).argmax(1).cpu().numpy()
            counts += np.bincount(preds, minlength=self.num_classes)
        return counts

    @torch.no_grad()
    def certify(
        self,
        x: torch.Tensor,
        n0: int = 100,
        n: int = 1000,
        alpha: float = 0.001,
        batch: int = 200,
    ) -> tuple[int, float]:
        """Cohen Algorithm 1. Returns (predicted_class | ABSTAIN, certified_radius).

        Radius is 0.0 when abstaining.
        """
        counts0 = self._sample_counts(x, n0, batch)
        c_a = int(counts0.argmax())
        counts = self._sample_counts(x, n, batch)
        n_a = int(counts[c_a])
        p_a_lower = clopper_pearson_lower(n_a, n, alpha)
        if p_a_lower > 0.5:
            radius = self.sigma * norm_ppf(p_a_lower)
            return c_a, float(radius)
        return ABSTAIN, 0.0

    @torch.no_grad()
    def predict(self, x: torch.Tensor, n: int = 100, batch: int = 200) -> int:
        """Hard smoothed prediction (no certificate); majority vote over `n` draws."""
        counts = self._sample_counts(x, n, batch)
        return int(counts.argmax())


def certified_accuracy_at(
    radii: np.ndarray, correct: np.ndarray, r: float
) -> float:
    """Fraction of points that are BOTH correctly predicted AND certified to radius >= r.

    Abstentions / wrong predictions have correct=False and never count.
    """
    radii = np.asarray(radii, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    return float(np.mean(correct & (radii >= r)))
