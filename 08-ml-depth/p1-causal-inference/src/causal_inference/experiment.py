"""Tie the SCM + estimators together into the reported experiment.

Two things are measured:

1. **Point estimates** on one dataset — naive is biased, the adjusted estimators
   recover the true ATE.
2. **Confidence-interval coverage** — repeat over many seeds and check how often
   each interval contains the truth. The doubly-robust AIPW interval should cover
   at close to its nominal 95%, while the naive "interval" almost never does.
"""

from __future__ import annotations

import numpy as np

from .estimators import (
    Z_95,
    aipw,
    all_estimators,
    estimate_propensity,
    naive_diff,
    standardized_mean_diff,
)
from .scm import make_scm


def point_estimates(n: int, p: int, tau: float, confounding: float, seed: int) -> dict:
    """All four estimates + their bias against the known ATE, on one draw."""
    scm = make_scm(n=n, p=p, tau=tau, confounding=confounding, seed=seed)
    results = all_estimators(scm.X, scm.T, scm.Y)
    estimates = {k: r.estimate for k, r in results.items()}
    bias = {k: v - scm.true_ate for k, v in estimates.items()}
    aipw_res = results["aipw"]
    return {
        "true_ate": scm.true_ate,
        "estimates": estimates,
        "bias": bias,
        "aipw_se": aipw_res.se,
        "aipw_ci": list(aipw_res.ci) if aipw_res.ci else None,
    }


def _naive_se(Y: np.ndarray, T: np.ndarray) -> float:
    """Two-sample SE for the naive difference (its own best-case interval)."""
    yt, yc = Y[T == 1], Y[T == 0]
    return float(np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc)))


def coverage_study(
    n: int, p: int, tau: float, confounding: float, n_sims: int, base_seed: int
) -> dict:
    """Fraction of 95% intervals covering the truth, for naive vs AIPW."""
    naive_hits, aipw_hits = 0, 0
    for i in range(n_sims):
        scm = make_scm(n=n, p=p, tau=tau, confounding=confounding, seed=base_seed + i)
        # Naive interval (best case: correct SE, but a biased centre).
        n_est = naive_diff(scm.X, scm.T, scm.Y).estimate
        n_se = _naive_se(scm.Y, scm.T)
        if abs(n_est - scm.true_ate) <= Z_95 * n_se:
            naive_hits += 1
        # Doubly-robust interval.
        a = aipw(scm.X, scm.T, scm.Y)
        lo, hi = a.ci
        if lo <= scm.true_ate <= hi:
            aipw_hits += 1
    return {
        "n_sims": n_sims,
        "naive_coverage": naive_hits / n_sims,
        "aipw_coverage": aipw_hits / n_sims,
    }


def balance_table(n: int, p: int, tau: float, confounding: float, seed: int) -> dict:
    """Per-covariate SMD before vs after inverse-propensity weighting."""
    scm = make_scm(n=n, p=p, tau=tau, confounding=confounding, seed=seed)
    e = estimate_propensity(scm.X, scm.T)
    weights = np.where(scm.T == 1, 1.0 / e, 1.0 / (1.0 - e))
    before = standardized_mean_diff(scm.X, scm.T)
    after = standardized_mean_diff(scm.X, scm.T, weights=weights)
    return {
        "smd_before": before.tolist(),
        "smd_after": after.tolist(),
        "mean_abs_smd_before": float(np.abs(before).mean()),
        "mean_abs_smd_after": float(np.abs(after).mean()),
    }
