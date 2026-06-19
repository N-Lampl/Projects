"""Drift metrics: Population Stability Index (PSI) and Kolmogorov-Smirnov (KS).

Both compare a *reference* (training-time) distribution against a *current*
(live-traffic) distribution for a single feature. They are the two workhorse
statistics for tabular model monitoring:

- **PSI** bins the reference distribution and measures how much probability mass
  has moved between bins. Industry rule of thumb: <0.1 stable, 0.1-0.25 moderate
  shift, >=0.25 major shift / investigate.
- **KS** is the maximum gap between the two empirical CDFs (a distribution-free,
  binning-free distance). The two-sample KS test also yields a p-value.

The KS p-value uses scipy if installed; otherwise a closed-form asymptotic
approximation keeps the default path dependency-free.
"""

from __future__ import annotations

import numpy as np

# PSI interpretation thresholds (Siddiqi, credit-scoring convention).
PSI_MODERATE = 0.10
PSI_MAJOR = 0.25


def psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Population Stability Index between two 1-D samples.

    PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i)

    Bin edges are quantiles of the *reference* sample so each reference bin holds
    ~equal mass (the standard "deciles of training data" approach). `epsilon`
    floors empty bins to keep the log finite.
    """
    reference = np.asarray(reference, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()
    if reference.size == 0 or current.size == 0:
        raise ValueError("reference and current must be non-empty")

    # Quantile edges from the reference; widen the outer edges to catch tails.
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:  # degenerate (constant) reference feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_frac = ref_counts / ref_counts.sum()
    cur_frac = cur_counts / cur_counts.sum()
    ref_frac = np.clip(ref_frac, epsilon, None)
    cur_frac = np.clip(cur_frac, epsilon, None)

    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max gap between empirical CDFs.

    Returned value is in [0, 1]; 0 == identical empirical distributions.
    """
    reference = np.sort(np.asarray(reference, dtype=float).ravel())
    current = np.sort(np.asarray(current, dtype=float).ravel())
    if reference.size == 0 or current.size == 0:
        raise ValueError("reference and current must be non-empty")

    grid = np.concatenate([reference, current])
    cdf_ref = np.searchsorted(reference, grid, side="right") / reference.size
    cdf_cur = np.searchsorted(current, grid, side="right") / current.size
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def ks_pvalue(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample KS p-value. Uses scipy if available, else asymptotic form."""
    reference = np.asarray(reference, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()
    try:  # scipy is an optional enhancement
        from scipy.stats import ks_2samp

        return float(ks_2samp(reference, current).pvalue)
    except ImportError:
        d = ks_statistic(reference, current)
        n, m = reference.size, current.size
        en = np.sqrt(n * m / (n + m))
        lam = (en + 0.12 + 0.11 / en) * d
        # Kolmogorov distribution survival function (series form).
        j = np.arange(1, 101)
        p = 2.0 * np.sum((-1.0) ** (j - 1) * np.exp(-2.0 * (j**2) * lam**2))
        return float(min(max(p, 0.0), 1.0))


def psi_severity(value: float) -> str:
    """Map a PSI value to a human label."""
    if value < PSI_MODERATE:
        return "stable"
    if value < PSI_MAJOR:
        return "moderate"
    return "major"
