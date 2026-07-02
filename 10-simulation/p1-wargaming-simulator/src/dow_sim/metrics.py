"""Statistical helpers for the Monte Carlo analysis."""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    More reliable than the normal approximation near 0 and 1, which matters for lopsided
    matchups. Returns ``(low, high)``; a degenerate ``n == 0`` yields ``(0, 1)``.
    """
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def distribution_stats(values: list[float] | np.ndarray) -> dict[str, float]:
    """Mean, std, and the 10th/50th/90th percentiles of a sample."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }
