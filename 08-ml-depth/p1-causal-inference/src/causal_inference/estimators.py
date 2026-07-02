"""Four ATE estimators, from naive to doubly-robust.

All operate on ``(X, T, Y)`` numpy arrays and return an :class:`ATEResult`.

- **naive** — treated mean minus control mean. Confounded: wrong on purpose.
- **regression adjustment (G-computation)** — fit outcome models ``mu0, mu1`` and
  average their difference over the sample. Unbiased *if the outcome model is
  right*.
- **IPW** — inverse-propensity weighting. Unbiased *if the propensity model is
  right*.
- **AIPW (doubly robust)** — combines both; consistent if *either* the outcome or
  the propensity model is right, and it comes with an influence-function standard
  error, so we get a confidence interval and can measure its coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

Z_95 = 1.959963984540054  # normal quantile for a 95% CI


@dataclass
class ATEResult:
    method: str
    estimate: float
    se: float | None = None  # standard error, when available

    @property
    def ci(self) -> tuple[float, float] | None:
        if self.se is None:
            return None
        return (self.estimate - Z_95 * self.se, self.estimate + Z_95 * self.se)


def naive_diff(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> ATEResult:
    """Treated mean minus control mean — ignores confounding."""
    est = Y[T == 1].mean() - Y[T == 0].mean()
    return ATEResult("naive", float(est))


def _fit_outcome_models(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sample predictions ``(mu0, mu1)`` from two outcome regressions."""
    m0 = LinearRegression().fit(X[T == 0], Y[T == 0])
    m1 = LinearRegression().fit(X[T == 1], Y[T == 1])
    return m0.predict(X), m1.predict(X)


def regression_adjustment(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> ATEResult:
    """G-computation: average the modeled individual treatment effect."""
    mu0, mu1 = _fit_outcome_models(X, T, Y)
    return ATEResult("regression", float(np.mean(mu1 - mu0)))


def estimate_propensity(X: np.ndarray, T: np.ndarray, clip: float = 0.02) -> np.ndarray:
    """Logistic P(T=1 | X), clipped away from 0/1 to keep weights finite."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, T)
    e = model.predict_proba(X)[:, 1]
    return np.clip(e, clip, 1.0 - clip)


def ipw(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> ATEResult:
    """Inverse-propensity-weighted difference in outcomes."""
    e = estimate_propensity(X, T)
    est = np.mean(T * Y / e - (1 - T) * Y / (1 - e))
    return ATEResult("ipw", float(est))


def aipw(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> ATEResult:
    """Augmented IPW (doubly robust) with an influence-function standard error."""
    mu0, mu1 = _fit_outcome_models(X, T, Y)
    e = estimate_propensity(X, T)
    # Per-sample influence (efficient influence function of the ATE).
    psi = (mu1 - mu0) + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
    est = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(len(psi)))
    return ATEResult("aipw", est, se)


def all_estimators(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> dict[str, ATEResult]:
    """Run every estimator and return them keyed by method name."""
    return {
        r.method: r
        for r in (
            naive_diff(X, T, Y),
            regression_adjustment(X, T, Y),
            ipw(X, T, Y),
            aipw(X, T, Y),
        )
    }


def standardized_mean_diff(
    X: np.ndarray, T: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Per-covariate standardized mean difference (SMD) between treated/control.

    With ``weights`` (e.g. inverse-propensity), returns the *weighted* SMD — the
    standard "love plot" diagnostic for whether adjustment balanced the groups.
    """
    treated, control = T == 1, T == 0

    def _wmean(a: np.ndarray, m: np.ndarray) -> np.ndarray:
        w = np.ones(m.sum()) if weights is None else weights[m]
        return np.average(a[m], axis=0, weights=w)

    m1, m0 = _wmean(X, treated), _wmean(X, control)
    # Pool the *unweighted* group variances for the denominator (convention).
    pooled_sd = np.sqrt((X[treated].var(axis=0, ddof=1) + X[control].var(axis=0, ddof=1)) / 2)
    pooled_sd = np.where(pooled_sd == 0, 1.0, pooled_sd)
    return (m1 - m0) / pooled_sd
