"""A structural causal model (SCM) with a *known* treatment effect.

The whole point of a causal-inference demo is that you know the right answer. We
generate confounders ``X``, a treatment ``T`` whose probability depends on ``X``
(so ``X`` is a genuine confounder), and an outcome ``Y`` that also depends on
``X``. The treatment adds a constant ``tau`` to ``Y``, so the true **ATE = tau**.

Because ``T`` and ``Y`` share the common cause ``X``, the naive treated-minus-
control difference is *confounded* and does not recover ``tau`` — adjusting for
``X`` (regression / IPW / doubly-robust) is what fixes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SCM:
    """A realized draw from the confounded SCM plus its ground-truth effect."""

    X: np.ndarray  # (n, p) confounders
    T: np.ndarray  # (n,) binary treatment
    Y: np.ndarray  # (n,) observed outcome
    propensity: np.ndarray  # (n,) true P(T=1 | X)
    true_ate: float  # the constant treatment effect tau


def make_scm(
    n: int = 3000,
    p: int = 5,
    tau: float = 2.0,
    confounding: float = 1.5,
    noise: float = 1.0,
    seed: int = 42,
) -> SCM:
    """Draw ``n`` samples from a linear-logistic confounded SCM.

    Parameters
    ----------
    n, p : sample size and number of confounders.
    tau : the constant treatment effect (== the true ATE).
    confounding : scales how strongly ``X`` drives *both* treatment and outcome;
        larger values make the naive estimate more biased.
    noise : outcome noise standard deviation.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))

    # Both coefficient vectors point the *same* way, so the confounding does not
    # cancel: treated units systematically differ in X, and that same X raises Y.
    # Treatment coefficients are kept modest to preserve overlap (no near-0/1
    # propensities); outcome coefficients are stronger to make the bias clear.
    alpha = confounding * rng.uniform(0.3, 0.6, size=p)
    beta = confounding * rng.uniform(0.8, 1.5, size=p)

    logits = X @ alpha
    propensity = _sigmoid(logits)
    T = (rng.uniform(size=n) < propensity).astype(float)

    # Outcome depends on the SAME X (plus the treatment effect tau).
    Y = X @ beta + tau * T + rng.normal(scale=noise, size=n)

    return SCM(X=X, T=T, Y=Y, propensity=propensity, true_ate=float(tau))
