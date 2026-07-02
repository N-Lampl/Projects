"""Inference helpers: summaries, baselines, R-hat, posterior predictive checks.

These turn raw posterior draws into the things an interviewer actually asks
about — credible intervals, convergence diagnostics, and how partial pooling
compares against the two degenerate baselines (no pooling / complete pooling).
"""

from __future__ import annotations

import numpy as np

from .model import GibbsPosterior


def credible_interval(draws: np.ndarray, level: float = 0.90) -> np.ndarray:
    """Equal-tailed central credible interval per column (last axis kept).

    ``draws`` is ``(n_draws,)`` or ``(n_draws, J)``; returns ``(2,)`` or
    ``(J, 2)`` with the lower/upper quantiles.
    """
    alpha = (1.0 - level) / 2.0
    lo = np.quantile(draws, alpha, axis=0)
    hi = np.quantile(draws, 1.0 - alpha, axis=0)
    return np.stack([lo, hi], axis=-1)


def posterior_summary(post: GibbsPosterior, level: float = 0.90) -> dict:
    """Posterior means + credible intervals for theta, mu, tau."""
    theta_ci = credible_interval(post.theta, level)
    return {
        "theta_mean": post.theta.mean(axis=0),
        "theta_ci": theta_ci,
        "mu_mean": float(post.mu.mean()),
        "mu_ci": credible_interval(post.mu, level).tolist(),
        "tau_mean": float(post.tau.mean()),
        "tau_ci": credible_interval(post.tau, level).tolist(),
        "level": level,
    }


def no_pooling(y: np.ndarray) -> np.ndarray:
    """Per-group MLE: each group estimated on its own data only."""
    return np.asarray(y, dtype=float).mean(axis=1)


def complete_pooling(y: np.ndarray) -> np.ndarray:
    """One global mean assigned to every group (ignores group identity)."""
    y = np.asarray(y, dtype=float)
    return np.full(y.shape[0], y.mean())


def rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-squared error against the known truth."""
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


def rhat(chains: np.ndarray) -> np.ndarray:
    """Gelman-Rubin potential scale reduction factor.

    ``chains`` is ``(n_chains, n_draws)`` for a scalar parameter or
    ``(n_chains, n_draws, J)`` for a vector; returns a scalar or ``(J,)``.
    Values near 1.0 (< 1.1) indicate the chains have mixed.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        chains = chains[:, :, None]
        squeeze = True
    else:
        squeeze = False

    m, n = chains.shape[0], chains.shape[1]
    chain_means = chains.mean(axis=1)  # (m, J)
    chain_vars = chains.var(axis=1, ddof=1)  # (m, J)
    grand_mean = chain_means.mean(axis=0)  # (J,)

    between = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    within = chain_vars.mean(axis=0)
    var_hat = (n - 1) / n * within + between / n
    out = np.sqrt(var_hat / np.maximum(within, 1e-12))
    return float(out[0]) if squeeze else out


def posterior_predictive_check(
    post: GibbsPosterior, y: np.ndarray, sigma: np.ndarray, seed: int = 0
) -> dict:
    """Simulate replicated data and compare a test statistic to the observed.

    Uses the per-group standard deviation of observations as the discrepancy.
    A Bayesian p-value near 0.5 means the model reproduces that feature well.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    j, n = y.shape

    # Subsample draws to keep the PPC cheap and deterministic.
    n_rep = min(500, post.theta.shape[0])
    idx = np.linspace(0, post.theta.shape[0] - 1, n_rep).astype(int)
    theta = post.theta[idx]  # (n_rep, J)

    obs_stat = y.std(axis=1)  # (J,)
    y_rep = theta[:, :, None] + sigma[None, :, None] * rng.standard_normal((n_rep, j, n))
    rep_stat = y_rep.std(axis=2)  # (n_rep, J)
    p_values = (rep_stat >= obs_stat[None, :]).mean(axis=0)  # (J,)
    return {
        "bayes_p_values": p_values.tolist(),
        "mean_bayes_p": float(p_values.mean()),
    }
