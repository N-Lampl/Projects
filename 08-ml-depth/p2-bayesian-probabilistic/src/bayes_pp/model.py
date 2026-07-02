"""A hierarchical normal model with a from-scratch numpy Gibbs sampler.

Model (partial pooling)::

    mu          ~ N(mu0, gamma0^2)            # global mean, weak prior
    tau^2       ~ Inv-Gamma(a_tau, b_tau)     # between-group variance
    theta_j     ~ N(mu, tau^2)                # true mean of group j
    y_ij        ~ N(theta_j, sigma_j^2)       # observations (sigma known)

Every full conditional is conjugate, so the sampler is a plain Gibbs sweep with
closed-form Normal / Inverse-Gamma draws — no autodiff, no PyMC on the fast
path. Given a seed the whole chain is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GibbsPosterior:
    """Posterior draws from the hierarchical model (chains stacked on axis 0)."""

    theta: np.ndarray  # (n_draws, J) group means
    mu: np.ndarray  # (n_draws,) global mean
    tau: np.ndarray  # (n_draws,) between-group sd
    chains_theta: np.ndarray  # (n_chains, n_kept, J) for R-hat
    chains_mu: np.ndarray  # (n_chains, n_kept)
    chains_tau: np.ndarray  # (n_chains, n_kept)


def _sample_one_chain(
    ybar: np.ndarray,
    n_j: np.ndarray,
    sigma: np.ndarray,
    n_iter: int,
    burn_in: int,
    thin: int,
    mu0: float,
    gamma0_sq: float,
    a_tau: float,
    b_tau: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one Gibbs chain, returning kept (theta, mu, tau) draws."""
    j = ybar.shape[0]
    sigma_sq = sigma**2
    # Initialise from the data so chains mix fast but from different starts.
    theta = ybar + rng.standard_normal(j)
    mu = float(ybar.mean() + rng.standard_normal())
    tau_sq = float(max(ybar.var(), 1.0))

    kept_theta, kept_mu, kept_tau = [], [], []
    for it in range(n_iter):
        # --- theta_j | rest : conjugate Normal (precision-weighted mean) ---
        prec_data = n_j / sigma_sq
        prec_prior = 1.0 / tau_sq
        post_var = 1.0 / (prec_data + prec_prior)
        post_mean = post_var * (prec_data * ybar + prec_prior * mu)
        theta = post_mean + np.sqrt(post_var) * rng.standard_normal(j)

        # --- mu | rest : conjugate Normal ---
        prec_mu = j / tau_sq + 1.0 / gamma0_sq
        mean_mu = (theta.sum() / tau_sq + mu0 / gamma0_sq) / prec_mu
        mu = float(mean_mu + np.sqrt(1.0 / prec_mu) * rng.standard_normal())

        # --- tau^2 | rest : conjugate Inverse-Gamma ---
        a_post = a_tau + j / 2.0
        b_post = b_tau + 0.5 * float(np.sum((theta - mu) ** 2))
        tau_sq = float(b_post / rng.gamma(shape=a_post, scale=1.0))

        if it >= burn_in and (it - burn_in) % thin == 0:
            kept_theta.append(theta.copy())
            kept_mu.append(mu)
            kept_tau.append(np.sqrt(tau_sq))

    return (
        np.asarray(kept_theta),
        np.asarray(kept_mu),
        np.asarray(kept_tau),
    )


def gibbs_sampler(
    y: np.ndarray,
    sigma: np.ndarray,
    n_iter: int = 4000,
    burn_in: int = 1000,
    thin: int = 1,
    n_chains: int = 2,
    mu0: float = 0.0,
    gamma0_sq: float = 1.0e4,
    a_tau: float = 2.0,
    b_tau: float = 5.0,
    seed: int = 0,
) -> GibbsPosterior:
    """Sample the hierarchical-normal posterior with a conjugate Gibbs sweep.

    ``y`` is ``(J, n)`` observations; ``sigma`` is the ``(J,)`` known obs sd.
    Runs ``n_chains`` independent chains (different seeds) and returns both the
    pooled draws and the per-chain draws (used for R-hat).
    """
    y = np.asarray(y, dtype=float)
    ybar = y.mean(axis=1)
    n_j = np.full(y.shape[0], y.shape[1], dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    chains_theta, chains_mu, chains_tau = [], [], []
    for c in range(n_chains):
        rng = np.random.default_rng(seed + 1000 * c)
        t, m, ta = _sample_one_chain(
            ybar, n_j, sigma, n_iter, burn_in, thin, mu0, gamma0_sq, a_tau, b_tau, rng
        )
        chains_theta.append(t)
        chains_mu.append(m)
        chains_tau.append(ta)

    chains_theta = np.stack(chains_theta)  # (n_chains, n_kept, J)
    chains_mu = np.stack(chains_mu)  # (n_chains, n_kept)
    chains_tau = np.stack(chains_tau)  # (n_chains, n_kept)

    return GibbsPosterior(
        theta=chains_theta.reshape(-1, chains_theta.shape[-1]),
        mu=chains_mu.reshape(-1),
        tau=chains_tau.reshape(-1),
        chains_theta=chains_theta,
        chains_mu=chains_mu,
        chains_tau=chains_tau,
    )
