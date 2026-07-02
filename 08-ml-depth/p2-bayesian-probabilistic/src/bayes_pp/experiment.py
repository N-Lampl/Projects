"""Tie the model + inference together into the reported experiment.

Three things are measured:

1. **Recovery** — on one dataset with known group means, the partial-pooling
   posterior means land near the truth and the chains mix (R-hat ~ 1).
2. **Shrinkage** — partial pooling beats both degenerate baselines (no pooling /
   complete pooling) in RMSE against the true group means.
3. **Calibration** — across many datasets, an ``x``% credible interval contains
   the truth about ``x``% of the time (the interval is honest).
"""

from __future__ import annotations

import numpy as np

from .data import make_hierarchical
from .inference import (
    complete_pooling,
    credible_interval,
    no_pooling,
    posterior_summary,
    rhat,
    rmse,
)
from .model import gibbs_sampler


def fit_dataset(ds, n_iter: int = 4000, burn_in: int = 1000, n_chains: int = 2, seed: int = 0):
    """Run the Gibbs sampler on a dataset and return (posterior, summary)."""
    post = gibbs_sampler(
        ds.y, ds.sigma, n_iter=n_iter, burn_in=burn_in, n_chains=n_chains, seed=seed
    )
    return post, posterior_summary(post)


def shrinkage_report(ds, post) -> dict:
    """RMSE of partial pooling vs no-pooling vs complete-pooling on true means."""
    truth = ds.group_true_means
    partial = post.theta.mean(axis=0)
    return {
        "partial_pooling_rmse": rmse(partial, truth),
        "no_pooling_rmse": rmse(no_pooling(ds.y), truth),
        "complete_pooling_rmse": rmse(complete_pooling(ds.y), truth),
        "partial_estimates": partial.tolist(),
        "no_pooling_estimates": no_pooling(ds.y).tolist(),
        "global_mean": float(ds.y.mean()),
    }


def shrinkage_study(
    n_sims: int = 50,
    n_groups: int = 16,
    n_per_group: int = 4,
    mu: float = 5.0,
    tau: float = 3.0,
    sigma: float = 7.0,
    n_iter: int = 1500,
    burn_in: int = 500,
    base_seed: int = 0,
) -> dict:
    """Average RMSE of each estimator over many datasets.

    A single small dataset is noisy, so the honest claim — partial pooling beats
    per-group MLEs — is a statement about the *average*. This runs many draws and
    reports the mean RMSE plus how often partial pooling wins.
    """
    partial, nopool, complete = [], [], []
    for i in range(n_sims):
        ds = make_hierarchical(
            n_groups=n_groups,
            n_per_group=n_per_group,
            mu=mu,
            tau=tau,
            sigma=sigma,
            seed=base_seed + i,
        )
        post = gibbs_sampler(
            ds.y, ds.sigma, n_iter=n_iter, burn_in=burn_in, n_chains=1, seed=base_seed + i
        )
        truth = ds.group_true_means
        partial.append(rmse(post.theta.mean(axis=0), truth))
        nopool.append(rmse(no_pooling(ds.y), truth))
        complete.append(rmse(complete_pooling(ds.y), truth))
    partial, nopool, complete = np.array(partial), np.array(nopool), np.array(complete)
    return {
        "n_sims": n_sims,
        "mean_partial_pooling_rmse": float(partial.mean()),
        "mean_no_pooling_rmse": float(nopool.mean()),
        "mean_complete_pooling_rmse": float(complete.mean()),
        "partial_win_rate": float((partial < nopool).mean()),
    }


def convergence(post) -> dict:
    """Max R-hat across the group means, plus mu and tau."""
    return {
        "rhat_theta_max": float(np.max(rhat(post.chains_theta))),
        "rhat_mu": rhat(post.chains_mu),
        "rhat_tau": rhat(post.chains_tau),
    }


def calibration_curve(
    levels: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95),
    n_sims: int = 100,
    n_groups: int = 16,
    n_per_group: int = 4,
    mu: float = 5.0,
    tau: float = 3.0,
    sigma: float = 7.0,
    n_iter: int = 1500,
    burn_in: int = 500,
    base_seed: int = 0,
) -> dict:
    """Empirical coverage of credible intervals across many datasets.

    For each simulated dataset we sample the posterior once and, at every
    nominal level, record the fraction of true group means falling inside their
    credible interval. A well-calibrated model tracks the diagonal.
    """
    hits = {lv: 0 for lv in levels}
    total = 0
    for i in range(n_sims):
        ds = make_hierarchical(
            n_groups=n_groups,
            n_per_group=n_per_group,
            mu=mu,
            tau=tau,
            sigma=sigma,
            seed=base_seed + i,
        )
        post = gibbs_sampler(
            ds.y, ds.sigma, n_iter=n_iter, burn_in=burn_in, n_chains=1, seed=base_seed + i
        )
        for lv in levels:
            ci = credible_interval(post.theta, lv)  # (J, 2)
            inside = (ci[:, 0] <= ds.group_true_means) & (ds.group_true_means <= ci[:, 1])
            hits[lv] += int(inside.sum())
        total += ds.n_groups
    empirical = [hits[lv] / total for lv in levels]
    return {
        "levels": list(levels),
        "empirical_coverage": empirical,
        "n_sims": n_sims,
        "mean_abs_calibration_error": float(
            np.mean(np.abs(np.array(empirical) - np.array(levels)))
        ),
    }
