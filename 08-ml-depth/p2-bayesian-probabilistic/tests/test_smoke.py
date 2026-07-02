"""Fast, offline, deterministic tests for the Bayesian hierarchical project.

They run the numpy Gibbs sampler on synthetic data with KNOWN group means, so
they assert real behaviour with no network and no PyMC: the sampler recovers the
truth, credible intervals cover it, partial pooling beats no pooling, and the
chains mix (R-hat ~ 1). The one PyMC cross-check is marked ``@slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayes_pp import (
    calibration_curve,
    complete_pooling,
    convergence,
    credible_interval,
    fit_dataset,
    gibbs_sampler,
    make_hierarchical,
    no_pooling,
    posterior_summary,
    rhat,
    rmse,
    set_seed,
    shrinkage_study,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    assert np.array_equal(a, b)


def test_gibbs_is_reproducible():
    ds = make_hierarchical(seed=1)
    p1 = gibbs_sampler(ds.y, ds.sigma, n_iter=1500, burn_in=500, seed=7)
    p2 = gibbs_sampler(ds.y, ds.sigma, n_iter=1500, burn_in=500, seed=7)
    assert np.allclose(p1.theta.mean(axis=0), p2.theta.mean(axis=0))


def test_recovers_global_parameters():
    ds = make_hierarchical(n_groups=12, n_per_group=20, mu=5.0, tau=3.0, seed=2)
    _post, summary = fit_dataset(ds, n_iter=3000, burn_in=1000, seed=2)
    # Global mean recovered within a sensible tolerance.
    assert abs(summary["mu_mean"] - ds.mu) < 1.5


def test_partial_pooling_beats_no_pooling_on_average():
    # A single small dataset is noisy; the shrinkage benefit is an *average* fact.
    study = shrinkage_study(n_sims=30, n_iter=1000, burn_in=300, base_seed=0)
    assert study["mean_partial_pooling_rmse"] < study["mean_no_pooling_rmse"]
    assert study["partial_win_rate"] > 0.5


def test_credible_interval_covers_truth():
    ds = make_hierarchical(n_groups=10, n_per_group=8, seed=4)
    post, summary = fit_dataset(ds, n_iter=3000, burn_in=1000, seed=4)
    ci = np.asarray(summary["theta_ci"])  # (J, 2)
    inside = (ci[:, 0] <= ds.group_true_means) & (ds.group_true_means <= ci[:, 1])
    # A 90% interval should cover most of the 10 groups.
    assert inside.mean() >= 0.7


def test_chains_mix_rhat_near_one():
    ds = make_hierarchical(seed=5)
    post, _ = fit_dataset(ds, n_iter=3000, burn_in=1000, n_chains=3, seed=5)
    conv = convergence(post)
    assert conv["rhat_theta_max"] < 1.1
    assert abs(conv["rhat_mu"] - 1.0) < 0.1


def test_rhat_and_baselines_shapes():
    ds = make_hierarchical(n_groups=6, seed=6)
    assert no_pooling(ds.y).shape == (6,)
    assert complete_pooling(ds.y).shape == (6,)
    chains = np.random.default_rng(0).standard_normal((3, 200, 6))
    assert rhat(chains).shape == (6,)


def test_credible_interval_helper():
    draws = np.random.default_rng(0).standard_normal((2000, 4))
    ci = credible_interval(draws, level=0.9)
    assert ci.shape == (4, 2)
    assert np.all(ci[:, 0] < ci[:, 1])


def test_posterior_summary_payload():
    ds = make_hierarchical(seed=7)
    post, _ = fit_dataset(ds, n_iter=1500, burn_in=500, seed=7)
    s = posterior_summary(post)
    assert {"theta_mean", "theta_ci", "mu_mean", "tau_mean", "level"} <= set(s)
    assert len(s["theta_mean"]) == ds.n_groups


def test_calibration_curve_is_reasonable():
    cal = calibration_curve(n_sims=25, n_iter=1000, burn_in=300, base_seed=0)
    assert len(cal["empirical_coverage"]) == len(cal["levels"])
    # Empirical coverage should track the nominal levels, not be wildly off.
    assert cal["mean_abs_calibration_error"] < 0.15


def test_rmse_zero_on_exact():
    x = np.array([1.0, 2.0, 3.0])
    assert rmse(x, x) == 0.0


@pytest.mark.slow
def test_pymc_agrees_with_gibbs():
    """PyMC NUTS posterior means should match the numpy Gibbs means."""
    try:
        import pymc as pm
    except Exception as exc:  # PyMC not installed
        pytest.skip(f"PyMC unavailable: {type(exc).__name__}")

    ds = make_hierarchical(n_groups=8, n_per_group=8, seed=1)
    post, _ = fit_dataset(ds, n_iter=3000, burn_in=1000, seed=1)
    gibbs_means = post.theta.mean(axis=0)

    with pm.Model():
        mu = pm.Normal("mu", 0.0, 100.0)
        tau = pm.HalfNormal("tau", 10.0)
        theta = pm.Normal("theta", mu, tau, shape=ds.n_groups)
        pm.Normal("y", theta[:, None], ds.sigma[:, None], observed=ds.y)
        idata = pm.sample(500, tune=500, chains=2, progressbar=False, random_seed=1)
    pymc_means = idata.posterior["theta"].mean(dim=("chain", "draw")).values
    assert np.max(np.abs(pymc_means - gibbs_means)) < 1.0
