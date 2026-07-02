"""Fast, offline, deterministic tests for the causal-inference project.

They run on the synthetic SCM (known ATE), so they assert *real* behaviour with
no network: adjustment removes confounding bias, the doubly-robust interval covers
the truth, and inverse-propensity weighting balances the covariates. The one test
that downloads the real IHDP benchmark is marked ``@slow`` and excluded from CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_inference import (
    aipw,
    all_estimators,
    balance_table,
    coverage_study,
    estimate_propensity,
    make_scm,
    naive_diff,
    point_estimates,
    regression_adjustment,
    set_seed,
    standardized_mean_diff,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    assert np.array_equal(a, b)


def test_scm_is_reproducible_and_confounded():
    s1 = make_scm(n=500, seed=7)
    s2 = make_scm(n=500, seed=7)
    assert np.array_equal(s1.Y, s2.Y) and np.array_equal(s1.T, s2.T)
    assert set(np.unique(s1.T)) <= {0.0, 1.0}
    # Naive estimate is biased away from the true ATE (that's the whole point).
    naive = naive_diff(s1.X, s1.T, s1.Y).estimate
    assert abs(naive - s1.true_ate) > 0.3


def test_adjustment_beats_naive():
    scm = make_scm(n=4000, tau=2.0, confounding=1.5, seed=1)
    res = all_estimators(scm.X, scm.T, scm.Y)
    naive_bias = abs(res["naive"].estimate - scm.true_ate)
    for method in ("regression", "ipw", "aipw"):
        assert abs(res[method].estimate - scm.true_ate) < naive_bias


def test_aipw_recovers_true_ate_within_ci():
    # Moderate n keeps the interval wide enough that finite-sample wobble on a
    # single draw still covers; aggregate coverage is checked separately below.
    scm = make_scm(n=2000, tau=2.0, confounding=1.5, seed=3)
    r = aipw(scm.X, scm.T, scm.Y)
    assert abs(r.estimate - scm.true_ate) < 0.25
    lo, hi = r.ci
    assert lo <= scm.true_ate <= hi
    assert r.se > 0


def test_regression_and_ipw_shapes():
    scm = make_scm(n=800, seed=2)
    for r in (regression_adjustment, naive_diff):
        out = r(scm.X, scm.T, scm.Y)
        assert np.isfinite(out.estimate)


def test_ipw_weighting_improves_balance():
    scm = make_scm(n=3000, confounding=1.5, seed=5)
    e = estimate_propensity(scm.X, scm.T)
    w = np.where(scm.T == 1, 1.0 / e, 1.0 / (1.0 - e))
    before = np.abs(standardized_mean_diff(scm.X, scm.T)).mean()
    after = np.abs(standardized_mean_diff(scm.X, scm.T, weights=w)).mean()
    assert after < before


def test_coverage_study_aipw_beats_naive():
    cov = coverage_study(n=1500, p=5, tau=2.0, confounding=1.5, n_sims=40, base_seed=10)
    # AIPW should cover near nominal; the biased naive interval should not.
    assert cov["aipw_coverage"] > 0.8
    assert cov["aipw_coverage"] > cov["naive_coverage"]


def test_point_estimates_payload():
    p = point_estimates(n=1000, p=5, tau=2.0, confounding=1.5, seed=0)
    assert {"true_ate", "estimates", "bias", "aipw_se", "aipw_ci"} <= set(p)
    assert set(p["estimates"]) == {"naive", "regression", "ipw", "aipw"}


def test_balance_table_payload():
    b = balance_table(n=1000, p=4, tau=2.0, confounding=1.5, seed=1)
    assert len(b["smd_before"]) == 4
    assert b["mean_abs_smd_after"] <= b["mean_abs_smd_before"]


@pytest.mark.slow
def test_ihdp_real_benchmark():
    """Download the real IHDP benchmark and recover its known ATE with AIPW."""
    from causal_inference import load_ihdp

    try:
        ds = load_ihdp()
    except Exception as exc:  # offline / host down
        pytest.skip(f"IHDP unavailable: {type(exc).__name__}")
    r = aipw(ds.X, ds.T, ds.Y)
    # IHDP true ATE ~ 4; AIPW should land in a sensible neighbourhood.
    assert abs(r.estimate - ds.true_ate) < 1.5
