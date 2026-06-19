"""Fast smoke tests (run in CI). The one slow end-to-end test that builds the
full stream and runs the monitor is marked @slow and excluded via -m "not slow".
"""

from __future__ import annotations

import numpy as np
import pytest

from drift_monitoring import (
    AlertThresholds,
    StreamConfig,
    evaluate_window,
    generate_stream,
    ks_pvalue,
    ks_statistic,
    psi,
    psi_severity,
    run_monitor,
    set_seed,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_psi_zero_on_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift():
    """PSI must increase monotonically as the current dist shifts further away."""
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    small = psi(ref, rng.normal(0.5, 1, 5000))
    large = psi(ref, rng.normal(2.0, 1, 5000))
    assert 0.0 < small < large


def test_psi_is_nonnegative():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(1.0, 2.0, 2000)
    assert psi(ref, cur) >= 0.0


def test_ks_statistic_bounds_and_identity():
    rng = np.random.default_rng(2)
    x = rng.normal(size=2000)
    assert ks_statistic(x, x) == pytest.approx(0.0, abs=1e-9)
    d = ks_statistic(x, rng.normal(3.0, 1.0, 2000))
    assert 0.0 <= d <= 1.0
    assert d > 0.5  # large shift -> large KS distance


def test_ks_pvalue_significance():
    rng = np.random.default_rng(3)
    same = ks_pvalue(rng.normal(size=2000), rng.normal(size=2000))
    diff = ks_pvalue(rng.normal(0, 1, 2000), rng.normal(2, 1, 2000))
    assert same > 0.05      # cannot reject equality
    assert diff < 0.01      # clearly different


def test_psi_severity_labels():
    assert psi_severity(0.05) == "stable"
    assert psi_severity(0.15) == "moderate"
    assert psi_severity(0.40) == "major"


def test_no_alert_on_clean_window():
    """A window drawn from the reference distribution must not alert."""
    set_seed(0)
    ref, windows, _ = generate_stream(StreamConfig(n_windows=2, drift_start=99))
    rep = evaluate_window(ref, windows[0], list("abcde")[:5], AlertThresholds())
    rep2 = evaluate_window(ref, windows[0],
                           ["f0", "f1", "f2", "f3", "f4"], AlertThresholds())
    assert rep["window_alert"] is False
    assert rep2["window_alert"] is False


def test_stable_control_feature_does_not_alarm():
    """session_len is never drifted; it must stay clear of an alert even late."""
    set_seed(0)
    cfg = StreamConfig()
    ref, windows, _ = generate_stream(cfg)
    reports = run_monitor(ref, windows, cfg.feature_names, AlertThresholds())
    assert all("session_len" not in r["alerting_features"] for r in reports)


@pytest.mark.slow
def test_monitor_detects_injected_drift_end_to_end():
    """Full pipeline: clean early windows stay quiet; drift gets detected after onset."""
    set_seed(42)
    cfg = StreamConfig()
    ref, windows, strengths = generate_stream(cfg)
    reports = run_monitor(ref, windows, cfg.feature_names, AlertThresholds())

    pre = [r for r in reports if r["window"] < cfg.drift_start]
    post = [r for r in reports if r["window"] >= cfg.drift_start]
    assert not any(r["window_alert"] for r in pre)   # no false alarms pre-drift
    assert any(r["window_alert"] for r in post)      # detects the injected drift
