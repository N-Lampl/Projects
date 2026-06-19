"""Fast smoke tests (run in CI). One slow end-to-end test is marked @slow
and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from nids_baseline import (
    get_pipeline_api,
    set_seed,
    soc_report,
    sweep_operating_points,
    threshold_for_alert_rate,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_shared_pipeline_imports_by_path():
    """The whole point of this project: the shared library loads by path."""
    api = get_pipeline_api()
    for fn in ("load_data", "build_pipeline", "train", "evaluate", "predict_proba"):
        assert hasattr(api, fn)


def test_threshold_for_alert_rate_matches_budget():
    """The chosen threshold should flag ~the requested fraction of flows."""
    rng = np.random.default_rng(0)
    scores = rng.random(10000)
    thr = threshold_for_alert_rate(scores, 0.10)
    flagged = (scores >= thr).mean()
    assert abs(flagged - 0.10) < 0.02


def test_soc_report_confusion_is_consistent():
    """TP+FP+FN+TN must equal n, and detection/precision must be in [0,1]."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_score = np.array([0.1, 0.9, 0.4, 0.8, 0.2, 0.3, 0.95, 0.6])
    rep = soc_report(y_true, y_score, threshold=0.5)
    c = rep["confusion"]
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == len(y_true)
    assert 0.0 <= rep["detection_rate"] <= 1.0
    assert 0.0 <= rep["alert_precision"] <= 1.0


def test_lower_budget_gives_higher_precision():
    """Tighter alert budget => fewer, higher-confidence alerts => >= precision.

    A monotone-ranking sanity check on the operating-point sweep.
    """
    rng = np.random.default_rng(1)
    n = 4000
    y_true = (rng.random(n) < 0.25).astype(int)
    # scores correlated with the label so ranking is meaningful
    y_score = np.clip(0.25 * rng.standard_normal(n) + 0.5 * y_true + 0.25, 0, 1)
    rows = sweep_operating_points(y_true, y_score, (0.05, 0.25))
    tight, wide = rows[0], rows[1]
    assert tight["alert_precision"] >= wide["alert_precision"] - 1e-9
    assert wide["detection_rate"] >= tight["detection_rate"] - 1e-9


@pytest.mark.slow
def test_end_to_end_trains_and_reports():
    """Load synthetic flows -> train the shared pipeline -> SOC report beats chance."""
    set_seed(42)
    api = get_pipeline_api()
    ds = api.load_data(synthetic=True, n_samples=3000)
    pipe = api.build_pipeline(ds, classifier="rf")
    api.train(pipe, ds)
    scores = api.predict_proba(pipe, ds.X_test)
    thr = threshold_for_alert_rate(scores, 0.10)
    rep = soc_report(ds.y_test, scores, threshold=thr)
    assert rep["roc_auc"] > 0.6  # clearly better than random ranking
    assert rep["detection_rate"] > 0.0
