"""Fast smoke tests (run in CI). The one slow end-to-end test that trains both
models and runs the full attack is marked @slow and excluded via -m "not slow".
"""

from __future__ import annotations

import numpy as np

from adv_fraud import (
    AttackConfig,
    attack_success_rate,
    detection_report,
    evade,
    feasible,
    make_dataset,
    make_model,
    set_seed,
    threshold_for_fpr,
)
from adv_fraud.data import FEATURES, MUTABLE


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_dataset_is_deterministic_and_imbalanced():
    d1 = make_dataset(n=2000, seed=7)
    d2 = make_dataset(n=2000, seed=7)
    assert np.array_equal(d1.X, d2.X)
    assert np.array_equal(d1.y, d2.y)
    assert d1.X.shape == (2000, len(FEATURES))
    rate = d1.y.mean()
    assert 0.0 < rate < 0.15  # genuinely imbalanced


def test_attack_never_changes_immutable_features():
    """Core threat-model invariant: account history must be untouched."""
    set_seed(0)
    ds = make_dataset(n=1500, seed=0)
    model = make_model().fit(ds.X, ds.y)
    thr = threshold_for_fpr(ds.y, model.predict_proba(ds.X)[:, 1], 0.05)
    X_fraud = ds.X[ds.y == 1][:30]
    X_adv = evade(model, X_fraud, thr, AttackConfig(steps=10))

    for i, f in enumerate(FEATURES):
        if not MUTABLE[f]:
            assert np.allclose(X_adv[:, i], X_fraud[:, i]), f"immutable {f} moved"


def test_attack_respects_feasibility_and_lowers_scores():
    set_seed(0)
    ds = make_dataset(n=1500, seed=0)
    model = make_model().fit(ds.X, ds.y)
    s = model.predict_proba(ds.X)[:, 1]
    thr = threshold_for_fpr(ds.y, s, 0.05)
    X_fraud = ds.X[(ds.y == 1) & (s >= thr)][:30]
    res = attack_success_rate(model, X_fraud, thr, AttackConfig(steps=20))

    # every crafted evasion obeys the constraint contract
    assert res["feasibility_rate"] == 1.0
    assert feasible(res["X_adv"], X_fraud).all()
    # attack should on average reduce the fraud score
    assert res["mean_score_drop"] > 0.0


def test_detection_report_keys():
    ds = make_dataset(n=1500, seed=1)
    model = make_model().fit(ds.X, ds.y)
    rep = detection_report(ds.y, model.predict_proba(ds.X)[:, 1])
    for k in ("pr_auc", "roc_auc", "recall_at_fpr_budget", "confusion"):
        assert k in rep
    assert 0.0 <= rep["pr_auc"] <= 1.0


import pytest  # noqa: E402


@pytest.mark.slow
def test_hardening_reduces_asr_end_to_end():
    """Adversarial training should not make robustness worse, and the attack
    must genuinely succeed against the undefended baseline."""
    from adv_fraud import adversarially_train

    set_seed(42)
    ds = make_dataset(n=8000, seed=42)
    n = len(ds.X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]

    base = make_model().fit(ds.X[tr], ds.y[tr])
    s = base.predict_proba(ds.X[te])[:, 1]
    thr = threshold_for_fpr(ds.y[te], s, 0.05)
    X_attack = ds.X[te][(ds.y[te] == 1) & (s >= thr)]

    before = attack_success_rate(base, X_attack, thr, AttackConfig(steps=30))
    assert before["asr"] > 0.1  # the baseline is genuinely evadable

    hardened = adversarially_train(base, ds.X[tr], ds.y[tr], config=AttackConfig(steps=30))
    s_h = hardened.predict_proba(ds.X[te])[:, 1]
    thr_h = threshold_for_fpr(ds.y[te], s_h, 0.05)
    X_attack_h = ds.X[te][(ds.y[te] == 1) & (s_h >= thr_h)]
    after = attack_success_rate(hardened, X_attack_h, thr_h, AttackConfig(steps=30))

    assert after["asr"] <= before["asr"] + 1e-9  # hardening does not worsen ASR
