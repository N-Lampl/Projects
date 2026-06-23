"""Fast smoke tests (run in CI). The one slow end-to-end test that fits both
detectors on the full graph is marked @slow and excluded via `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from aml_typologies import (
    FEATURE_NAMES,
    build_features,
    cycle_members,
    evaluate,
    generate_aml_graph,
    precision_at_k,
    recall_at_fpr_budget,
    score_isolation_forest,
    score_rules_rf,
    set_seed,
)
from aml_typologies.features import _cycle_members_fallback


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_graph_generation_is_deterministic():
    g1 = generate_aml_graph(n_accounts=300, seed=7)
    g2 = generate_aml_graph(n_accounts=300, seed=7)
    assert g1.transactions.equals(g2.transactions)
    assert g1.labels.equals(g2.labels)


def test_graph_has_planted_typologies():
    g = generate_aml_graph(n_accounts=400, seed=1)
    assert g.labels.sum() > 0  # some suspicious accounts exist
    assert (g.typology == "structuring").any()
    assert (g.typology == "layering").any()
    assert len(g.rings) > 0  # at least one layering cycle planted


def test_feature_matrix_shape_and_columns():
    g = generate_aml_graph(n_accounts=400, seed=2)
    feats = build_features(g)
    assert list(feats.columns) == FEATURE_NAMES
    assert len(feats) == g.n_accounts
    assert not feats.isna().any().any()


def test_fallback_cycle_detection_finds_planted_ring():
    """Pure-python SCC fallback must flag every account in a planted ring."""
    g = generate_aml_graph(n_accounts=400, seed=3)
    members = _cycle_members_fallback(g.transactions, g.accounts)
    assert members  # found some cycle members
    ring = set(g.rings[0])
    assert ring.issubset(members)


def test_cycle_detection_matches_fallback_on_ring_members():
    g = generate_aml_graph(n_accounts=400, seed=3)
    members = cycle_members(g.transactions, g.accounts)
    assert set(g.rings[0]).issubset(members)


def test_precision_at_k_and_recall_budget_bounds():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    labels = np.array([1, 0, 1, 0, 0])
    assert precision_at_k(scores, labels, 2) == 0.5
    thresh, recall, fpr, prec, conf = recall_at_fpr_budget(scores, labels, 0.0)
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fpr <= 1.0
    assert sum(conf.values()) == len(labels)


def test_evaluate_keys_present():
    # uses the fast unsupervised detector; the heavy cross-validated rules_rf path
    # is exercised by the @slow end-to-end test below.
    g = generate_aml_graph(n_accounts=400, seed=4)
    feats = build_features(g)
    scores = score_isolation_forest(feats, seed=4)
    ev = evaluate(scores, g.labels.to_numpy(), fpr_budget=0.05)
    for key in ("pr_auc", "roc_auc", "recall_at_budget", "ks_statistic", "precision_at_k"):
        assert key in ev
    assert 0.0 <= ev["pr_auc"] <= 1.0


@pytest.mark.slow
def test_detector_beats_random_end_to_end():
    """Full graph: the rules+RF detector must separate planted typologies well."""
    set_seed(42)
    g = generate_aml_graph(seed=42)
    feats = build_features(g)
    scores = score_rules_rf(feats, g.labels, seed=42)
    ev = evaluate(scores, g.labels.to_numpy(), fpr_budget=0.01)
    assert ev["pr_auc"] > ev["prevalence"] * 2  # far better than random ranking
    assert ev["roc_auc"] > 0.8
