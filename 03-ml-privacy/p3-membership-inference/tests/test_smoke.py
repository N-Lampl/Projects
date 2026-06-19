"""Fast smoke tests (run in CI). The one slow end-to-end test that trains shadows
is marked @slow and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from lira_mia import (
    ShadowSignals,
    auc,
    lira_scores,
    make_synthetic_pool,
    roc_from_scores,
    set_seed,
    tpr_at_fpr,
)
from lira_mia.attack import fit_in_out_gaussians
from lira_mia.model import SmallMLP, logit_confidence
from lira_mia.scipy_stub import norm_logpdf


def test_set_seed_is_deterministic():
    import torch

    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_norm_logpdf_matches_closed_form():
    # log N(0; 0, 1) = -0.5*log(2*pi)
    val = norm_logpdf(np.array([0.0]), np.array([0.0]), np.array([1.0]))[0]
    assert abs(val - (-0.5 * np.log(2 * np.pi))) < 1e-9


def test_logit_confidence_shape_and_finite():
    pool = make_synthetic_pool(n_samples=200, n_features=10, n_informative=6, n_classes=3)
    model = SmallMLP(pool.n_features, pool.n_classes)
    phi = logit_confidence(model, pool.X[:32], pool.y[:32])
    assert phi.shape == (32,)
    assert np.isfinite(phi).all()


def test_roc_auc_perfect_and_chance():
    # perfectly separated scores -> AUC = 1
    scores = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    member = np.array([True, True, True, False, False, False])
    fpr, tpr, _ = roc_from_scores(scores, member)
    assert abs(auc(fpr, tpr) - 1.0) < 1e-9
    assert tpr_at_fpr(fpr, tpr, 0.01) == pytest.approx(1.0)

    # identical scores -> no separation, AUC ~ 0.5
    flat = np.zeros(100)
    rng = np.random.RandomState(0)
    m = rng.rand(100) > 0.5
    fpr2, tpr2, _ = roc_from_scores(flat + rng.randn(100) * 1e-9, m)
    assert 0.35 < auc(fpr2, tpr2) < 0.65


def test_lira_scores_separate_members_on_planted_signal():
    """If IN-shadow confidences are systematically higher than OUT, the target's
    high confidence on a member must yield a higher LiRA score than a non-member.
    """
    n_ex, n_shadow = 4, 10
    rng = np.random.RandomState(0)
    member_mask = np.zeros((n_ex, n_shadow), dtype=bool)
    member_mask[:, : n_shadow // 2] = True
    phi = np.where(member_mask, 3.0, -3.0) + rng.randn(n_ex, n_shadow) * 0.1
    sig = ShadowSignals(phi=phi, member=member_mask)

    # target looks "IN" for example 0, "OUT" for example 1
    phi_target = np.array([3.0, -3.0, 3.0, -3.0])
    scores = lira_scores(phi_target, sig)
    assert scores[0] > scores[1]


def test_fit_in_out_gaussians_handles_empty_world():
    # one example with no OUT shadows must still produce finite stats
    phi = np.array([[1.0, 1.1, 0.9, 1.05]])
    member = np.array([[True, True, True, True]])
    mu_in, s_in, mu_out, s_out = fit_in_out_gaussians(ShadowSignals(phi, member))
    assert np.isfinite([mu_in, s_in, mu_out, s_out]).all()
    assert s_in[0] > 0 and s_out[0] > 0


@pytest.mark.slow
def test_lira_beats_chance_end_to_end():
    """Small but real: train a target + a few warm-started shadows and confirm the
    LiRA AUC clears chance on a memorising model.
    """
    from lira_mia import (
        build_target,
        collect_shadow_signals,
        make_warm_start,
        target_confidences,
    )

    set_seed(42)
    pool = make_synthetic_pool(n_samples=1500)
    warm = make_warm_start(pool, epochs=8)
    tw = build_target(pool, n_query=200, target_epochs=60, warm_start=warm)
    sig = collect_shadow_signals(pool, tw.query_idx, n_shadows=8, shadow_epochs=60,
                                 warm_start=warm)
    scores = lira_scores(target_confidences(tw.model, pool, tw.query_idx), sig)
    fpr, tpr, _ = roc_from_scores(scores, tw.is_member)
    assert auc(fpr, tpr) > 0.55  # better than a coin flip
