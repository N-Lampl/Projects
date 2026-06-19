"""Fast smoke tests (run in CI) for the DP capstone + one @slow end-to-end test.

Fast tests check invariants that must hold regardless of randomness:
  * seeding is deterministic,
  * the RDP accountant is monotone and matches the closed-form Gaussian at q=1,
  * find_noise_multiplier actually hits the requested epsilon,
  * per-sample clipping bounds gradient sensitivity,
  * the LiRA ROC machinery is correct on a constructed example.
The one @slow test runs the whole pipeline and asserts the privacy direction:
DP (eps=1) lowers the train-test gap / MIA AUC relative to non-private (eps=inf).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dp_defenses import (
    SmallMLP,
    auc,
    compute_epsilon,
    find_noise_multiplier,
    lira_scores,
    roc_from_scores,
    set_seed,
    tpr_at_fpr,
)
from dp_defenses.dp_train import _clip_and_noise, _per_sample_grads, rdp_subsampled_gaussian
from dp_defenses.mia import ShadowSignals


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_model_output_shape():
    model = SmallMLP(n_features=8, n_classes=4).eval()
    out = model(torch.rand(6, 8))
    assert out.shape == (6, 4)


def test_epsilon_decreases_with_more_noise():
    """More noise (bigger sigma) must give a SMALLER epsilon (stronger privacy)."""
    q, steps, delta = 0.1, 200, 1e-5
    eps_lo = compute_epsilon(0.5, q, steps, delta)
    eps_hi = compute_epsilon(4.0, q, steps, delta)
    assert eps_hi < eps_lo
    assert compute_epsilon(0.0, q, steps, delta) == math.inf


def test_rdp_matches_closed_form_at_q1():
    """At q=1 the subsampled-Gaussian RDP reduces to alpha/(2 sigma^2)."""
    orders = [2.0, 5.0, 10.0]
    sigma = 1.5
    rdp = rdp_subsampled_gaussian(1.0, sigma, orders)
    expected = [a / (2 * sigma**2) for a in orders]
    assert np.allclose(rdp, expected, rtol=1e-9)


def test_find_noise_multiplier_hits_target_epsilon():
    """The bisection should land within tolerance of the requested epsilon."""
    q, steps, delta = 0.1, 300, 1e-5
    for target in (3.0, 1.0):
        sigma = find_noise_multiplier(target, q, steps, delta)
        eps = compute_epsilon(sigma, q, steps, delta)
        assert eps <= target + 0.05  # never exceed the requested budget
        assert eps >= target - 0.5  # and not absurdly conservative


def test_find_noise_multiplier_inf_is_zero_noise():
    assert find_noise_multiplier(math.inf, 0.1, 100, 1e-5) == 0.0


def test_per_sample_clip_bounds_sensitivity():
    """After clipping to C and removing noise, the summed grad L2 <= B * C."""
    set_seed(0)
    model = SmallMLP(n_features=10, n_classes=3)
    xb = torch.randn(16, 10)
    yb = torch.randint(0, 3, (16,))
    grads, _ = _per_sample_grads(model, xb, yb)

    C = 0.7
    summed = _clip_and_noise(grads, max_grad_norm=C, noise_multiplier=0.0, batch_size=16)
    # _clip_and_noise averages by batch -> multiply back to get the clipped sum
    total_sq = sum((g * 16).pow(2).sum().item() for g in summed)
    assert math.sqrt(total_sq) <= 16 * C + 1e-3


def test_roc_and_auc_perfect_separation():
    """A perfectly separating score must give AUC=1 and TPR@1%FPR high."""
    scores = np.array([5.0, 4.0, 3.0, -3.0, -4.0, -5.0])
    member = np.array([True, True, True, False, False, False])
    fpr, tpr, _ = roc_from_scores(scores, member)
    assert auc(fpr, tpr) == pytest.approx(1.0, abs=1e-9)
    assert tpr_at_fpr(fpr, tpr, 0.01) == pytest.approx(1.0)


def test_lira_scores_shape_and_finiteness():
    rng = np.random.RandomState(0)
    n_ex, n_sh = 20, 8
    phi = rng.randn(n_ex, n_sh)
    member = rng.rand(n_ex, n_sh) > 0.5
    sig = ShadowSignals(phi=phi, member=member)
    phi_target = rng.randn(n_ex)
    s = lira_scores(phi_target, sig)
    assert s.shape == (n_ex,)
    assert np.isfinite(s).all()


@pytest.mark.slow
def test_dp_reduces_leakage_end_to_end():
    """The headline claim: DP (eps=1) shrinks the train-test gap AND the MIA AUC
    versus non-private training, at the cost of some test accuracy."""
    from dp_defenses import build_shared_world, evaluate_epsilon, make_synthetic_pool

    set_seed(42)
    pool = make_synthetic_pool(n_samples=1200)
    world = build_shared_world(pool, n_shadows=6, shadow_epochs=15)

    non_private = evaluate_epsilon(world, math.inf, epochs=20)
    private = evaluate_epsilon(world, 1.0, epochs=20)

    # privacy improves: smaller memorisation gap and weaker membership inference
    assert private.gen_gap <= non_private.gen_gap + 1e-6
    assert private.mia_auc <= non_private.mia_auc + 0.02
    # utility is non-trivial but DP costs accuracy
    assert non_private.test_acc > 0.3
    assert private.accounted_epsilon <= 1.0 + 0.05
