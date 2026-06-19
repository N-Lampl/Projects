"""Fast smoke tests (run in CI). The one slow end-to-end test that trains is
marked @slow and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from rand_smoothing import (
    ABSTAIN,
    SmallCNN,
    SmoothedClassifier,
    certified_accuracy_at,
    clopper_pearson_lower,
    make_synthetic,
    norm_ppf,
    set_seed,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_norm_ppf_known_values():
    assert abs(norm_ppf(0.5) - 0.0) < 1e-6
    assert abs(norm_ppf(0.975) - 1.959964) < 1e-3  # ~1.96
    assert norm_ppf(0.5) < norm_ppf(0.6)  # monotone


def test_clopper_pearson_bounds_and_monotonicity():
    """Lower bound is a valid probability, 0 when k=0, and tighter as n grows."""
    assert clopper_pearson_lower(0, 100, 0.001) == 0.0
    lb = clopper_pearson_lower(90, 100, 0.05)
    assert 0.0 < lb < 0.90  # strictly below the point estimate
    # more samples at the same proportion -> tighter (higher) lower bound
    lb_small = clopper_pearson_lower(90, 100, 0.05)
    lb_large = clopper_pearson_lower(900, 1000, 0.05)
    assert lb_large > lb_small


def test_clopper_pearson_lower_below_mle():
    for k, n in [(50, 100), (700, 1000), (995, 1000)]:
        assert clopper_pearson_lower(k, n, 0.001) <= k / n


def test_radius_formula_matches_definition():
    """R = sigma * Phi^-1(pA): grows with pA and with sigma; 0 at pA=0.5."""
    sigma = 0.25
    assert abs(sigma * norm_ppf(0.5)) < 1e-9
    assert sigma * norm_ppf(0.9) > 0
    assert 0.5 * norm_ppf(0.9) > 0.25 * norm_ppf(0.9)  # scales with sigma


def test_certified_accuracy_is_monotone_nonincreasing():
    radii = np.array([0.1, 0.4, 0.8, 0.0, 0.6])
    correct = np.array([True, True, True, False, True])
    vals = [certified_accuracy_at(radii, correct, r) for r in [0.0, 0.3, 0.5, 0.9]]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    # a wrong/abstained point (correct=False) is never counted
    assert certified_accuracy_at(radii, correct, 0.0) == 4 / 5


def test_smoothed_classifier_certifies_a_confident_constant_model():
    """A base model that always predicts class 3 -> pA=1 -> a large finite radius."""

    class Const(torch.nn.Module):
        def forward(self, x):  # noqa: D401
            logits = torch.full((x.shape[0], 10), -10.0)
            logits[:, 3] = 10.0
            return logits

    sc = SmoothedClassifier(Const(), sigma=0.5, num_classes=10)
    x = torch.rand(1, 1, 28, 28)
    cls, r = sc.certify(x, n0=20, n=200, alpha=0.001, batch=64)
    assert cls == 3
    assert r > 0 and math.isfinite(r)


def test_make_synthetic_shapes_and_range():
    ds = make_synthetic(32, seed=0)
    x, y = ds[0]
    assert x.shape == (1, 28, 28)
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert 0 <= int(y) < 10


def test_model_output_shape():
    out = SmallCNN().eval()(torch.rand(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_abstain_constant():
    assert ABSTAIN == -1


import pytest  # noqa: E402


@pytest.mark.slow
def test_end_to_end_certifies_above_chance():
    """Train a noise-augmented base on synthetic data, then certify a few points.

    Expect certified clean accuracy clearly above the 10% random baseline.
    """
    from rand_smoothing import evaluate, get_loaders, train

    set_seed(42)
    sigma = 0.25
    train_loader, test_loader = get_loaders(
        dataset="synthetic", batch_size=128, train_subset=4000, test_subset=200
    )
    model = SmallCNN()
    train(model, train_loader, sigma=sigma, epochs=2, log_every=0)
    assert evaluate(model, test_loader) > 0.4  # learned the synthetic structure

    sc = SmoothedClassifier(model, sigma=sigma, num_classes=10)
    xs, ys = next(iter(get_loaders(dataset="synthetic", batch_size=30, test_subset=30)[1]))
    radii, correct = [], []
    for i in range(len(xs)):
        c, r = sc.certify(xs[i : i + 1], n0=50, n=300, alpha=0.01, batch=100)
        radii.append(r)
        correct.append(c == int(ys[i]) and c != ABSTAIN)
    radii, correct = np.array(radii), np.array(correct)
    assert certified_accuracy_at(radii, correct, 0.0) > 0.3
    # certified accuracy must not increase with radius
    assert certified_accuracy_at(radii, correct, 0.3) <= certified_accuracy_at(radii, correct, 0.0)
