"""Fast smoke tests (run in CI). The one slow end-to-end test that trains the
target + detector is marked @slow and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import torch

from adv_detector import (
    FEATURE_NAMES,
    SmallCNN,
    bit_depth_reduce,
    detector_features,
    fgsm_perturb,
    median_blur,
    pick_threshold_at_fpr,
    set_seed,
    squeeze_scores,
    synthetic_digits,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_synthetic_digits_shape_and_range():
    x, y = synthetic_digits(20, seed=0)
    assert x.shape == (20, 1, 28, 28)
    assert y.shape == (20,)
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert set(y.tolist()).issubset(set(range(10)))


def test_bit_depth_reduction_quantizes():
    x = torch.rand(4, 1, 28, 28)
    xq = bit_depth_reduce(x, bits=1)  # -> {0, 1}
    assert set(torch.unique(xq).tolist()).issubset({0.0, 1.0})
    assert xq.shape == x.shape


def test_median_blur_preserves_shape_and_range():
    x = torch.rand(3, 1, 28, 28)
    xb = median_blur(x, kernel=3)
    assert xb.shape == x.shape
    assert xb.min() >= 0.0 and xb.max() <= 1.0


def test_fgsm_respects_epsilon_and_pixel_bounds():
    """Core attack invariants used to manufacture adversarial inputs."""
    set_seed(0)
    model = SmallCNN().eval()
    x = torch.rand(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    eps = 0.1
    x_adv = fgsm_perturb(model, x, y, eps)
    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0
    assert (x_adv - x).abs().max().item() <= eps + 1e-5


def test_detector_feature_vector_shape():
    set_seed(0)
    model = SmallCNN().eval()
    x = torch.rand(6, 1, 28, 28)
    feats = detector_features(model, x)
    assert feats.shape == (6, len(FEATURE_NAMES))
    assert torch.isfinite(feats).all()


def test_squeeze_scores_nonnegative():
    """L1 softmax distances are non-negative and the max column dominates."""
    set_seed(0)
    model = SmallCNN().eval()
    x = torch.rand(5, 1, 28, 28)
    s = squeeze_scores(model, x)
    assert s.shape == (5, 3)
    assert (s >= -1e-6).all()
    assert (s[:, 2] >= s[:, 0] - 1e-6).all()
    assert (s[:, 2] >= s[:, 1] - 1e-6).all()


def test_threshold_at_fpr_respects_budget():
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 100)
    scores = np.concatenate([rng.uniform(0, 0.5, 100), rng.uniform(0.5, 1.0, 100)])
    thr = pick_threshold_at_fpr(y, scores, target_fpr=0.05)
    realized_fpr = (scores[y == 0] >= thr).mean()
    assert realized_fpr <= 0.05 + 1e-9


import pytest  # noqa: E402


@pytest.mark.slow
def test_detector_end_to_end_catches_fgsm():
    """Train target + detector on synthetic data; detector should beat chance."""
    from sklearn.metrics import roc_auc_score

    from adv_detector import (
        build_feature_dataset,
        evaluate,
        get_loaders,
        train,
        train_detector,
    )

    set_seed(42)
    train_loader, test_loader = get_loaders(
        dataset="synthetic", train_subset=3000, test_subset=1000
    )
    model = SmallCNN()
    train(model, train_loader, epochs=2, log_every=0)
    assert evaluate(model, test_loader) > 0.7  # target learned the synthetic task

    X_tr, y_tr = build_feature_dataset(model, train_loader, epsilon=0.2)
    X_te, y_te = build_feature_dataset(model, test_loader, epsilon=0.2)
    assert y_tr.sum() > 0 and y_te.sum() > 0  # FGSM actually flipped some predictions

    bundle = train_detector(X_tr, y_tr, target_fpr=0.05)
    auc = roc_auc_score(y_te, bundle.score(X_te))
    assert auc > 0.7  # detector clearly separates clean vs adversarial
