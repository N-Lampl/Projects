"""Fast smoke tests (run in CI). The one slow test that trains a model end-to-end
is marked @slow and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from inversion_attribute import (
    SmallCNN,
    class_prototypes,
    invert_class,
    make_attribute_dataset,
    make_synthetic,
    reconstruction_quality,
    run_attribute_inference,
    set_seed,
    train_target,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_model_output_shape():
    model = SmallCNN().eval()
    out = model(torch.rand(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_synthetic_data_shapes_and_range():
    X, y = make_synthetic(n_per_class=5)
    assert X.shape == (50, 1, 28, 28)
    assert y.shape == (50,)
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert set(y.tolist()) == set(range(10))


def test_prototypes_are_distinct():
    protos = class_prototypes()
    assert protos.shape == (10, 28, 28)
    # no two class prototypes should be identical
    flat = protos.reshape(10, -1)
    for i in range(10):
        for j in range(i + 1, 10):
            assert not np.allclose(flat[i], flat[j])


def test_invert_class_returns_valid_image():
    """Core inversion invariants: output is a valid [0,1] image of right shape,
    and a few steps push class confidence up from the blank-init value."""
    set_seed(0)
    model = SmallCNN().eval()
    img, conf = invert_class(model, target_class=3, steps=15, device=torch.device("cpu"))
    assert img.shape == (1, 1, 28, 28)
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert 0.0 <= conf <= 1.0


def test_inversion_raises_target_confidence():
    """On a trained model, inversion drives class confidence well above chance."""
    from inversion_attribute import get_loaders, train

    set_seed(42)
    train_loader, _ = get_loaders(n_per_class=100)
    model = SmallCNN()
    train(model, train_loader, epochs=3)
    _, conf = invert_class(model, target_class=5, steps=200)
    assert conf > 0.5  # far above the 0.1 chance level for 10 classes


def test_reconstruction_quality_perfect_for_identity():
    protos = torch.tensor(class_prototypes())
    q = reconstruction_quality(protos.unsqueeze(1), protos)
    assert q["top1_match_rate"] == 1.0
    assert q["mean_own_class_correlation"] > 0.99


def test_attribute_inference_beats_baseline_when_signal_strong():
    """With a strong sensitive->label dependence, the attack should clearly beat
    the majority baseline; with no dependence it should not."""
    set_seed(42)
    strong = make_attribute_dataset(n=1500, s_signal=3.0)
    clf, X_te, _ = train_target(strong)
    res = run_attribute_inference(strong, clf, X_te)
    assert res["attack_accuracy"] > res["baseline_accuracy"]
    assert res["lift_over_baseline"] > 0.05


def test_attribute_inference_no_leak_without_signal():
    set_seed(42)
    none = make_attribute_dataset(n=1500, s_signal=0.0)
    clf, X_te, _ = train_target(none)
    res = run_attribute_inference(none, clf, X_te)
    # no real dependence -> attack should not meaningfully beat the baseline
    assert res["lift_over_baseline"] < 0.05


@pytest.mark.slow
def test_inversion_recovers_class_signatures_end_to_end():
    """Train the target on synthetic data, invert all classes, and confirm the
    reconstructions correlate with the right class prototypes (privacy leakage)."""
    from inversion_attribute import (
        evaluate,
        get_loaders,
        invert_all_classes,
        train,
    )

    set_seed(42)
    train_loader, test_loader = get_loaders(n_per_class=300)
    model = SmallCNN()
    train(model, train_loader, epochs=5)
    assert evaluate(model, test_loader) > 0.9

    recon, confs = invert_all_classes(model, n_classes=10, steps=300)
    protos = torch.tensor(class_prototypes())
    q = reconstruction_quality(recon, protos)
    assert sum(confs) / len(confs) > 0.8
    assert q["top1_match_rate"] >= 0.6  # majority of classes recovered recognisably
