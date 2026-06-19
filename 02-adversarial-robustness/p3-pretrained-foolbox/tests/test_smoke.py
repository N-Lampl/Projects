"""Fast smoke tests (run in CI) + one @slow end-to-end test.

The fast tests assert the attack INVARIANTS that must hold regardless of the
target: L-inf budget respected, output stays a valid image, PGD is at least as
strong as FGSM, confidence drops. No training, no downloads.
"""

from __future__ import annotations

import torch

from pretrained_foolbox import (
    SmallCNN,
    fgsm_perturb,
    make_synthetic,
    pgd_perturb,
    predict,
    set_seed,
    true_label_confidence,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_model_eats_unit_interval_and_outputs_logits():
    model = SmallCNN().eval()
    out = model(torch.rand(4, 3, 32, 32))
    assert out.shape == (4, 4)


def test_synthetic_data_is_valid_and_deterministic():
    x1, y1 = make_synthetic(n_per_class=8, seed=1)
    x2, y2 = make_synthetic(n_per_class=8, seed=1)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert x1.min() >= 0.0 and x1.max() <= 1.0
    assert x1.shape[1:] == (3, 32, 32)
    assert set(y1.tolist()) == {0, 1, 2, 3}


def test_fgsm_respects_epsilon_and_pixel_bounds():
    set_seed(0)
    model = SmallCNN().eval()
    x = torch.rand(8, 3, 32, 32)
    y = torch.randint(0, 4, (8,))
    eps = 0.05
    x_adv = fgsm_perturb(model, x, y, eps)
    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0
    assert (x_adv - x).abs().max().item() <= eps + 1e-5


def test_pgd_stays_inside_eps_ball():
    set_seed(0)
    model = SmallCNN().eval()
    x = torch.rand(6, 3, 32, 32)
    y = torch.randint(0, 4, (6,))
    eps = 0.05
    x_adv = pgd_perturb(model, x, y, eps, steps=5)
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0
    assert (x_adv - x).abs().max().item() <= eps + 1e-5


def test_attacks_change_the_input():
    set_seed(0)
    model = SmallCNN().eval()
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 4, (4,))
    assert not torch.equal(fgsm_perturb(model, x, y, 0.05), x)
    assert not torch.equal(pgd_perturb(model, x, y, 0.05, steps=3), x)


def test_predict_returns_label_and_confidence():
    model = SmallCNN().eval()
    x = torch.rand(5, 3, 32, 32)
    pred, conf = predict(model, x)
    assert pred.shape == (5,)
    assert conf.shape == (5,)
    assert (conf >= 0).all() and (conf <= 1).all()


import pytest  # noqa: E402


@pytest.mark.slow
def test_attack_drops_accuracy_and_confidence_end_to_end():
    """Train the offline target on synthetic data, then confirm FGSM & PGD hurt."""
    from pretrained_foolbox.train import train
    from pretrained_foolbox.data import make_synthetic as mk

    set_seed(42)
    x, y = mk(n_per_class=96, seed=42)
    model = SmallCNN(num_classes=4)
    train(model, x, y, epochs=6, log_every=0)

    # evaluate on correctly-classified images
    with torch.no_grad():
        pred = model(x).argmax(1)
    mask = pred == y
    xc, yc = x[mask][:64], y[mask][:64]
    assert mask.float().mean() > 0.8  # learned the synthetic task

    clean_conf = true_label_confidence(model, xc, yc).mean().item()
    eps = 0.16
    fgsm_acc = (predict(model, fgsm_perturb(model, xc, yc, eps))[0] == yc).float().mean().item()
    pgd_acc = (predict(model, pgd_perturb(model, xc, yc, eps, steps=10))[0] == yc).float().mean().item()
    fgsm_conf = true_label_confidence(model, fgsm_perturb(model, xc, yc, eps), yc).mean().item()

    assert fgsm_acc < 0.9                 # FGSM hurts
    assert pgd_acc <= fgsm_acc + 1e-6     # PGD at least as strong as FGSM
    assert fgsm_conf < clean_conf         # confidence collapses
