"""Fast smoke tests (run in CI). The one slow test that trains both models is
marked @slow and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import pytest
import torch

from adv_training import (
    SmallCNN,
    accuracy_under_attack,
    make_synthetic,
    pgd_perturb,
    set_seed,
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


def test_synthetic_data_is_in_pixel_range_and_balanced():
    ds = make_synthetic(20, seed=0)
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.stack([ds[i][1] for i in range(len(ds))])
    assert xs.shape[1:] == (1, 28, 28)
    assert xs.min() >= 0.0 and xs.max() <= 1.0
    assert len(ds) == 20 * 10
    # balanced: 20 per class
    counts = torch.bincount(ys, minlength=10)
    assert (counts == 20).all()


def test_pgd_respects_epsilon_and_pixel_bounds():
    """Core attack invariants: L-inf step <= epsilon, output stays a valid image."""
    set_seed(0)
    model = SmallCNN().eval()
    x = torch.rand(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    eps = 0.1

    x_adv = pgd_perturb(model, x, y, eps, steps=5)

    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0  # clipped to valid range
    # total displacement bounded by epsilon (L-inf ball projection), tiny float slack
    assert (x_adv - x).abs().max().item() <= eps + 1e-5


def test_pgd_zero_epsilon_is_identity():
    set_seed(0)
    model = SmallCNN().eval()
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    x_adv = pgd_perturb(model, x, y, 0.0, steps=5)
    assert torch.equal(x_adv, x)


def test_pgd_changes_the_input():
    set_seed(0)
    model = SmallCNN().eval()
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    x_adv = pgd_perturb(model, x, y, 0.1, steps=5)
    assert not torch.equal(x_adv, x)


def test_accuracy_under_attack_monotone_ish_and_clean_matches():
    """Sweep returns a value per epsilon; eps=0 equals plain accuracy and
    attacking never *increases* accuracy on an untrained model."""
    set_seed(0)
    model = SmallCNN().eval()
    from torch.utils.data import DataLoader

    ds = make_synthetic(10, seed=7)
    loader = DataLoader(ds, batch_size=32)
    res = accuracy_under_attack(model, loader, [0.0, 0.1, 0.2], steps=3)
    assert set(res) == {0.0, 0.1, 0.2}
    assert all(0.0 <= v <= 1.0 for v in res.values())
    assert res[0.2] <= res[0.0] + 1e-9


@pytest.mark.slow
def test_adversarial_training_beats_standard_under_pgd():
    """Train standard vs PGD-adversarial on a small synthetic set; confirm the
    adversarially-trained model is more robust under PGD at the training eps."""
    from adv_training import evaluate, get_loaders
    from adv_training.train import train

    set_seed(42)
    train_loader, test_loader = get_loaders(
        batch_size=128, train_per_class=200, test_per_class=60
    )

    set_seed(42)
    std = SmallCNN()
    train(std, train_loader, epochs=4)

    set_seed(42)
    adv = SmallCNN()
    train(adv, train_loader, epochs=4, adv_epsilon=0.1, adv_steps=5)

    # both learned the clean task reasonably (AT trades some clean accuracy)
    assert evaluate(std, test_loader) > 0.6
    assert evaluate(adv, test_loader) > 0.5

    std_r = accuracy_under_attack(std, test_loader, [0.1], steps=7)[0.1]
    adv_r = accuracy_under_attack(adv, test_loader, [0.1], steps=7)[0.1]
    # the whole point of adversarial training:
    assert adv_r > std_r
