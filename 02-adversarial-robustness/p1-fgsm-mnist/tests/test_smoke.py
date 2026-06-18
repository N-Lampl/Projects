"""Fast smoke tests (run in CI). The one slow test that trains is marked @slow
and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import torch

from fgsm_mnist import SmallCNN, fgsm_perturb, set_seed
from fgsm_mnist.attack import accuracy_under_attack


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


def test_fgsm_respects_epsilon_and_pixel_bounds():
    """Core attack invariants: L-inf step <= epsilon, output stays a valid image."""
    set_seed(0)
    model = SmallCNN().eval()
    x = torch.rand(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    eps = 0.1

    x_adv = fgsm_perturb(model, x, y, eps)

    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0  # clipped to valid range
    # each pixel moved by at most epsilon (allow tiny float slack)
    assert (x_adv - x).abs().max().item() <= eps + 1e-5


def test_fgsm_changes_the_input():
    set_seed(0)
    model = SmallCNN().eval()
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    x_adv = fgsm_perturb(model, x, y, 0.1)
    assert not torch.equal(x_adv, x)


import pytest  # noqa: E402


@pytest.mark.slow
def test_attack_reduces_accuracy_end_to_end():
    """Train one quick epoch on a subset, then confirm FGSM drops accuracy."""
    from fgsm_mnist import get_loaders
    from fgsm_mnist.train import train

    set_seed(42)
    train_loader, test_loader = get_loaders(batch_size=128, train_subset=4000, test_subset=1000)
    model = SmallCNN()
    train(model, train_loader, epochs=1, log_every=0)
    res = accuracy_under_attack(model, test_loader, [0.0, 0.2])
    assert res[0.0] > 0.8  # learned something
    assert res[0.2] < res[0.0]  # attack hurts
