"""Fast smoke tests (run in CI). The one slow end-to-end test is @slow and
excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import pytest
import torch

from attack_zoo import (
    SmallCNN,
    cw_l2,
    deepfool,
    make_synthetic,
    pgd,
    run_attack,
    set_seed,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_model_handles_both_channel_counts():
    set_seed(0)
    out_rgb = SmallCNN(in_channels=3, num_classes=3).eval()(torch.rand(2, 3, 32, 32))
    out_gray = SmallCNN(in_channels=1, num_classes=10).eval()(torch.rand(2, 1, 28, 28))
    assert out_rgb.shape == (2, 3)
    assert out_gray.shape == (2, 10)


def test_synthetic_dataset_shape_and_range():
    ds = make_synthetic(n_per_class=10, num_classes=3)
    x, y = ds[0]
    assert x.shape == (3, 32, 32)
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert len(ds) == 30


def test_pgd_respects_epsilon_and_bounds():
    """Core PGD invariants: L-inf step <= epsilon, output stays a valid image."""
    set_seed(0)
    model = SmallCNN(in_channels=3, num_classes=3).eval()
    x = torch.rand(6, 3, 32, 32)
    y = torch.randint(0, 3, (6,))
    eps = 0.05
    x_adv = pgd(model, x, y, epsilon=eps, steps=5, random_start=False)
    assert x_adv.shape == x.shape
    assert float(x_adv.min()) >= 0.0 and float(x_adv.max()) <= 1.0
    assert (x_adv - x).abs().max().item() <= eps + 1e-5


def test_cw_keeps_valid_image_and_changes_input():
    set_seed(0)
    model = SmallCNN(in_channels=3, num_classes=3).eval()
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 3, (4,))
    x_adv = cw_l2(model, x, y, steps=10, lr=0.05)
    assert float(x_adv.min()) >= 0.0 and float(x_adv.max()) <= 1.0
    assert x_adv.shape == x.shape


def test_deepfool_runs_and_stays_valid():
    set_seed(0)
    model = SmallCNN(in_channels=3, num_classes=3).eval()
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 3, (4,))
    x_adv = deepfool(model, x, y, steps=5)
    assert x_adv.shape == x.shape
    assert float(x_adv.min()) >= 0.0 and float(x_adv.max()) <= 1.0


def test_run_attack_returns_expected_keys():
    from torch.utils.data import DataLoader

    set_seed(0)
    model = SmallCNN(in_channels=3, num_classes=3).eval()
    ds = make_synthetic(n_per_class=8, num_classes=3)
    loader = DataLoader(ds, batch_size=8)
    m = run_attack(model, loader, pgd, epsilon=0.05, steps=3)
    for k in ("success_rate", "mean_l2", "mean_linf", "runtime_s", "n_correct"):
        assert k in m
    assert 0.0 <= m["success_rate"] <= 1.0


@pytest.mark.slow
def test_attacks_succeed_end_to_end():
    """Train on synthetic data, then confirm each attack flips most predictions."""
    from torch.utils.data import DataLoader

    from attack_zoo import cw_l2, deepfool, evaluate, get_loaders, pgd
    from attack_zoo.train import train

    set_seed(42)
    train_loader, test_loader, meta = get_loaders(source="synthetic", num_classes=3)
    model = SmallCNN(in_channels=meta["in_channels"], num_classes=meta["num_classes"])
    train(model, train_loader, epochs=3, log_every=0)
    assert evaluate(model, test_loader) > 0.7  # learned the task

    small = DataLoader(test_loader.dataset, batch_size=32)
    for fn, kw in [
        (pgd, dict(epsilon=0.1, steps=20)),
        (cw_l2, dict(steps=60)),
        (deepfool, dict(steps=40)),
    ]:
        m = run_attack(model, small, fn, **kw)
        assert m["success_rate"] > 0.5, (fn.__name__, m)
