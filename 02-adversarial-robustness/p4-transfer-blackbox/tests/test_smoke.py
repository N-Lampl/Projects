"""Fast smoke tests (run in CI). The one slow test that trains is marked @slow
and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import pytest
import torch

from transfer_blackbox import (
    QueryOracle,
    boundary_attack,
    build_model,
    make_synthetic,
    pgd_perturb,
    set_seed,
    square_attack,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_two_models_are_different_architectures():
    cnn, mlp = build_model("cnn"), build_model("mlp")
    assert type(cnn).__name__ == "SmallCNN"
    assert type(mlp).__name__ == "SmallMLP"
    # both accept 8x8 (synthetic) AND 28x28 (real) inputs -> 10-way output
    for size in (8, 28):
        x = torch.rand(3, 1, size, size)
        assert cnn(x).shape == (3, 10)
        assert mlp(x).shape == (3, 10)


def test_synthetic_data_shape_and_range():
    x, y = make_synthetic(n_per_class=10, seed=0)
    assert x.shape == (100, 1, 8, 8)
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert set(y.tolist()) == set(range(10))


def test_pgd_respects_epsilon_and_pixel_bounds():
    set_seed(0)
    model = build_model("cnn").eval()
    x = torch.rand(8, 1, 8, 8)
    y = torch.randint(0, 10, (8,))
    eps = 0.1
    x_adv = pgd_perturb(model, x, y, eps, steps=5)
    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0
    assert (x_adv - x).abs().max().item() <= eps + 1e-5  # L-inf ball respected


def test_query_oracle_counts_and_respects_budget():
    set_seed(0)
    model = build_model("mlp").eval()
    x = torch.rand(4, 1, 8, 8)
    y = torch.randint(0, 10, (4,))
    budget = 20
    oracle = QueryOracle(model)
    res = square_attack(oracle, x, y, epsilon=0.3, query_budget=budget, seed=1)
    assert oracle.n_queries <= budget * x.shape[0]
    assert res.x_adv.shape == x.shape
    # square attack is L-inf bounded around the originals
    assert (res.x_adv - x).abs().max().item() <= 0.3 + 1e-4


def test_square_perturbation_stays_a_valid_image():
    set_seed(0)
    model = build_model("mlp").eval()
    x = torch.rand(3, 1, 8, 8)
    y = torch.randint(0, 10, (3,))
    res = square_attack(QueryOracle(model), x, y, epsilon=0.3, query_budget=15, seed=2)
    assert res.x_adv.min() >= 0.0 and res.x_adv.max() <= 1.0


def test_boundary_attack_runs_within_budget():
    set_seed(0)
    model = build_model("mlp").eval()
    x = torch.rand(3, 1, 8, 8)
    y = torch.randint(0, 10, (3,))
    budget = 30
    oracle = QueryOracle(model)
    res = boundary_attack(oracle, x, y, query_budget=budget, seed=3)
    assert oracle.n_queries <= budget * x.shape[0]
    assert res.x_adv.min() >= 0.0 and res.x_adv.max() <= 1.0


@pytest.mark.slow
def test_transfer_and_blackbox_end_to_end():
    """Train two different models on synthetic data; confirm PGD on the surrogate
    transfers to the target AND a query attack beats it under a budget."""
    from transfer_blackbox import (
        evaluate,
        get_loaders,
        transfer_accuracy,
    )
    from transfer_blackbox.train import train

    set_seed(42)
    train_loader, test_loader = get_loaders(n_per_class=200)
    set_seed(42)
    surrogate = train(build_model("cnn"), train_loader, epochs=6)
    set_seed(7)
    target = train(build_model("mlp"), train_loader, epochs=6)

    assert evaluate(surrogate, test_loader) > 0.8
    assert evaluate(target, test_loader) > 0.8

    res = transfer_accuracy(surrogate, target, test_loader, [0.0, 0.3], steps=10)
    assert res["target"][0.0] > 0.8           # clean baseline on the target
    assert res["target"][0.3] < res["target"][0.0]  # transfer hurts the target
    assert res["surrogate"][0.3] <= res["target"][0.3] + 1e-6  # white-box >= transfer

    # black-box: take some correctly-classified images, attack within a budget
    x, y = next(iter(test_loader))
    with torch.no_grad():
        keep = (target(x).argmax(1) == y).nonzero(as_tuple=True)[0][:12]
    x, y = x[keep], y[keep]
    res_sq = square_attack(QueryOracle(target), x, y, epsilon=0.3, query_budget=300, seed=0)
    assert res_sq.success.float().mean().item() > 0.3
