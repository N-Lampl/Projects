"""Fast smoke tests (run in CI). The one slow test that trains end to end is
marked @slow and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import pytest
import torch

from model_extraction import (
    MLP,
    QueryBudgetExceeded,
    VictimAPI,
    get_splits,
    make_victim,
    set_seed,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_synthetic_splits_are_disjoint_and_shaped():
    set_seed(0)
    s = get_splits("synthetic", n_victim=100, n_attack=200, n_test=50)
    assert s.victim_x.shape == (100, 1, s.img_size, s.img_size)
    assert s.attack_x.shape[0] == 200
    assert s.test_x.shape[0] == 50
    # pixels normalised to [0, 1]
    assert s.victim_x.min() >= 0.0 and s.victim_x.max() <= 1.0
    # labels in range
    assert int(s.victim_y.max()) < s.n_classes


def test_model_output_shape():
    model = MLP(img_size=16, n_classes=10).eval()
    out = model(torch.rand(4, 1, 16, 16))
    assert out.shape == (4, 10)


def test_victim_api_returns_hard_labels():
    set_seed(0)
    victim = make_victim(16, 10).eval()
    api = VictimAPI(victim)
    x = torch.rand(8, 1, 16, 16)
    labels = api.predict(x)
    assert labels.shape == (8,)
    assert labels.dtype == torch.int64
    assert api.queries_used == 8


def test_rate_limit_defense_blocks_over_budget():
    """The defense: once the budget is spent, further queries are rejected."""
    set_seed(0)
    victim = make_victim(16, 10).eval()
    api = VictimAPI(victim, max_queries=10)
    api.predict(torch.rand(10, 1, 16, 16))  # exactly spends the budget
    assert api.budget_remaining == 0
    with pytest.raises(QueryBudgetExceeded):
        api.predict(torch.rand(1, 1, 16, 16))
    # rejected call must NOT have been metered
    assert api.queries_used == 10


def test_budget_partial_batch_rejected_whole():
    set_seed(0)
    api = VictimAPI(make_victim(16, 10).eval(), max_queries=5)
    with pytest.raises(QueryBudgetExceeded):
        api.predict(torch.rand(6, 1, 16, 16))
    assert api.queries_used == 0


@pytest.mark.slow
def test_extraction_fidelity_increases_with_budget_end_to_end():
    """Train a real victim, then steal it at two budgets; more queries -> more
    fidelity, and the rate-limit defense caps the thief."""
    from model_extraction import evaluate, fidelity_vs_budget, loader
    from model_extraction.train import train

    set_seed(42)
    s = get_splits("synthetic", n_victim=2000, n_attack=4000, n_test=1000)
    victim = make_victim(s.img_size, s.n_classes)
    train(victim, loader(s.victim_x, s.victim_y, shuffle=True), epochs=8)
    vacc = evaluate(victim, loader(s.test_x, s.test_y))
    assert vacc > 0.5  # victim learned the task

    res = fidelity_vs_budget(victim, s, [250, 4000], epochs=6)
    assert res[1].thief_fidelity > res[0].thief_fidelity  # more queries -> more fidelity
    assert res[1].thief_fidelity > 0.6  # high-budget thief clones the victim well

    # defense: cap at 500 throttles the 4000-budget run below the unconstrained one
    defended = fidelity_vs_budget(victim, s, [4000], api_max_queries=500, epochs=6)
    assert defended[0].rejected
    assert defended[0].thief_fidelity < res[1].thief_fidelity
