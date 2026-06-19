"""The model-stealing attack, hand-rolled (no ART / no attack library).

Recipe (label-only, jacobian-free, in-distribution queries):
    1. Take a pool of unlabelled inputs the attacker can sample.
    2. Query the victim API for HARD labels on a budget of `n` of them.
    3. Train a fresh thief model on (input, victim_label) pairs.
    4. Report two numbers on a held-out test set:
         * accuracy  -- does the thief solve the task?
         * fidelity  -- does the thief AGREE with the victim's decisions?
       Fidelity is the metric that matters for extraction: a perfect clone agrees
       with the victim everywhere, even where the victim is wrong.

`fidelity_vs_budget` runs step 1-4 for a sweep of budgets to draw the
fidelity-vs-query-budget curve, and runs each budget against a rate-limited API
so we can show the defense capping what the thief can achieve.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import Splits, loader
from .model import make_thief
from .train import agreement, evaluate, train
from .victim_api import QueryBudgetExceeded, VictimAPI


@dataclass
class StealResult:
    budget_requested: int
    queries_used: int
    rejected: bool
    thief_test_acc: float
    thief_fidelity: float  # agreement with victim on the test set


def label_pool_with_victim(
    api: VictimAPI, x: torch.Tensor, batch_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Query the victim for hard labels on every row of `x`.

    Returns (queried_x, victim_labels, rejected). If the API's budget is hit
    mid-way, we keep whatever labels we already obtained and flag `rejected`.
    """
    xs, ys = [], []
    rejected = False
    for i in range(0, x.shape[0], batch_size):
        batch = x[i : i + batch_size]
        try:
            labels = api.predict(batch)
        except QueryBudgetExceeded:
            rejected = True
            break
        xs.append(batch)
        ys.append(labels)
    if not xs:
        empty = x[:0]
        return empty, torch.empty(0, dtype=torch.int64), True
    return torch.cat(xs), torch.cat(ys), rejected


def steal_once(
    victim: torch.nn.Module,
    splits: Splits,
    budget: int,
    *,
    api_max_queries: int | None = None,
    epochs: int = 8,
    seed: int = 42,
    device: torch.device | None = None,
) -> StealResult:
    """Run one extraction at a given query budget against a (possibly capped) API."""
    torch.manual_seed(seed)
    api = VictimAPI(victim, max_queries=api_max_queries, device=device)

    pool = splits.attack_x[:budget]
    qx, qy, rejected = label_pool_with_victim(api, pool)

    thief = make_thief(splits.img_size, splits.n_classes)
    if qx.shape[0] > 0:
        train(thief, loader(qx, qy, shuffle=True), epochs=epochs, device=device)

    test = loader(splits.test_x, splits.test_y)
    acc = evaluate(thief, test, device=device)
    fid = agreement(thief, victim, test, device=device)
    return StealResult(
        budget_requested=budget,
        queries_used=api.queries_used,
        rejected=rejected,
        thief_test_acc=acc,
        thief_fidelity=fid,
    )


def fidelity_vs_budget(
    victim: torch.nn.Module,
    splits: Splits,
    budgets: list[int],
    *,
    api_max_queries: int | None = None,
    epochs: int = 8,
    seed: int = 42,
    device: torch.device | None = None,
) -> list[StealResult]:
    """Sweep query budgets, returning one StealResult per budget.

    Pass `api_max_queries` to simulate the rate-limit defense (the same cap is
    applied at every budget point, so budgets above the cap get throttled)."""
    results = []
    for b in budgets:
        res = steal_once(
            victim,
            splits,
            b,
            api_max_queries=api_max_queries,
            epochs=epochs,
            seed=seed,
            device=device,
        )
        results.append(res)
    return results
