"""Build the target world + warm-started shadow models and collect LiRA signals.

This is the experiment driver that `scripts/run_lira.py` calls. It stays small
and CPU-friendly: a shared warm-start checkpoint + 8-16 shadows, each trained for
a handful of epochs on a random half of the population pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .attack import ShadowSignals
from .data import Dataset
from .model import SmallMLP, accuracy, logit_confidence, train_model


@dataclass
class TargetWorld:
    """The model we attack + ground truth of which query examples were members."""

    model: SmallMLP
    query_idx: np.ndarray  # indices (into the pool) we run MIA on
    is_member: np.ndarray  # bool, aligned with query_idx
    train_acc: float
    test_acc: float


def _new_model(pool: Dataset) -> SmallMLP:
    return SmallMLP(n_features=pool.n_features, n_classes=pool.n_classes)


def make_warm_start(pool: Dataset, epochs: int = 8, seed: int = 0) -> dict:
    """A cheap shared checkpoint trained on a *disjoint* slice of the pool.

    Shadows warm-start from this so they need only a few epochs each. We train it
    on the last 25% of the pool, which we never use as query/membership data.
    """
    rng = np.random.RandomState(seed)
    n = pool.X.shape[0]
    ws_idx = rng.permutation(n)[: n // 4]
    m = _new_model(pool)
    train_model(m, pool.X[ws_idx], pool.y[ws_idx], epochs=epochs)
    return {k: v.clone() for k, v in m.state_dict().items()}


def build_target(
    pool: Dataset,
    n_query: int = 500,
    target_epochs: int = 40,
    warm_start: dict | None = None,
    seed: int = 1,
) -> TargetWorld:
    """Train the target on a random half of the first 75% of the pool.

    The query set = `n_query` examples, half of which the target trained on
    (members) and half it never saw (non-members) -- a balanced MIA evaluation.
    """
    rng = np.random.RandomState(seed)
    n = pool.X.shape[0]
    usable = rng.permutation(n)[: (3 * n) // 4]  # leave room for warm-start slice
    half = len(usable) // 2
    train_idx = usable[:half]
    out_idx = usable[half:]

    model = _new_model(pool)
    train_model(model, pool.X[train_idx], pool.y[train_idx], epochs=target_epochs,
                warm_start=warm_start)

    n_each = n_query // 2
    q_members = rng.permutation(train_idx)[:n_each]
    q_nonmembers = rng.permutation(out_idx)[:n_each]
    query_idx = np.concatenate([q_members, q_nonmembers])
    is_member = np.concatenate([np.ones(len(q_members), bool), np.zeros(len(q_nonmembers), bool)])

    tr_acc = accuracy(model, pool.X[train_idx], pool.y[train_idx])
    te_acc = accuracy(model, pool.X[out_idx], pool.y[out_idx])
    return TargetWorld(model=model, query_idx=query_idx, is_member=is_member,
                       train_acc=tr_acc, test_acc=te_acc)


def collect_shadow_signals(
    pool: Dataset,
    query_idx: np.ndarray,
    n_shadows: int = 12,
    shadow_epochs: int = 12,
    warm_start: dict | None = None,
    seed: int = 100,
    progress: bool = False,
) -> ShadowSignals:
    """Train `n_shadows`; record each shadow's confidence on every query example
    and whether that example was IN the shadow's training set.
    """
    n = pool.X.shape[0]
    n_q = len(query_idx)
    phi = np.zeros((n_q, n_shadows), dtype=np.float64)
    member = np.zeros((n_q, n_shadows), dtype=bool)

    for s in range(n_shadows):
        rng = np.random.RandomState(seed + s)
        # random half of the pool -> each query example is IN ~half the shadows
        in_idx = rng.permutation(n)[: n // 2]
        in_set = np.zeros(n, dtype=bool)
        in_set[in_idx] = True

        model = _new_model(pool)
        train_model(model, pool.X[in_idx], pool.y[in_idx], epochs=shadow_epochs,
                    warm_start=warm_start)

        phi[:, s] = logit_confidence(model, pool.X[query_idx], pool.y[query_idx])
        member[:, s] = in_set[query_idx]
        if progress:
            print(f"  shadow {s + 1}/{n_shadows} trained "
                  f"(IN={int(member[:, s].sum())}/{n_q} queries)")

    return ShadowSignals(phi=phi, member=member)


def target_confidences(model: SmallMLP, pool: Dataset, query_idx: np.ndarray) -> np.ndarray:
    """The target model's logit-confidence on each query example's true label."""
    return logit_confidence(model, pool.X[query_idx], pool.y[query_idx])


def global_threshold_baseline(phi_target: np.ndarray) -> np.ndarray:
    """A naive baseline: just use raw target confidence as the membership score.

    This is the 'threshold the loss' attack LiRA improves on -- we report both so
    the per-example calibration win is visible in the ROC.
    """
    return phi_target.astype(np.float64)


# convenience for set_seed import via package
def seed_torch(seed: int = 42) -> None:
    torch.manual_seed(seed)
