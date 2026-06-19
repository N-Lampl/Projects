"""The privacy-utility experiment driver.

Wires the pieces together for the overnight batch:

  1. Build ONE synthetic population pool + ONE balanced MIA query set (members /
     non-members) + ONE shared set of non-private shadow models.  (shared across
     all targets so the attacker is held fixed)
  2. For each requested epsilon in {inf, 3, 1, ...}: train the DP target with
     manual DP-SGD, record utility (train/test accuracy + the train-test gap).
  3. Re-run BOTH privacy attacks against that target:
        * LiRA membership inference  -> AUC, TPR@1%FPR
        * model extraction (label-only thief) -> stolen accuracy, fidelity
  4. Collect everything into a tidy list of per-epsilon results for plotting.

Every model (target, shadows, thief) is the same SmallMLP, so the only variable
across epsilon rows is the target's training procedure -- that is the whole point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .data import Dataset
from .dp_train import DPConfig, DPReport, train_dp_manual
from .mia import ShadowSignals, auc, lira_scores, roc_from_scores, tpr_at_fpr
from .model import SmallMLP, accuracy, logit_confidence, predict_labels


# ---------------------------------------------------------------------------
# Shared world: pool split, query set, shadow models (all attacker-side, fixed).
# ---------------------------------------------------------------------------


@dataclass
class SharedWorld:
    pool: Dataset
    train_idx: np.ndarray  # the *target population* members live here
    out_idx: np.ndarray  # held-out non-members + test set
    query_idx: np.ndarray  # balanced members / non-members for MIA
    is_member: np.ndarray
    attack_idx: np.ndarray  # unlabelled inputs the extraction thief may query
    test_idx: np.ndarray  # clean test set for utility + fidelity
    shadows: ShadowSignals


def _new_model(pool: Dataset) -> SmallMLP:
    return SmallMLP(n_features=pool.n_features, n_classes=pool.n_classes)


def _train_plain(model, X, y, epochs=30, lr=0.05, seed=0):
    """Plain (non-private) SGD -- used for the shared shadow models and as eps=inf."""
    model.train()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    n = Xt.shape[0]
    bs = min(128, n)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    return model.eval()


def build_shared_world(
    pool: Dataset,
    n_query: int = 400,
    n_shadows: int = 8,
    shadow_epochs: int = 60,
    shadow_lr: float = 0.1,
    seed: int = 1,
) -> SharedWorld:
    """Split the pool and train the shared, non-private shadow models once.

    For LiRA's per-example Gaussian calibration to be meaningful, the shadows must
    mimic the target's training regime (same architecture, comparable epochs/lr).
    We therefore default the shadow regime to the same well-fit settings the
    non-private target uses, then attack every DP target with these same shadows.
    """
    rng = np.random.RandomState(seed)
    n = pool.X.shape[0]
    perm = rng.permutation(n)
    # 50% pool is the target-population (members drawn here), 50% held out
    half = n // 2
    pop_idx = perm[:half]  # target trains on (a sample of) these
    out_idx = perm[half:]

    # the target will train on the whole population slice; members come from it
    train_idx = pop_idx
    n_each = n_query // 2
    q_members = rng.permutation(train_idx)[:n_each]
    q_nonmembers = rng.permutation(out_idx)[:n_each]
    query_idx = np.concatenate([q_members, q_nonmembers])
    is_member = np.concatenate(
        [np.ones(len(q_members), bool), np.zeros(len(q_nonmembers), bool)]
    )

    # extraction thief queries unlabelled inputs from the out pool; test set too
    rest_out = np.setdiff1d(out_idx, q_nonmembers, assume_unique=False)
    rng.shuffle(rest_out)
    attack_idx = rest_out[: len(rest_out) // 2]
    test_idx = rest_out[len(rest_out) // 2 :]

    # shared shadows: each trains on a random half of the FULL pool, non-private
    n_q = len(query_idx)
    phi = np.zeros((n_q, n_shadows), dtype=np.float64)
    member = np.zeros((n_q, n_shadows), dtype=bool)
    for s in range(n_shadows):
        srng = np.random.RandomState(1000 + s)
        in_idx = srng.permutation(n)[: n // 2]
        in_set = np.zeros(n, dtype=bool)
        in_set[in_idx] = True
        m = _new_model(pool)
        _train_plain(m, pool.X[in_idx], pool.y[in_idx], epochs=shadow_epochs,
                     lr=shadow_lr, seed=s)
        phi[:, s] = logit_confidence(m, pool.X[query_idx], pool.y[query_idx])
        member[:, s] = in_set[query_idx]

    return SharedWorld(
        pool=pool,
        train_idx=train_idx,
        out_idx=out_idx,
        query_idx=query_idx,
        is_member=is_member,
        attack_idx=attack_idx,
        test_idx=test_idx,
        shadows=ShadowSignals(phi=phi, member=member),
    )


# ---------------------------------------------------------------------------
# Per-epsilon evaluation: train DP target, run MIA + extraction.
# ---------------------------------------------------------------------------


@dataclass
class EpsResult:
    epsilon: float  # requested (inf for non-private)
    accounted_epsilon: float
    noise_multiplier: float
    train_acc: float
    test_acc: float
    gen_gap: float  # train_acc - test_acc (the memorisation signal)
    mia_auc: float
    mia_tpr_at_1pct: float
    steal_acc: float
    steal_fidelity: float
    report: DPReport = field(repr=False, default=None)
    roc: tuple = field(repr=False, default=None)


def _run_extraction(target, world: SharedWorld, epochs: int, seed: int) -> tuple[float, float]:
    """Label-only model extraction: thief queries target for hard labels, then clones.

    Returns (thief test accuracy, fidelity = agreement with target on the test set).
    Fidelity is the extraction-relevant number: a good clone agrees with the target
    even where the target is wrong. DP noise *lowers the target's own quality*, which
    indirectly caps how good (and how faithful) any clone can be.
    """
    pool = world.pool
    qx = pool.X[world.attack_idx]
    stolen_labels = predict_labels(target, qx)  # the only signal the thief gets

    thief = _new_model(pool)
    _train_plain(thief, qx, stolen_labels, epochs=epochs, seed=seed)

    test_x, test_y = pool.X[world.test_idx], pool.y[world.test_idx]
    steal_acc = accuracy(thief, test_x, test_y)
    target_pred = predict_labels(target, test_x)
    thief_pred = predict_labels(thief, test_x)
    fidelity = float((thief_pred == target_pred).mean())
    return steal_acc, fidelity


def evaluate_epsilon(
    world: SharedWorld,
    epsilon: float,
    *,
    epochs: int = 60,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
    lr: float = 0.1,
    thief_epochs: int = 40,
    seed: int = 7,
) -> EpsResult:
    """Train one DP target at `epsilon` and run both attacks against it."""
    pool = world.pool
    cfg = DPConfig(
        target_epsilon=None if math.isinf(epsilon) else epsilon,
        max_grad_norm=max_grad_norm,
        delta=delta,
        epochs=epochs,
        lr=lr,
    )
    target = _new_model(pool)
    target, report = train_dp_manual(
        target, pool.X[world.train_idx], pool.y[world.train_idx], cfg, seed=seed
    )

    train_acc = accuracy(target, pool.X[world.train_idx], pool.y[world.train_idx])
    test_acc = accuracy(target, pool.X[world.test_idx], pool.y[world.test_idx])

    # --- membership inference (shared shadows) ---
    phi_target = logit_confidence(target, pool.X[world.query_idx], pool.y[world.query_idx])
    scores = lira_scores(phi_target, world.shadows)
    fpr, tpr, _ = roc_from_scores(scores, world.is_member)
    mia_auc = auc(fpr, tpr)
    mia_tpr = tpr_at_fpr(fpr, tpr, 0.01)

    # --- model extraction ---
    steal_acc, fidelity = _run_extraction(target, world, thief_epochs, seed + 1)

    return EpsResult(
        epsilon=epsilon,
        accounted_epsilon=report.accounted_epsilon,
        noise_multiplier=report.noise_multiplier,
        train_acc=train_acc,
        test_acc=test_acc,
        gen_gap=train_acc - test_acc,
        mia_auc=mia_auc,
        mia_tpr_at_1pct=mia_tpr,
        steal_acc=steal_acc,
        steal_fidelity=fidelity,
        report=report,
        roc=(fpr, tpr),
    )
