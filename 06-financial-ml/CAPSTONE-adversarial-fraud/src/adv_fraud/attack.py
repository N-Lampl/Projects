"""Hand-rolled evasion attack against a tabular fraud model (numpy only).

Goal: take transactions the model *correctly flags as fraud* and minimally
perturb them so the model scores them **below its operating threshold** - i.e.
the fraud slips through - while respecting what a real fraudster can actually do:

  1. MUTABILITY     only ``amount / hour / merchant_risk / distance_from_home /
                    n_items`` may change; account history is server-side and
                    immutable.
  2. BOX BOUNDS     each mutable feature stays inside a plausible range.
  3. INTEGER FIELDS ``hour`` and ``n_items`` are rounded to integers.
  4. CONSISTENCY    ``amount`` cannot collapse to an implausibly tiny fraction
                    of the account's own 30-day average (a $0.01 charge is not a
                    realistic large-purchase fraud); enforced as a floor.

The search is a **greedy coordinate / finite-difference descent**: at each step
we numerically estimate d(score)/d(feature) for every mutable feature (the
sklearn pipeline is not differentiable, so we use central finite differences -
the "gradient-style" part), take a normalized step downhill in score space,
project back into the feasible set, and repeat. This is a faithful tabular
analogue of FGSM/PGD without any attack library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import BOUNDS, FEATURES, INTEGER_FEATURES, MUTABLE


@dataclass
class AttackConfig:
    steps: int = 40
    step_frac: float = 0.06  # step size as a fraction of each feature's range
    fd_eps_frac: float = 1e-3  # finite-difference probe size (fraction of range)
    consistency_amount_floor: float = 0.05  # amount >= 5% of avg_amount_30d


def _feature_ranges() -> dict[str, float]:
    return {f: (hi - lo) for f, (lo, hi) in BOUNDS.items()}


def _mutable_idx() -> list[int]:
    return [i for i, f in enumerate(FEATURES) if MUTABLE[f]]


def _project(X: np.ndarray, X0: np.ndarray) -> np.ndarray:
    """Clip mutable features to bounds, round integers, enforce consistency.

    Immutable columns are forced back to their original values, so an attack can
    never touch account history.
    """
    X = X.copy()
    avg_amount_idx = FEATURES.index("avg_amount_30d")
    amount_idx = FEATURES.index("amount")

    for i, f in enumerate(FEATURES):
        if not MUTABLE[f]:
            X[:, i] = X0[:, i]  # immutable: revert
            continue
        lo, hi = BOUNDS[f]
        X[:, i] = np.clip(X[:, i], lo, hi)
        if f in INTEGER_FEATURES:
            X[:, i] = np.round(X[:, i])

    # consistency: amount must remain a plausible fraction of historical spend
    floor = 0.05 * X0[:, avg_amount_idx]
    X[:, amount_idx] = np.maximum(X[:, amount_idx], np.minimum(floor, BOUNDS["amount"][1]))
    return X


def feasible(X_adv: np.ndarray, X0: np.ndarray) -> np.ndarray:
    """Boolean mask: which rows are valid evasions w.r.t. the threat model.

    A row is feasible iff (a) no immutable feature moved and (b) every mutable
    feature is inside bounds. Integer/consistency are guaranteed by projection,
    so feasibility here is the honest audit of the *constraint contract*.
    """
    ok = np.ones(len(X_adv), dtype=bool)
    for i, f in enumerate(FEATURES):
        if not MUTABLE[f]:
            ok &= np.isclose(X_adv[:, i], X0[:, i])
        else:
            lo, hi = BOUNDS[f]
            ok &= (X_adv[:, i] >= lo - 1e-6) & (X_adv[:, i] <= hi + 1e-6)
    return ok


def evade(
    model,
    X0: np.ndarray,
    threshold: float,
    config: AttackConfig | None = None,
) -> np.ndarray:
    """Greedy finite-difference evasion. Returns perturbed feature matrix.

    ``model`` must expose ``predict_proba``; column 1 is treated as P(fraud).
    The objective is to drive that probability below ``threshold``.
    """
    cfg = config or AttackConfig()
    ranges = _feature_ranges()
    mut = _mutable_idx()

    X = X0.copy().astype(float)
    step_abs = {i: cfg.step_frac * ranges[FEATURES[i]] for i in mut}
    fd_abs = {i: cfg.fd_eps_frac * ranges[FEATURES[i]] for i in mut}

    def score(M: np.ndarray) -> np.ndarray:
        return model.predict_proba(M)[:, 1]

    for _ in range(cfg.steps):
        s = score(X)
        active = s >= threshold  # only keep attacking rows still detected
        if not active.any():
            break

        # central finite-difference gradient over mutable features
        grad = np.zeros_like(X)
        for i in mut:
            h = fd_abs[i]
            Xp = X.copy()
            Xm = X.copy()
            Xp[:, i] += h
            Xm[:, i] -= h
            grad[:, i] = (score(Xp) - score(Xm)) / (2.0 * h)

        # normalized descent step (lower the score) only on active rows
        gmut = grad[:, mut]
        norm = np.linalg.norm(gmut, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        direction = gmut / norm  # unit vector per row in mutable subspace

        for j, i in enumerate(mut):
            move = active * (-direction[:, j]) * step_abs[i]
            X[:, i] = X[:, i] + move

        X = _project(X, X0)

    return _project(X, X0)


def attack_success_rate(
    model,
    X0: np.ndarray,
    threshold: float,
    config: AttackConfig | None = None,
) -> dict:
    """Evade a set of *originally-detected* frauds; report ASR + feasibility.

    ASR = fraction of attacked frauds that end up scored below threshold AND
    remain feasible (an infeasible "evasion" is not counted as a success).
    """
    X_adv = evade(model, X0, threshold, config)
    s_before = model.predict_proba(X0)[:, 1]
    s_after = model.predict_proba(X_adv)[:, 1]

    feas = feasible(X_adv, X0)
    evaded = s_after < threshold
    success = evaded & feas

    return {
        "X_adv": X_adv,
        "score_before": s_before,
        "score_after": s_after,
        "feasible": feas,
        "evaded": evaded,
        "asr": float(np.mean(success)) if len(X0) else 0.0,
        "feasibility_rate": float(np.mean(feas)) if len(X0) else 0.0,
        "mean_score_drop": float(np.mean(s_before - s_after)) if len(X0) else 0.0,
    }
