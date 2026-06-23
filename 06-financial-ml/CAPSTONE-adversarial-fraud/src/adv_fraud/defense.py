"""Hardening: iterative adversarial training against the crafted evasions.

The defense is the standard adversarial-training loop adapted to tabular fraud,
run for several rounds (a tabular analogue of Madry et al.'s min-max training):

  1. Train the baseline model on clean data.
  2. Craft evasions for the fraud rows the *current* model still catches, under
     the same mutability / feasibility threat model used at test time.
  3. Append those evasions back into the training set, *still labelled fraud*
     (they are — only the model's belief changed), then refit.
  4. Repeat: each round attacks the freshly hardened model, so the defender keeps
     chasing the region the attacker keeps walking into.

A single round barely helps a linear model (the attacker just steps around the
shifted hyperplane), which is exactly why the loop iterates and why a non-linear
hardened head (gradient boosting) is offered: it can carve out the bounded fraud
region instead of relying on one separating line. We re-measure ASR with the SAME
attack so the before/after comparison is apples-to-apples.
"""

from __future__ import annotations

import numpy as np

from .attack import AttackConfig, evade
from .metrics import threshold_for_fpr
from .model import make_model


def adversarially_train(
    base_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    fpr_budget: float = 0.05,
    kind: str = "gboost",
    seed: int = 42,
    config: AttackConfig | None = None,
    rounds: int = 3,
):
    """Return a hardened model retrained on clean + crafted-evasion data.

    ``rounds`` controls how many attack/retrain iterations to run. The hardened
    head defaults to a non-linear gradient-boosting model, which can isolate the
    bounded fraud region a single hyperplane cannot; pass ``kind="logreg"`` to
    keep the defender linear for an apples-to-apples architecture comparison.
    """
    cfg = config or AttackConfig()

    X_aug = X_train.copy()
    y_aug = y_train.copy()
    current = base_model

    for _ in range(max(1, rounds)):
        scores = current.predict_proba(X_aug)[:, 1]
        thr = threshold_for_fpr(y_aug, scores, fpr_budget)

        # craft evasions for frauds the CURRENT model still catches
        fraud_mask = (y_aug == 1) & (scores >= thr)
        X_fraud = X_aug[fraud_mask]
        if len(X_fraud) > 0:
            X_evasions = evade(current, X_fraud, thr, cfg)
            X_aug = np.vstack([X_aug, X_evasions])
            y_aug = np.concatenate([y_aug, np.ones(len(X_evasions), dtype=int)])

        current = make_model(kind=kind, seed=seed)
        current.fit(X_aug, y_aug)

    return current
