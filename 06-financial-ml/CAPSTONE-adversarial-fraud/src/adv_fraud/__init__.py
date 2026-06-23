"""Adversarial fraud capstone: attack AND defend a tabular fraud classifier.

Public API:
    set_seed, get_device          -- reproducibility helpers (CPU-only, no torch)
    make_dataset, mutable_indices -- seeded synthetic fraud data + threat model
    make_model, fraud_proba       -- the scikit-learn fraud classifier
    detection_report              -- PR-AUC / ROC-AUC / p@k / recall@FPR-budget
    evade, attack_success_rate    -- hand-rolled feasibility-constrained evasion
    feasible, AttackConfig        -- threat-model audit + attack knobs
    adversarially_train           -- the hardening defense
"""

from .attack import AttackConfig, attack_success_rate, evade, feasible
from .data import make_dataset, mutable_indices
from .defense import adversarially_train
from .metrics import detection_report, threshold_for_fpr
from .model import fraud_proba, make_model
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "make_dataset",
    "mutable_indices",
    "make_model",
    "fraud_proba",
    "detection_report",
    "threshold_for_fpr",
    "evade",
    "attack_success_rate",
    "feasible",
    "AttackConfig",
    "adversarially_train",
]
