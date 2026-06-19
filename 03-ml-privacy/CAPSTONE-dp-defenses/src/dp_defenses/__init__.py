"""CAPSTONE: differential privacy as a defense against MIA + model extraction.

Train the same MLP target at epsilon in {inf, 3, 1, ...} with DP-SGD, then re-run
membership inference (LiRA, shared shadow set) and model extraction against each
to chart the privacy-utility tradeoff.

Public API:
    set_seed, get_device                 -- reproducibility helpers
    make_synthetic_pool, Dataset         -- the offline population pool
    SmallMLP                             -- shared target/shadow/thief model
    DPConfig, DPReport                   -- DP-SGD configuration + audit report
    train_dp_manual                      -- manual DP-SGD (no Opacus needed)
    train_dp_opacus                      -- OPTIONAL Opacus path (lazy import)
    compute_epsilon, find_noise_multiplier -- the RDP accountant
    lira_scores, roc_from_scores, auc, tpr_at_fpr -- the MIA attack
    build_shared_world, evaluate_epsilon -- the experiment driver
"""

from .data import Dataset, make_synthetic_pool
from .dp_train import (
    DPConfig,
    DPReport,
    compute_epsilon,
    find_noise_multiplier,
    train_dp_manual,
    train_dp_opacus,
)
from .experiment import EpsResult, SharedWorld, build_shared_world, evaluate_epsilon
from .mia import ShadowSignals, auc, lira_scores, roc_from_scores, tpr_at_fpr
from .model import SmallMLP, accuracy, logit_confidence, predict_labels
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "Dataset",
    "make_synthetic_pool",
    "SmallMLP",
    "accuracy",
    "predict_labels",
    "logit_confidence",
    "DPConfig",
    "DPReport",
    "train_dp_manual",
    "train_dp_opacus",
    "compute_epsilon",
    "find_noise_multiplier",
    "ShadowSignals",
    "lira_scores",
    "roc_from_scores",
    "auc",
    "tpr_at_fpr",
    "SharedWorld",
    "EpsResult",
    "build_shared_world",
    "evaluate_epsilon",
]
