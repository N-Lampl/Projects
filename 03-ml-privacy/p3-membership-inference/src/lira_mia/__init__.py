"""Small online LiRA membership-inference attack on a self-trained model.

Public API:
    set_seed, get_device              -- reproducibility helpers
    make_synthetic_pool, Dataset      -- default offline population pool
    SmallMLP, train_model, accuracy   -- the target/shadow model
    logit_confidence                  -- the per-example LiRA signal
    make_warm_start, build_target,
      collect_shadow_signals          -- experiment drivers (warm-started shadows)
    ShadowSignals, lira_scores,
      roc_from_scores, auc,
      tpr_at_fpr                      -- the likelihood-ratio attack + metrics
"""

from .attack import (
    ShadowSignals,
    auc,
    lira_scores,
    roc_from_scores,
    tpr_at_fpr,
)
from .data import Dataset, make_synthetic_pool
from .model import SmallMLP, accuracy, logit_confidence, train_model
from .shadows import (
    TargetWorld,
    build_target,
    collect_shadow_signals,
    global_threshold_baseline,
    make_warm_start,
    target_confidences,
)
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "Dataset",
    "make_synthetic_pool",
    "SmallMLP",
    "train_model",
    "accuracy",
    "logit_confidence",
    "make_warm_start",
    "build_target",
    "collect_shadow_signals",
    "target_confidences",
    "global_threshold_baseline",
    "TargetWorld",
    "ShadowSignals",
    "lira_scores",
    "roc_from_scores",
    "auc",
    "tpr_at_fpr",
]
