"""Runtime adversarial-input detector for an MNIST CNN.

Combines feature squeezing (bit-depth reduction + median blur) with cheap input
statistics, then trains a scikit-learn detector on clean vs FGSM examples.

Public API:
    set_seed, get_device          -- reproducibility helpers
    SmallCNN                      -- the target classifier we monitor
    get_loaders, synthetic_digits -- data (offline synthetic / optional MNIST)
    train, evaluate               -- target training + clean accuracy
    fgsm_perturb                  -- source of adversarial examples
    bit_depth_reduce, median_blur -- the two feature squeezers
    squeeze_scores, statistical_features, detector_features
    build_feature_dataset, train_detector, pick_threshold_at_fpr
    DetectorBundle                -- fitted scaler + classifier + threshold
"""

from .attack import fgsm_perturb
from .data import get_loaders, synthetic_digits
from .detector import (
    DetectorBundle,
    build_feature_dataset,
    pick_threshold_at_fpr,
    train_detector,
)
from .model import SmallCNN
from .squeeze import (
    FEATURE_NAMES,
    bit_depth_reduce,
    detector_features,
    median_blur,
    squeeze_scores,
    statistical_features,
)
from .train import evaluate, train
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "SmallCNN",
    "get_loaders",
    "synthetic_digits",
    "train",
    "evaluate",
    "fgsm_perturb",
    "bit_depth_reduce",
    "median_blur",
    "squeeze_scores",
    "statistical_features",
    "detector_features",
    "FEATURE_NAMES",
    "build_feature_dataset",
    "train_detector",
    "pick_threshold_at_fpr",
    "DetectorBundle",
]
