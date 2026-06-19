"""Certified L2 robustness via randomized smoothing (Cohen et al., 2019).

Public API:
    set_seed, get_device           -- reproducibility helpers
    SmallCNN                       -- the base classifier we smooth
    get_loaders, make_synthetic    -- offline synthetic data (+ optional MNIST)
    train, evaluate                -- noise-augmented training + base accuracy
    SmoothedClassifier             -- the randomized-smoothing certifier
    clopper_pearson_lower, norm_ppf-- the statistics behind the certificate
    certified_accuracy_at          -- certified accuracy at a given L2 radius
"""

from .data import get_loaders, make_synthetic
from .model import SmallCNN
from .smoothing import (
    ABSTAIN,
    SmoothedClassifier,
    certified_accuracy_at,
    clopper_pearson_lower,
    norm_ppf,
)
from .train import evaluate, train
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "SmallCNN",
    "get_loaders",
    "make_synthetic",
    "train",
    "evaluate",
    "SmoothedClassifier",
    "ABSTAIN",
    "clopper_pearson_lower",
    "norm_ppf",
    "certified_accuracy_at",
]
