"""Phishing-URL detection: lexical features + sklearn classifier (default),
char-CNN (torch) optional.

Public API:
    set_seed, get_device          -- reproducibility helpers
    make_synthetic                -- offline synthetic URL generator (default data)
    train_test_split_df           -- deterministic split
    load_phiusiil                 -- OPTIONAL real PhiUSIIL data via ucimlrepo
    FEATURE_NAMES, extract_one,
        extract_features          -- lexical feature extraction
    build_classifier, evaluate,
        top_feature_weights       -- the sklearn detector
"""

from .data import load_phiusiil, make_synthetic, train_test_split_df
from .features import FEATURE_NAMES, extract_features, extract_one
from .model import build_classifier, evaluate, top_feature_weights
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "make_synthetic",
    "train_test_split_df",
    "load_phiusiil",
    "FEATURE_NAMES",
    "extract_features",
    "extract_one",
    "build_classifier",
    "evaluate",
    "top_feature_weights",
]
