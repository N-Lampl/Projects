"""DGA domain detection from char-level features (self-contained, synthetic data).

Public API:
    set_seed, get_device          -- reproducibility helpers
    make_dataset, train_test_split_df  -- synthetic benign/DGA domains
    extract_stats, shannon_entropy, FeatureExtractor  -- char features
    DGAClassifier                 -- the default sklearn detector
    EntropyBaseline               -- naive entropy-threshold baseline
    evaluate                      -- metrics bundle
"""

from .data import make_dataset, train_test_split_df
from .features import FeatureExtractor, extract_stats, shannon_entropy
from .model import DGAClassifier, EntropyBaseline, evaluate
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "make_dataset",
    "train_test_split_df",
    "extract_stats",
    "shannon_entropy",
    "FeatureExtractor",
    "DGAClassifier",
    "EntropyBaseline",
    "evaluate",
]
