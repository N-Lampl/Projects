"""FGSM-on-MNIST: the gradient-sign evasion attack, implemented from scratch.

Public API:
    set_seed, get_device      -- reproducibility helpers
    SmallCNN                  -- the 2-conv-layer target classifier
    get_loaders               -- MNIST data (pixels in [0, 1], no normalization)
    train, evaluate           -- training + clean-accuracy
    fgsm_perturb              -- the ~10-line core attack
    accuracy_under_attack     -- accuracy across an epsilon sweep
"""

from .attack import accuracy_under_attack, fgsm_perturb
from .data import get_loaders
from .model import SmallCNN
from .train import evaluate, train
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "SmallCNN",
    "get_loaders",
    "train",
    "evaluate",
    "fgsm_perturb",
    "accuracy_under_attack",
]
