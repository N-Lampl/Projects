"""PGD adversarial training on MNIST (offline default: synthetic data).

Compare a STANDARD-trained CNN against a PGD-ADVERSARIALLY-trained one across an
L-inf epsilon sweep, producing robustness curves + metrics.json.

Public API:
    set_seed, get_device          -- reproducibility helpers
    SmallCNN                      -- the small target classifier
    get_loaders, make_synthetic   -- data (synthetic by default, MNIST optional)
    train, evaluate               -- standard / adversarial training + clean acc
    pgd_perturb                   -- the from-scratch PGD attack
    accuracy_under_attack         -- accuracy across an epsilon sweep under PGD
"""

from .attack import accuracy_under_attack, pgd_perturb
from .data import get_loaders, make_synthetic
from .model import SmallCNN
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
    "pgd_perturb",
    "accuracy_under_attack",
]
