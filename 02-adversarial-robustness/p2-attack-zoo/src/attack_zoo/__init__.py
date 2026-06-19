"""attack-zoo: PGD, C&W-L2 and DeepFool evasion attacks, implemented from scratch.

Public API:
    set_seed, get_device      -- reproducibility helpers
    SmallCNN                  -- the small target classifier (MNIST/CIFAR/synthetic)
    get_loaders, make_synthetic -- data (default: offline synthetic; opt: CIFAR/MNIST)
    train, evaluate           -- training + clean-accuracy
    pgd, cw_l2, deepfool      -- the three attacks (one function each)
    run_attack                -- benchmark one attack -> metrics dict
"""

from .attacks import cw_l2, deepfool, pgd
from .data import get_loaders, make_synthetic
from .evaluate import run_attack
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
    "pgd",
    "cw_l2",
    "deepfool",
    "run_attack",
]
