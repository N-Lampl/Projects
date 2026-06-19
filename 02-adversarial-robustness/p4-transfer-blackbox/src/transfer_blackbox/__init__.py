"""Transferability + black-box attacks on small MNIST-style classifiers.

Public API:
    set_seed, get_device       -- reproducibility helpers
    SmallCNN, SmallMLP, build_model -- the two DIFFERENT target/surrogate nets
    get_loaders, make_synthetic     -- offline synthetic data (real MNIST optional)
    train, evaluate            -- training + clean-accuracy
    pgd_perturb                -- white-box craft on the surrogate
    transfer_accuracy          -- transfer of surrogate adversarials to the target
    QueryOracle                -- counts queries to the (black-box) target
    square_attack              -- score-based L-inf query attack (hand-rolled)
    boundary_attack            -- decision-based L2 query attack (hand-rolled)
    AttackResult               -- attack output dataclass
"""

from .attacks import (
    AttackResult,
    QueryOracle,
    boundary_attack,
    pgd_perturb,
    square_attack,
    transfer_accuracy,
)
from .data import get_loaders, make_synthetic
from .model import SmallCNN, SmallMLP, build_model
from .train import evaluate, train
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "SmallCNN",
    "SmallMLP",
    "build_model",
    "get_loaders",
    "make_synthetic",
    "train",
    "evaluate",
    "pgd_perturb",
    "transfer_accuracy",
    "QueryOracle",
    "square_attack",
    "boundary_attack",
    "AttackResult",
]
