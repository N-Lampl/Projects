"""Model-extraction (model stealing) from scratch -- no ART, no attack library.

Train a victim classifier, expose it as a black-box label-only API, then train a
THIEF model purely on the victim's query responses. Draw the fidelity-vs-query-
budget curve and show a query-budget / rate-limit defense capping the thief.

Public API:
    set_seed, get_device          -- reproducibility helpers
    get_splits, Splits, loader    -- victim / attack-pool / test data splits
    MLP, make_victim, make_thief  -- the two small classifiers
    train, evaluate, agreement    -- training, accuracy, victim<->thief fidelity
    VictimAPI, QueryBudgetExceeded-- black-box API + the rate-limit defense
    steal_once, fidelity_vs_budget, StealResult -- the extraction attack
"""

from .data import Splits, get_splits, loader
from .extract import StealResult, fidelity_vs_budget, steal_once
from .model import MLP, make_thief, make_victim
from .train import agreement, evaluate, train
from .utils import get_device, set_seed
from .victim_api import QueryBudgetExceeded, VictimAPI

__all__ = [
    "set_seed",
    "get_device",
    "get_splits",
    "Splits",
    "loader",
    "MLP",
    "make_victim",
    "make_thief",
    "train",
    "evaluate",
    "agreement",
    "VictimAPI",
    "QueryBudgetExceeded",
    "steal_once",
    "fidelity_vs_budget",
    "StealResult",
]
