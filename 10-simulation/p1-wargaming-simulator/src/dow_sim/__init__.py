"""DOW: hex-grid tactical war-gaming with Monte Carlo analysis (CPU, offline, no LLM)."""

from __future__ import annotations

from .engine import Action, BattleResult, Snapshot, run_battle
from .hexgrid import Hex, distance, line, neighbors
from .metrics import distribution_stats, wilson_interval
from .montecarlo import MonteCarloResult, compare, monte_carlo, sensitivity_sweep
from .payoff import best_target, score_attack, solve_zero_sum_2x2
from .policies import (
    AggressivePolicy,
    DefensivePolicy,
    ObjectivePolicy,
    PayoffPolicy,
    make_policy,
)
from .scenarios import list_scenarios, load_scenario
from .seeding import make_rng, set_seed
from .state import GameState
from .terrain import Terrain
from .units import BLUE, RED, Unit, UnitKind, make_unit

__all__ = [
    "Action",
    "AggressivePolicy",
    "BLUE",
    "BattleResult",
    "DefensivePolicy",
    "GameState",
    "Hex",
    "MonteCarloResult",
    "ObjectivePolicy",
    "PayoffPolicy",
    "RED",
    "Snapshot",
    "Terrain",
    "Unit",
    "UnitKind",
    "best_target",
    "compare",
    "distance",
    "distribution_stats",
    "line",
    "list_scenarios",
    "load_scenario",
    "make_policy",
    "make_rng",
    "make_unit",
    "monte_carlo",
    "neighbors",
    "run_battle",
    "score_attack",
    "sensitivity_sweep",
    "set_seed",
    "solve_zero_sum_2x2",
    "wilson_interval",
]
