"""Probabilistic combat resolution.

Two selectable, RNG-driven resolvers (deterministic given a generator):

* **CRT** (default) - a classic Combat Results Table. The attack:defense odds ratio (modified
  by the defender's terrain) selects a column; a d6 selects the row; the cell gives the damage
  dealt to defender and attacker as fractions of their max HP. Discrete, explainable, replayable.
* **Lanchester** - continuous aimed-fire attrition. Damage scales with the attacker's firepower
  and a stochastic multiplier, eroding HP smoothly. Feeds richer casualty distributions.

Both consume a ``numpy.random.Generator`` so a battle is fully reproducible from its seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import GameState
from .terrain import info
from .units import Unit

# CRT columns by odds ratio (attacker attack / defender effective defense).
_ODDS_COLUMNS = [0.5, 1.0, 2.0, 3.0, 5.0]  # boundaries -> 6 columns (indices 0..5)

# Rows are columns 0..5 (worst->best odds for the attacker); each holds the six d6 outcomes as
# (defender, attacker) fractions of max HP lost. Higher odds hurt the defender more and expose the
# attacker less; low odds risk an attacker repulse (the attacker takes the damage).
_CRT: dict[int, list[tuple[float, float]]] = {
    0: [(0.0, 0.30), (0.0, 0.25), (0.10, 0.20), (0.15, 0.15), (0.20, 0.10), (0.25, 0.0)],
    1: [(0.0, 0.25), (0.10, 0.20), (0.15, 0.15), (0.20, 0.10), (0.25, 0.05), (0.30, 0.0)],
    2: [(0.10, 0.20), (0.20, 0.15), (0.25, 0.10), (0.30, 0.10), (0.35, 0.05), (0.45, 0.0)],
    3: [(0.20, 0.15), (0.25, 0.10), (0.35, 0.10), (0.40, 0.05), (0.50, 0.0), (0.60, 0.0)],
    4: [(0.25, 0.10), (0.35, 0.05), (0.45, 0.05), (0.55, 0.0), (0.65, 0.0), (0.75, 0.0)],
    5: [(0.40, 0.0), (0.50, 0.0), (0.60, 0.0), (0.70, 0.0), (0.80, 0.0), (0.90, 0.0)],
}

_LANCHESTER_K = 0.6  # attrition coefficient tuning battle length


@dataclass(slots=True)
class CombatResult:
    defender_damage: float
    attacker_damage: float
    odds: float
    roll: int | None  # d6 roll for CRT; None for Lanchester
    description: str


def _odds_column(odds: float) -> int:
    col = 0
    for boundary in _ODDS_COLUMNS:
        if odds >= boundary:
            col += 1
    return col


def effective_defense(state: GameState, defender: Unit) -> float:
    """Defender's defense scaled by the terrain it occupies."""
    mod = info(state.terrain_at(defender.pos)).defense_mod
    return defender.stats.defense * mod


def resolve_crt(
    state: GameState, attacker: Unit, defender: Unit, rng: np.random.Generator
) -> CombatResult:
    dfn = effective_defense(state, defender)
    odds = attacker.stats.attack / dfn if dfn > 0 else float("inf")
    col = _odds_column(odds)
    roll = int(rng.integers(1, 7))  # d6 in [1, 6]
    def_frac, atk_frac = _CRT[col][roll - 1]
    def_dmg = def_frac * defender.stats.max_hp
    atk_dmg = atk_frac * attacker.stats.max_hp
    desc = (
        f"{attacker.uid} -> {defender.uid} | odds {odds:.2f} (col {col}) roll {roll}: "
        f"defender -{def_dmg:.1f}hp, attacker -{atk_dmg:.1f}hp"
    )
    return CombatResult(def_dmg, atk_dmg, odds, roll, desc)


def resolve_lanchester(
    state: GameState, attacker: Unit, defender: Unit, rng: np.random.Generator
) -> CombatResult:
    dfn = effective_defense(state, defender)
    odds = attacker.stats.attack / dfn if dfn > 0 else float("inf")
    # Aimed-fire attrition with a stochastic multiplier in ~[0.5, 1.5].
    mult = 0.5 + rng.random()
    def_dmg = _LANCHESTER_K * attacker.stats.attack / max(dfn, 1.0) * defender.stats.max_hp * mult
    def_dmg = min(def_dmg, defender.hp)
    # Defender returns a little fire if it survives and is in range is handled by the engine;
    # here the attacker takes light suppression proportional to how even the fight is.
    atk_dmg = 0.15 * attacker.stats.max_hp * mult / max(odds, 1.0)
    desc = (
        f"{attacker.uid} -> {defender.uid} | odds {odds:.2f} lanchester x{mult:.2f}: "
        f"defender -{def_dmg:.1f}hp, attacker -{atk_dmg:.1f}hp"
    )
    return CombatResult(def_dmg, atk_dmg, odds, None, desc)


def resolve(
    state: GameState,
    attacker: Unit,
    defender: Unit,
    rng: np.random.Generator,
    model: str = "crt",
) -> CombatResult:
    if model == "lanchester":
        return resolve_lanchester(state, attacker, defender, rng)
    return resolve_crt(state, attacker, defender, rng)


def apply(attacker: Unit, defender: Unit, result: CombatResult) -> None:
    """Apply a resolved combat's damage to the two units (clamped at zero HP)."""
    defender.hp = max(0.0, defender.hp - result.defender_damage)
    attacker.hp = max(0.0, attacker.hp - result.attacker_damage)


def expected_defender_damage(state: GameState, attacker: Unit, defender: Unit) -> float:
    """Mean defender HP loss under the CRT (used by the game-theoretic payoff policy)."""
    dfn = effective_defense(state, defender)
    odds = attacker.stats.attack / dfn if dfn > 0 else float("inf")
    col = _odds_column(odds)
    mean_frac = float(np.mean([cell[0] for cell in _CRT[col]]))
    return mean_frac * defender.stats.max_hp


def expected_attacker_damage(state: GameState, attacker: Unit, defender: Unit) -> float:
    """Mean attacker HP loss under the CRT (return fire / repulse risk)."""
    dfn = effective_defense(state, defender)
    odds = attacker.stats.attack / dfn if dfn > 0 else float("inf")
    col = _odds_column(odds)
    mean_frac = float(np.mean([cell[1] for cell in _CRT[col]]))
    return mean_frac * attacker.stats.max_hp
