"""Game-theoretic decision layer.

Two pieces:

* :func:`score_attack` scores a candidate engagement by its expected casualty exchange under the
  CRT (expected damage dealt minus expected damage taken). The payoff policy attacks the target
  that maximises this expected value.
* :func:`solve_zero_sum_2x2` solves a 2x2 zero-sum game for its mixed-strategy Nash equilibrium in
  closed form (no external solver). It backs a toy "which flank to commit" decision and is exposed
  so the analysis notebook can show an explicit equilibrium computation.
"""

from __future__ import annotations

import numpy as np

from . import combat
from .state import GameState
from .units import Unit


def score_attack(state: GameState, attacker: Unit, defender: Unit) -> float:
    """Expected value of attacking ``defender``: damage dealt minus damage taken (CRT means)."""
    dealt = combat.expected_defender_damage(state, attacker, defender)
    taken = combat.expected_attacker_damage(state, attacker, defender)
    # Reward finishing a wounded unit: value damage up to the defender's remaining HP.
    dealt = min(dealt, defender.hp)
    return dealt - taken


def best_target(state: GameState, attacker: Unit, defenders: list[Unit]) -> Unit | None:
    """The defender with the highest expected-value engagement, or None if the list is empty."""
    if not defenders:
        return None
    return max(defenders, key=lambda d: score_attack(state, attacker, d))


def solve_zero_sum_2x2(payoff: np.ndarray) -> tuple[np.ndarray, float]:
    """Mixed-strategy Nash for a 2x2 zero-sum game (row player maximises).

    ``payoff`` is the row player's 2x2 payoff matrix. Returns ``(row_strategy, game_value)``.
    Falls back to the pure saddle point when one exists.
    """
    a, b = payoff[0]
    c, d = payoff[1]
    # Pure-strategy saddle point (row maximin == column minimax).
    row_min = payoff.min(axis=1)
    col_max = payoff.max(axis=0)
    if row_min.max() == col_max.min():
        i = int(np.argmax(row_min))
        return (np.array([1.0, 0.0]) if i == 0 else np.array([0.0, 1.0]), float(row_min.max()))
    denom = a - b - c + d
    if denom == 0:
        return np.array([0.5, 0.5]), float(payoff.mean())
    p = (d - c) / denom
    p = float(np.clip(p, 0.0, 1.0))
    value = (a * d - b * c) / denom
    return np.array([p, 1.0 - p]), float(value)
