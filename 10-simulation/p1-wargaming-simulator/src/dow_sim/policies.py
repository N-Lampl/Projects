"""Heuristic AI policies (no LLM).

Every policy implements ``choose_action(state, unit, rng) -> Action | None``. They share small
helpers for finding enemies, moving toward a goal, and locating a hex from which a target can be
engaged. ``PayoffPolicy`` delegates target selection to the game-theoretic layer in
:mod:`dow_sim.payoff`.
"""

from __future__ import annotations

import numpy as np

from . import payoff
from .engine import Action
from .hexgrid import Hex, distance
from .los import can_target
from .movement import reachable
from .state import GameState
from .terrain import info
from .units import Unit, enemy_of


def _enemies(state: GameState, unit: Unit) -> list[Unit]:
    return state.units_of(enemy_of(unit.side))


def _nearest(unit: Unit, others: list[Unit]) -> Unit | None:
    return min(others, key=lambda o: distance(unit.pos, o.pos)) if others else None


def _positions(state: GameState, unit: Unit) -> dict[Hex, int]:
    """Reachable hexes plus the unit's current hex (staying put is always legal)."""
    opts = {unit.pos: 0}
    opts.update(reachable(state, unit))
    return opts


def _best_engagement(
    state: GameState, unit: Unit, targets: list[Unit]
) -> tuple[Hex | None, Unit | None, float]:
    """Best (position, target, expected-value) over hexes the unit can reach and fire from."""
    best_pos: Hex | None = None
    best_target: Unit | None = None
    best_score = float("-inf")
    for pos in _positions(state, unit):
        for t in targets:
            if can_target(state, pos, t.pos, unit.stats.rng):
                score = payoff.score_attack(state, unit, t)
                if score > best_score:
                    best_score, best_pos, best_target = score, pos, t
    return best_pos, best_target, best_score


def _move_toward(state: GameState, unit: Unit, goal: Hex) -> Hex | None:
    """The reachable hex that gets closest to ``goal`` (None if nothing improves)."""
    opts = reachable(state, unit)
    if not opts:
        return None
    here = distance(unit.pos, goal)
    best = min(opts, key=lambda h: distance(h, goal))
    return best if distance(best, goal) < here else None


class AggressivePolicy:
    """Close with the enemy and attack the best available target every turn."""

    name = "aggressive"

    def choose_action(
        self, state: GameState, unit: Unit, rng: np.random.Generator
    ) -> Action | None:
        targets = _enemies(state, unit)
        if not targets:
            return None
        pos, target, _score = _best_engagement(state, unit, targets)
        if target is not None:
            move_to = pos if pos != unit.pos else None
            return Action(unit.uid, move_to=move_to, target_id=target.uid)
        # No shot available: advance on the nearest enemy.
        goal = _nearest(unit, targets)
        step = _move_toward(state, unit, goal.pos) if goal else None
        return Action(unit.uid, move_to=step)


class DefensivePolicy:
    """Hold the best defensive terrain nearby; fire only when the exchange is favourable."""

    name = "defensive"

    def choose_action(
        self, state: GameState, unit: Unit, rng: np.random.Generator
    ) -> Action | None:
        targets = _enemies(state, unit)
        if not targets:
            return None
        pos, target, score = _best_engagement(state, unit, targets)
        if target is not None and score > 0:
            move_to = pos if pos != unit.pos else None
            return Action(unit.uid, move_to=move_to, target_id=target.uid)
        # No good shot: sidestep to the strongest defensive terrain within reach.
        opts = _positions(state, unit)
        best = max(opts, key=lambda h: info(state.terrain_at(h)).defense_mod)
        move_to = best if best != unit.pos else None
        return Action(unit.uid, move_to=move_to)


class ObjectivePolicy:
    """Push toward the objective hex (or the enemy if none), taking shots of opportunity."""

    name = "objective"

    def choose_action(
        self, state: GameState, unit: Unit, rng: np.random.Generator
    ) -> Action | None:
        targets = _enemies(state, unit)
        goal = state.objective
        # Fire if a shot is already available from a hex on the way; otherwise keep advancing.
        pos, target, score = (
            _best_engagement(state, unit, targets) if targets else (None, None, 0.0)
        )
        if goal is None:
            closer = True
        else:
            closer = pos is not None and distance(pos, goal) <= distance(unit.pos, goal)
        if target is not None and score > 0 and closer:
            move_to = pos if pos != unit.pos else None
            return Action(unit.uid, move_to=move_to, target_id=target.uid)
        if goal is not None:
            step = _move_toward(state, unit, goal)
            return Action(unit.uid, move_to=step)
        nearest = _nearest(unit, targets) if targets else None
        step = _move_toward(state, unit, nearest.pos) if nearest else None
        return Action(unit.uid, move_to=step)


class PayoffPolicy:
    """Maximise expected casualty exchange (game-theoretic); reposition when no shot pays off."""

    name = "payoff"

    def choose_action(
        self, state: GameState, unit: Unit, rng: np.random.Generator
    ) -> Action | None:
        targets = _enemies(state, unit)
        if not targets:
            return None
        pos, target, score = _best_engagement(state, unit, targets)
        if target is not None and score > 0:
            move_to = pos if pos != unit.pos else None
            return Action(unit.uid, move_to=move_to, target_id=target.uid)
        # Every available shot is a net loss: advance on the most profitable prospective target.
        goal = payoff.best_target(state, unit, targets)
        step = _move_toward(state, unit, goal.pos) if goal else None
        return Action(unit.uid, move_to=step)


POLICIES: dict[str, type] = {
    AggressivePolicy.name: AggressivePolicy,
    DefensivePolicy.name: DefensivePolicy,
    ObjectivePolicy.name: ObjectivePolicy,
    PayoffPolicy.name: PayoffPolicy,
}


def make_policy(name: str):
    """Instantiate a policy by name (used by the CLI, dashboard, and Monte Carlo runner)."""
    try:
        return POLICIES[name]()
    except KeyError as exc:
        raise ValueError(f"unknown policy {name!r}; choices: {sorted(POLICIES)}") from exc
