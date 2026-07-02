"""The turn/battle loop.

Each turn, both sides act in order (BLUE then RED). For every living unit the side's policy
returns an :class:`Action` (an optional move followed by an optional attack); the engine validates
and applies it against the movement/LOS/combat rules. The loop ends at a victory condition or the
scenario turn cap and returns a :class:`BattleResult`. Optionally it records a per-turn snapshot
list for the dashboard's battle visualizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import combat
from .hexgrid import Hex
from .los import can_target
from .movement import reachable
from .state import GameState
from .units import BLUE, RED, Unit


@dataclass(slots=True)
class Action:
    unit_id: str
    move_to: Hex | None = None
    target_id: str | None = None


@dataclass(slots=True)
class BattleResult:
    winner: str  # "BLUE" | "RED" | "DRAW"
    turns: int
    blue_losses: int
    red_losses: int
    surviving_blue: int
    surviving_red: int
    seed: int


@dataclass(slots=True)
class Snapshot:
    turn: int
    units: list[tuple[str, str, str, float, int, int]]  # uid, side, kind, hp, q, r
    log: list[str] = field(default_factory=list)


def _apply_action(
    state: GameState, unit: Unit, action: Action, rng: np.random.Generator, model: str
) -> str | None:
    """Validate and apply one unit's action; return a combat-log line if a fight occurred."""
    if action.move_to is not None and not unit.has_moved:
        if action.move_to in reachable(state, unit):
            unit.pos = action.move_to
            unit.has_moved = True
    if action.target_id is not None and not unit.has_fired:
        target = state.units.get(action.target_id)
        if (
            target is not None
            and target.alive
            and target.side != unit.side
            and can_target(state, unit.pos, target.pos, unit.stats.rng)
        ):
            result = combat.resolve(state, unit, target, rng, model=model)
            combat.apply(unit, target, result)
            unit.has_fired = True
            return result.description
    return None


def _snapshot(state: GameState, log: list[str]) -> Snapshot:
    return Snapshot(
        turn=state.turn,
        units=[
            (u.uid, u.side, u.kind.value, round(u.hp, 1), u.pos.q, u.pos.r)
            for u in state.units.values()
            if u.alive
        ],
        log=list(log),
    )


def run_battle(
    state: GameState,
    blue_policy,
    red_policy,
    seed: int,
    model: str = "crt",
    record: bool = False,
) -> tuple[BattleResult, list[Snapshot]]:
    """Play a scenario to completion. Returns the result and (if ``record``) turn snapshots."""
    rng = np.random.default_rng(seed)
    init_blue = state.alive_count(BLUE)
    init_red = state.alive_count(RED)
    policies = {BLUE: blue_policy, RED: red_policy}
    snapshots: list[Snapshot] = []
    if record:
        snapshots.append(_snapshot(state, ["initial deployment"]))

    while not state.is_over():
        turn_log: list[str] = []
        # Randomise initiative each turn so neither side has a systematic first-mover edge;
        # deterministic given the battle seed.
        order = (BLUE, RED) if rng.random() < 0.5 else (RED, BLUE)
        for side in order:
            policy = policies[side]
            for unit in state.units_of(side):  # re-query each step; skips the dead
                if not unit.alive:
                    continue
                action = policy.choose_action(state, unit, rng)
                if action is not None:
                    line = _apply_action(state, unit, action, rng, model)
                    if line:
                        turn_log.append(line)
            if state.is_over():
                break
        state.record_objective_progress()
        if record:
            snapshots.append(_snapshot(state, turn_log))
        for unit in state.units.values():
            unit.reset_turn()
        state.turn += 1

    winner = state.winner() or "DRAW"
    surv_blue = state.alive_count(BLUE)
    surv_red = state.alive_count(RED)
    result = BattleResult(
        winner=winner,
        turns=state.turn - 1,
        blue_losses=init_blue - surv_blue,
        red_losses=init_red - surv_red,
        surviving_blue=surv_blue,
        surviving_red=surv_red,
        seed=seed,
    )
    return result, snapshots
