"""Fast, offline, deterministic tests for the DOW simulator.

They cover the geometry, mechanics, and a small Monte Carlo run, asserting real behaviour with no
network. The single large statistical check lives in ``test_full.py`` and is marked ``@slow``.
"""

from __future__ import annotations

import numpy as np

from dow_sim import (
    BLUE,
    RED,
    Hex,
    distance,
    line,
    list_scenarios,
    load_scenario,
    make_policy,
    monte_carlo,
    neighbors,
    run_battle,
    sensitivity_sweep,
    set_seed,
    solve_zero_sum_2x2,
    wilson_interval,
)
from dow_sim.combat import resolve
from dow_sim.los import has_los
from dow_sim.movement import reachable
from dow_sim.seeding import make_rng
from dow_sim.terrain import Terrain


# --- geometry --------------------------------------------------------------
def test_hex_distance_and_neighbors():
    origin = Hex(0, 0)
    assert distance(origin, origin) == 0
    assert len(neighbors(origin)) == 6
    for nb in neighbors(origin):
        assert distance(origin, nb) == 1
    assert distance(Hex(0, 0), Hex(3, 0)) == 3
    assert distance(Hex(0, 0), Hex(-1, -1)) == 2  # symmetry across the cube constraint


def test_hex_line_endpoints():
    a, b = Hex(0, 0), Hex(3, -1)
    path = line(a, b)
    assert path[0] == a and path[-1] == b
    assert len(path) == distance(a, b) + 1


# --- seeding / determinism -------------------------------------------------
def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    assert np.array_equal(a, b)


def test_make_rng_streams():
    same = make_rng(9).random(4)
    again = make_rng(9).random(4)
    other = make_rng(10).random(4)
    assert np.array_equal(same, again)
    assert not np.array_equal(same, other)


# --- terrain / LOS ---------------------------------------------------------
def test_los_blocked_by_forest():
    tiles = {Hex(q, 0): Terrain.CLEAR for q in range(5)}
    state = load_scenario("meeting_engagement")
    state.tiles = tiles
    assert has_los(state, Hex(0, 0), Hex(4, 0))  # all clear
    state.tiles[Hex(2, 0)] = Terrain.FOREST  # block the middle
    assert not has_los(state, Hex(0, 0), Hex(4, 0))
    # an endpoint in cover does not block: you can still see into a forested edge hex
    state.tiles[Hex(2, 0)] = Terrain.CLEAR
    state.tiles[Hex(4, 0)] = Terrain.FOREST
    assert has_los(state, Hex(0, 0), Hex(4, 0))


# --- movement --------------------------------------------------------------
def test_movement_respects_points_and_occupancy():
    state = load_scenario("meeting_engagement")
    unit = state.units_of(BLUE)[0]
    dests = reachable(state, unit)
    assert unit.pos not in dests  # the current hex is not a "destination"
    for hex_, cost in dests.items():
        assert 0 < cost <= unit.stats.move
        assert state.passable(hex_)
        assert not state.occupied(hex_)  # cannot end on another unit


# --- combat ----------------------------------------------------------------
def test_combat_is_bounded_and_deterministic():
    state = load_scenario("meeting_engagement")
    attacker = state.units_of(BLUE)[0]
    defender = state.units_of(RED)[0]
    r1 = resolve(state, attacker, defender, make_rng(3))
    r2 = resolve(state, attacker, defender, make_rng(3))
    assert r1.defender_damage == r2.defender_damage  # same seed -> same roll
    assert 0 <= r1.defender_damage <= defender.stats.max_hp
    assert 0 <= r1.attacker_damage <= attacker.stats.max_hp


# --- engine ----------------------------------------------------------------
def test_single_battle_terminates_with_valid_result():
    state = load_scenario("meeting_engagement")
    result, snaps = run_battle(
        state, make_policy("aggressive"), make_policy("defensive"), seed=1, record=True
    )
    assert result.winner in {BLUE, RED, "DRAW"}
    assert 1 <= result.turns <= state.max_turns + 1
    assert result.blue_losses >= 0 and result.red_losses >= 0
    assert len(snaps) >= 2  # at least the initial deployment + one turn


def test_battle_is_reproducible():
    def play():
        return run_battle(
            load_scenario("combined_arms"), make_policy("payoff"), make_policy("aggressive"), seed=7
        )[0]

    assert play() == play()  # same seed -> identical BattleResult


def test_different_seeds_produce_variety():
    # Different seeds must not all collapse to one outcome; sample a handful and expect variety.
    outcomes = {
        run_battle(
            load_scenario("meeting_engagement"),
            make_policy("aggressive"),
            make_policy("aggressive"),
            seed=s,
        )[0].winner
        for s in range(12)
    }
    assert len(outcomes) >= 2


# --- scenarios -------------------------------------------------------------
def test_all_scenarios_load_and_place_units_legally():
    for name, _desc in list_scenarios():
        state = load_scenario(name)
        assert state.alive_count(BLUE) > 0 and state.alive_count(RED) > 0
        for u in state.units.values():
            assert state.passable(u.pos)  # no unit deployed onto impassable terrain


# --- monte carlo / metrics -------------------------------------------------
def test_monte_carlo_small_is_valid_and_reproducible():
    a = monte_carlo("meeting_engagement", "aggressive", "defensive", n=30)
    b = monte_carlo("meeting_engagement", "aggressive", "defensive", n=30)
    assert 0.0 <= a.blue_winrate <= 1.0
    assert a.ci[0] <= a.blue_winrate <= a.ci[1]
    assert a.blue_wins + a.red_wins + a.draws == 30
    assert a.blue_winrate == b.blue_winrate  # deterministic base seed


def test_wilson_interval_brackets_and_degenerates():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_sensitivity_sweep_shape():
    sw = sensitivity_sweep("meeting_engagement", "red_count", [1, 2], n=20)
    assert sw["param"] == "red_count"
    assert len(sw["winrate"]) == 2
    assert all(0.0 <= w <= 1.0 for w in sw["winrate"])


# --- game theory -----------------------------------------------------------
def test_zero_sum_2x2_matching_pennies():
    # Matching pennies has a 50/50 mixed equilibrium and value 0.
    payoff = np.array([[1.0, -1.0], [-1.0, 1.0]])
    strat, value = solve_zero_sum_2x2(payoff)
    assert abs(strat[0] - 0.5) < 1e-9
    assert abs(value) < 1e-9


def test_zero_sum_2x2_saddle_point():
    # A dominant row gives a pure strategy.
    payoff = np.array([[3.0, 2.0], [1.0, 0.0]])
    strat, value = solve_zero_sum_2x2(payoff)
    assert strat[0] == 1.0 and value == 2.0
