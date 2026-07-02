"""Monte Carlo battle runner and the analyses built on it.

Every battle ``i`` runs from seed ``base_seed + i`` on a freshly loaded scenario, so runs are
independent and the whole sweep is reproducible. Because the worker is a top-level function taking
only picklable arguments, batches parallelise across processes with ``concurrent.futures``.

Three analyses:
* :func:`monte_carlo` - win probability + Wilson CI, casualty and turn-length distributions.
* :func:`sensitivity_sweep` - how BLUE's win rate responds to one scenario parameter.
* :func:`compare` - an A/B win-rate delta using **common random numbers** (the same seed sequence
  for both arms) so the difference has a much tighter interval.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from .engine import BattleResult, run_battle
from .metrics import distribution_stats, wilson_interval
from .policies import make_policy
from .scenarios import load_scenario
from .state import GameState
from .units import BLUE, RED


@dataclass(slots=True)
class MonteCarloResult:
    scenario: str
    blue_policy: str
    red_policy: str
    n: int
    blue_wins: int
    red_wins: int
    draws: int
    blue_winrate: float
    ci: tuple[float, float]
    mean_turns: float
    blue_losses: dict[str, float] = field(default_factory=dict)
    red_losses: dict[str, float] = field(default_factory=dict)
    turns: dict[str, float] = field(default_factory=dict)


def _apply_param(state: GameState, param: str | None, value) -> None:
    """Mutate a freshly loaded state to realise a sensitivity-sweep parameter value."""
    if param is None:
        return
    if param in ("red_count", "blue_count"):
        side = RED if param == "red_count" else BLUE
        keep = int(value)
        uids = sorted(u.uid for u in state.units_of(side))
        for uid in uids[keep:]:
            state.units.pop(uid, None)
    elif param == "hold_turns":
        state.objective_hold_turns = int(value)
    else:
        raise ValueError(f"unknown sweep parameter {param!r}")


def _run_one(args) -> BattleResult:
    scenario, blue_name, red_name, seed, model, param, value = args
    state = load_scenario(scenario)
    _apply_param(state, param, value)
    result, _ = run_battle(state, make_policy(blue_name), make_policy(red_name), seed, model=model)
    return result


def _run_batch(arg_list, workers: int | None) -> list[BattleResult]:
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_run_one, arg_list))
    return [_run_one(a) for a in arg_list]


def monte_carlo(
    scenario: str,
    blue_policy: str = "aggressive",
    red_policy: str = "aggressive",
    n: int = 2000,
    base_seed: int = 42,
    model: str = "crt",
    workers: int | None = None,
    param: str | None = None,
    value=None,
) -> MonteCarloResult:
    """Run ``n`` battles and summarise BLUE's win probability and casualty distributions."""
    arg_list = [
        (scenario, blue_policy, red_policy, base_seed + i, model, param, value) for i in range(n)
    ]
    results = _run_batch(arg_list, workers)

    blue_wins = sum(r.winner == BLUE for r in results)
    red_wins = sum(r.winner == RED for r in results)
    draws = sum(r.winner == "DRAW" for r in results)
    winrate = blue_wins / n if n else 0.0
    return MonteCarloResult(
        scenario=scenario,
        blue_policy=blue_policy,
        red_policy=red_policy,
        n=n,
        blue_wins=blue_wins,
        red_wins=red_wins,
        draws=draws,
        blue_winrate=winrate,
        ci=wilson_interval(blue_wins, n),
        mean_turns=sum(r.turns for r in results) / n if n else 0.0,
        blue_losses=distribution_stats([r.blue_losses for r in results]),
        red_losses=distribution_stats([r.red_losses for r in results]),
        turns=distribution_stats([r.turns for r in results]),
    )


def sensitivity_sweep(
    scenario: str,
    param: str,
    values: list,
    blue_policy: str = "aggressive",
    red_policy: str = "defensive",
    n: int = 500,
    base_seed: int = 42,
    model: str = "crt",
    workers: int | None = None,
) -> dict:
    """Sweep ``param`` over ``values``; return BLUE win rate + CI at each level."""
    winrate, lo, hi = [], [], []
    for v in values:
        mc = monte_carlo(
            scenario, blue_policy, red_policy, n=n, base_seed=base_seed,
            model=model, workers=workers, param=param, value=v,
        )
        winrate.append(mc.blue_winrate)
        lo.append(mc.ci[0])
        hi.append(mc.ci[1])
    return {"param": param, "values": list(values), "winrate": winrate, "ci_low": lo, "ci_high": hi}


def compare(
    scenario: str,
    arm_a: tuple[str, str],
    arm_b: tuple[str, str],
    n: int = 1000,
    base_seed: int = 42,
    model: str = "crt",
    workers: int | None = None,
) -> dict:
    """A/B two (blue, red) policy arms with common random numbers; report the win-rate delta."""
    kw = dict(n=n, base_seed=base_seed, model=model, workers=workers)
    a = monte_carlo(scenario, arm_a[0], arm_a[1], **kw)
    b = monte_carlo(scenario, arm_b[0], arm_b[1], **kw)
    return {
        "arm_a": {"policies": arm_a, "winrate": a.blue_winrate, "ci": a.ci},
        "arm_b": {"policies": arm_b, "winrate": b.blue_winrate, "ci": b.ci},
        "delta": a.blue_winrate - b.blue_winrate,
        "common_random_numbers": True,
    }
