"""Command-line interface: ``dow <subcommand>``.

Subcommands: ``scenario list|show``, ``sim`` (one battle, turn-by-turn), ``montecarlo`` (win
probability + CI for a matchup), and ``sweep`` (a sensitivity table). All offline and deterministic.
"""

from __future__ import annotations

import argparse

from .engine import run_battle
from .montecarlo import monte_carlo, sensitivity_sweep
from .policies import POLICIES, make_policy
from .scenarios import list_scenarios, load_scenario
from .units import BLUE, RED

_POLICY_CHOICES = sorted(POLICIES)


def _cmd_scenario(args) -> None:
    if args.action == "list":
        for name, desc in list_scenarios():
            print(f"{name}\n    {desc}")
    else:
        state = load_scenario(args.name)
        print(f"Scenario: {args.name}")
        print(f"  map: {max(h.q for h in state.tiles) + 1} x {max(h.r for h in state.tiles) + 1}")
        print(f"  objective: {state.objective}  max_turns: {state.max_turns}")
        for side in (BLUE, RED):
            roster = ", ".join(
                f"{u.uid}:{u.kind.value}@({u.pos.q},{u.pos.r})" for u in state.units_of(side)
            )
            print(f"  {side}: {roster}")


def _cmd_sim(args) -> None:
    state = load_scenario(args.scenario)
    result, snapshots = run_battle(
        state, make_policy(args.blue), make_policy(args.red), seed=args.seed,
        model=args.model, record=True,
    )
    for snap in snapshots:
        print(f"--- turn {snap.turn} ---")
        for line in snap.log:
            print(f"  {line}")
    print(
        f"\nResult: {result.winner} in {result.turns} turns | "
        f"BLUE lost {result.blue_losses}, RED lost {result.red_losses}"
    )


def _cmd_montecarlo(args) -> None:
    mc = monte_carlo(
        args.scenario, args.blue, args.red, n=args.n, base_seed=args.seed,
        model=args.model, workers=args.workers,
    )
    print(f"Scenario: {args.scenario}  BLUE={args.blue} vs RED={args.red}  (n={args.n})")
    print(f"  BLUE win probability: {mc.blue_winrate:.3f}  95% CI ({mc.ci[0]:.3f}, {mc.ci[1]:.3f})")
    print(
        f"  wins B/R/draw: {mc.blue_wins}/{mc.red_wins}/{mc.draws}  "
        f"mean turns: {mc.mean_turns:.1f}"
    )
    print(
        f"  BLUE mean losses/battle {mc.blue_losses['mean']:.2f}, "
        f"RED {mc.red_losses['mean']:.2f}"
    )


def _cmd_sweep(args) -> None:
    sw = sensitivity_sweep(
        args.scenario, args.param, args.values, args.blue, args.red,
        n=args.n, base_seed=args.seed, model=args.model, workers=args.workers,
    )
    print(f"Sensitivity of BLUE win rate to {args.param} ({args.scenario}, n={args.n}/level):")
    for v, w, lo, hi in zip(sw["values"], sw["winrate"], sw["ci_low"], sw["ci_high"], strict=True):
        print(f"  {args.param}={v}: {w:.3f}  95% CI ({lo:.3f}, {hi:.3f})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dow", description="DOW tactical war-gaming simulator")
    sub = p.add_subparsers(dest="command", required=True)

    sc = sub.add_parser("scenario", help="list or show scenarios")
    sc.add_argument("action", choices=["list", "show"])
    sc.add_argument("name", nargs="?", default="meeting_engagement")
    sc.set_defaults(func=_cmd_scenario)

    common = dict(
        blue={"default": "aggressive", "choices": _POLICY_CHOICES},
        red={"default": "defensive", "choices": _POLICY_CHOICES},
    )

    sim = sub.add_parser("sim", help="play one battle turn-by-turn")
    sim.add_argument("scenario")
    sim.add_argument("--blue", **common["blue"])
    sim.add_argument("--red", **common["red"])
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--model", choices=["crt", "lanchester"], default="crt")
    sim.set_defaults(func=_cmd_sim)

    mc = sub.add_parser("montecarlo", help="win probability + CI for a matchup")
    mc.add_argument("scenario")
    mc.add_argument("--blue", **common["blue"])
    mc.add_argument("--red", **common["red"])
    mc.add_argument("--n", type=int, default=2000)
    mc.add_argument("--seed", type=int, default=42)
    mc.add_argument("--model", choices=["crt", "lanchester"], default="crt")
    mc.add_argument("--workers", type=int, default=None)
    mc.set_defaults(func=_cmd_montecarlo)

    sw = sub.add_parser("sweep", help="sensitivity of win rate to a parameter")
    sw.add_argument("scenario")
    sw.add_argument(
        "--param", default="red_count", choices=["red_count", "blue_count", "hold_turns"]
    )
    sw.add_argument("--values", type=int, nargs="+", default=[1, 2, 3])
    sw.add_argument("--blue", **common["blue"])
    sw.add_argument("--red", **common["red"])
    sw.add_argument("--n", type=int, default=500)
    sw.add_argument("--seed", type=int, default=42)
    sw.add_argument("--model", choices=["crt", "lanchester"], default="crt")
    sw.add_argument("--workers", type=int, default=None)
    sw.set_defaults(func=_cmd_sweep)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
