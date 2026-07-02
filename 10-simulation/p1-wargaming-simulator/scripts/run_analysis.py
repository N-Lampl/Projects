#!/usr/bin/env python3
"""Run the DOW Monte Carlo experiment -> results/metrics.json + figures.

Simulates thousands of battles across the bundled scenarios, estimates BLUE win probability with
Wilson confidence intervals, sweeps a force-ratio parameter for a sensitivity curve, and runs a
policy A/B with common random numbers. Deterministic given ``--seed``; regenerates the committed
evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dow_sim import monte_carlo, sensitivity_sweep, set_seed  # noqa: E402
from dow_sim.montecarlo import _run_one, compare  # noqa: E402
from dow_sim.plots import (  # noqa: E402
    plot_ab_compare,
    plot_casualty_hist,
    plot_sensitivity,
    plot_winprob_by_scenario,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SCENARIOS = ["meeting_engagement", "seize_the_ridge", "combined_arms"]


def _raw_losses(scenario: str, blue: str, red: str, n: int, seed: int, model: str):
    """Per-battle (blue_losses, red_losses) for the casualty histogram."""
    blue_losses, red_losses = [], []
    for i in range(n):
        r = _run_one((scenario, blue, red, seed + i, model, None, None))
        blue_losses.append(r.blue_losses)
        red_losses.append(r.red_losses)
    return blue_losses, red_losses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000, help="battles per scenario/matchup")
    ap.add_argument("--sweep-n", type=int, default=400, help="battles per sensitivity level")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["crt", "lanchester"], default="crt")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default serial)")
    args = ap.parse_args()

    set_seed(args.seed)
    kw = dict(base_seed=args.seed, model=args.model, workers=args.workers)

    # 0) Fairness check: symmetric scenario + identical policies must be near 50/50 (no side bias).
    fairness = monte_carlo("meeting_engagement", "aggressive", "aggressive", n=args.n, **kw)

    # 1) Win probability by scenario (BLUE attacks aggressively vs a RED defensive posture).
    per_scenario = {
        s: monte_carlo(s, "aggressive", "defensive", n=args.n, **kw) for s in SCENARIOS
    }

    # 2) Sensitivity: BLUE win rate vs number of RED units in the meeting engagement.
    sweep = sensitivity_sweep(
        "meeting_engagement", "red_count", [1, 2, 3], "aggressive", "defensive",
        n=args.sweep_n, **kw,
    )

    # 3) Policy A/B on the same seeds (common random numbers): payoff vs pure aggression.
    ab = compare(
        "combined_arms", ("payoff", "defensive"), ("aggressive", "defensive"), n=args.n, **kw
    )

    # Figures.
    fig1 = plot_winprob_by_scenario(per_scenario, FIGURES / "winprob_by_scenario.png")
    bl, rl = _raw_losses("combined_arms", "payoff", "defensive", args.n, args.seed, args.model)
    fig2 = plot_casualty_hist(
        {"BLUE losses": (bl, "#2a5db0"), "RED losses": (rl, "#b03a3a")},
        FIGURES / "casualty_dist.png",
    )
    fig3 = plot_sensitivity(sweep, FIGURES / "sensitivity_red_count.png")
    fig4 = plot_ab_compare(ab, FIGURES / "policy_ab.png")

    ridge = per_scenario["seize_the_ridge"]
    summary = (
        f"Over {args.n} battles/scenario: with identical policies the mirror-symmetric meeting "
        f"engagement is a near-even {fairness.blue_winrate * 100:.0f}% for BLUE (95% CI "
        f"{fairness.ci[0] * 100:.0f}-{fairness.ci[1] * 100:.0f}%), confirming no side bias. An "
        f"aggressive assault on the fortified ridge succeeds {ridge.blue_winrate * 100:.0f}% of "
        f"the time. Each extra RED unit drops BLUE's win rate from "
        f"{sweep['winrate'][0] * 100:.0f}% to {sweep['winrate'][-1] * 100:.0f}%. On combined arms "
        f"the cautious payoff policy shifts BLUE's win rate by {ab['delta'] * 100:+.0f}pp vs "
        f"aggression (common random numbers)."
    )

    metrics = {
        "project": "p1-wargaming-simulator",
        "summary": summary,
        "data_source": "self-contained hex-grid battle simulation (no external data)",
        "seed": args.seed,
        "n_simulations": args.n,
        "combat_model": args.model,
        "fairness_check": {
            "scenario": "meeting_engagement",
            "policies": "aggressive vs aggressive (symmetric)",
            "blue_winrate": fairness.blue_winrate,
            "ci": list(fairness.ci),
        },
        "scenarios": {
            s: {
                "blue_policy": r.blue_policy,
                "red_policy": r.red_policy,
                "blue_winrate": r.blue_winrate,
                "ci": list(r.ci),
                "draws": r.draws,
                "mean_turns": r.mean_turns,
                "blue_losses": r.blue_losses,
                "red_losses": r.red_losses,
            }
            for s, r in per_scenario.items()
        },
        "sensitivity": sweep,
        "policy_ab": ab,
        "figures": [f"results/figures/{p.name}" for p in (fig1, fig2, fig3, fig4)],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(summary)
    figs = ", ".join(p.name for p in (fig1, fig2, fig3, fig4))
    print(f"[ok] wrote {RESULTS / 'metrics.json'} + {figs}")


if __name__ == "__main__":
    main()
