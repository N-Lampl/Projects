#!/usr/bin/env python3
"""Run the confused-deputy / tool-abuse suite against the mock agent, measure the
unsafe-tool-invocation rate before vs after the guardrail, and write a figure +
metrics.json. Run via `make attack`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_tool_abuse import (  # noqa: E402
    MockLLM,
    default_scenarios,
    evaluate,
    set_seed,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _plot(before, after) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    rates = [before.unsafe_episode_rate * 100, after.unsafe_episode_rate * 100]
    bars = ax.bar(
        ["before\n(no guardrail)", "after\n(tool guardrail)"],
        rates,
        color=["#c0392b", "#27ae60"],
        width=0.55,
    )
    ax.set_ylabel("unsafe-tool-invocation rate (% of episodes)")
    ax.set_title("Agent tool-abuse: guardrail collapses unsafe tool calls", pad=12)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    for b, r in zip(bars, rates):
        ax.annotate(
            f"{r:.0f}%",
            (b.get_x() + b.get_width() / 2, r),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    fig.tight_layout()
    out = FIG_DIR / "unsafe_rate_before_after.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--provider",
        choices=["mock", "anthropic"],
        default="mock",
        help="mock = offline deterministic (default); anthropic = real API (needs key)",
    )
    ap.add_argument("--model", default="claude-sonnet-4-5", help="model id for --provider anthropic")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.provider == "anthropic":
        from agent_tool_abuse import AnthropicLLM

        llm = AnthropicLLM(model=args.model)
        print(f"using real Anthropic provider (model={args.model}) -- makes network calls")
    else:
        llm = MockLLM(obedient_to_tool_output=True)
        print("using deterministic MOCK LLM provider (offline)")

    scenarios = default_scenarios()
    attack_scenarios = [s for s in scenarios if not s.benign]
    print(f"{len(scenarios)} scenarios ({len(attack_scenarios)} attacks, "
          f"{len(scenarios) - len(attack_scenarios)} benign controls)")

    res = evaluate(scenarios, llm)
    before, after = res["before"], res["after"]

    print("\n-- BEFORE (no guardrail) --")
    for name, unsafe in before.per_scenario.items():
        print(f"  {'UNSAFE' if unsafe else 'safe  '}  {name}")
    print(f"  unsafe-episode rate: {before.unsafe_episode_rate * 100:.0f}% "
          f"({before.n_unsafe_calls} unsafe tool calls executed)")

    print("\n-- AFTER (tool guardrail) --")
    for name, unsafe in after.per_scenario.items():
        print(f"  {'UNSAFE' if unsafe else 'safe  '}  {name}")
    print(f"  unsafe-episode rate: {after.unsafe_episode_rate * 100:.0f}% "
          f"({after.n_unsafe_calls} unsafe tool calls executed, "
          f"{after.n_blocked_calls} blocked by guardrail)")

    fig = _plot(before, after)

    metrics = {
        "project": "p6-agent-tool-abuse",
        "summary": (
            f"Confused-deputy injections drove the agent to an unsafe tool call in "
            f"{before.unsafe_episode_rate * 100:.0f}% of episodes; the tool-call "
            f"guardrail reduced that to {after.unsafe_episode_rate * 100:.0f}% while "
            f"leaving benign tasks working."
        ),
        "provider": args.provider,
        "seed": 42,
        "n_scenarios": before.n_scenarios,
        "n_attack_scenarios": len(attack_scenarios),
        "unsafe_rate_before": round(before.unsafe_episode_rate, 4),
        "unsafe_rate_after": round(after.unsafe_episode_rate, 4),
        "unsafe_calls_before": before.n_unsafe_calls,
        "unsafe_calls_after": after.n_unsafe_calls,
        "blocked_calls": after.n_blocked_calls,
        "benign_calls_still_working": after.n_benign_executed,
        "per_scenario_unsafe_before": before.per_scenario,
        "per_scenario_unsafe_after": after.per_scenario,
        "figures": [str(fig.relative_to(PROJECT))],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {fig.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
