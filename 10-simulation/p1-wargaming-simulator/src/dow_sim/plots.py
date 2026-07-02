"""Analysis figures (matplotlib, Agg backend) written to ``results/figures/*.png``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .montecarlo import MonteCarloResult


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_winprob_by_scenario(results: dict[str, MonteCarloResult], path: str | Path) -> Path:
    """Bar chart of BLUE win probability with Wilson CI error bars, one bar per scenario."""
    names = list(results)
    rates = [results[n].blue_winrate for n in names]
    lows = [results[n].blue_winrate - results[n].ci[0] for n in names]
    highs = [results[n].ci[1] - results[n].blue_winrate for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(names, rates, yerr=[lows, highs], capsize=6, color="#2a5db0", alpha=0.85)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="even (50%)")
    ax.set_ylabel("BLUE win probability")
    ax.set_ylim(0, 1)
    ax.set_title("Win probability by scenario (95% Wilson CI)")
    ax.legend()
    plt.xticks(rotation=15, ha="right")
    return _save(fig, path)


def plot_casualty_hist(results_by_scenario: dict, path: str | Path) -> Path:
    """Histogram of BLUE vs RED casualties for a representative scenario's raw battle losses."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, (losses, color) in results_by_scenario.items():
        ax.hist(losses, bins=range(0, max(losses) + 2), alpha=0.6, label=label, color=color)
    ax.set_xlabel("units lost in a battle")
    ax.set_ylabel("count of battles")
    ax.set_title("Casualty distribution")
    ax.legend()
    return _save(fig, path)


def plot_sensitivity(sweep: dict, path: str | Path) -> Path:
    """Line plot of BLUE win rate vs a swept parameter, with a CI band."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = sweep["values"]
    ax.plot(x, sweep["winrate"], "-o", color="#2a5db0", label="BLUE win rate")
    ax.fill_between(x, sweep["ci_low"], sweep["ci_high"], alpha=0.2, color="#2a5db0")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel(sweep["param"])
    ax.set_ylabel("BLUE win probability")
    ax.set_ylim(0, 1)
    ax.set_title(f"Sensitivity to {sweep['param']}")
    ax.legend()
    return _save(fig, path)


def plot_ab_compare(comparison: dict, path: str | Path) -> Path:
    """Side-by-side win-rate bars for two policy arms (common random numbers)."""
    arms = [comparison["arm_a"], comparison["arm_b"]]
    labels = [f"BLUE={a['policies'][0]}\nRED={a['policies'][1]}" for a in arms]
    rates = [a["winrate"] for a in arms]
    lows = [a["winrate"] - a["ci"][0] for a in arms]
    highs = [a["ci"][1] - a["winrate"] for a in arms]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, rates, yerr=[lows, highs], capsize=6, color=["#2a5db0", "#7aa0d8"])
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylabel("BLUE win probability")
    ax.set_ylim(0, 1)
    ax.set_title(f"Policy A/B (delta {comparison['delta']:+.2f}, common random numbers)")
    return _save(fig, path)
