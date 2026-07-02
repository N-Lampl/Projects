"""Matplotlib figures written to results/figures/ (the committed evidence)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe
import matplotlib.pyplot as plt  # noqa: E402

_METHOD_LABELS = {
    "naive": "Naive\n(diff in means)",
    "regression": "Regression\nadjustment",
    "ipw": "IPW",
    "aipw": "AIPW\n(doubly robust)",
}


def plot_ate_by_method(point: dict, aipw_se: float | None, out: Path) -> Path:
    """Bar of each estimate vs the true ATE, with the AIPW 95% CI drawn in."""
    methods = list(_METHOD_LABELS)
    estimates = [point["estimates"][m] for m in methods]
    true_ate = point["true_ate"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#c0392b", "#2980b9", "#2980b9", "#27ae60"]
    bars = ax.bar([_METHOD_LABELS[m] for m in methods], estimates, color=colors, alpha=0.85)
    if aipw_se is not None:
        ax.errorbar(
            len(methods) - 1,
            point["estimates"]["aipw"],
            yerr=1.959963984540054 * aipw_se,
            fmt="none",
            ecolor="black",
            capsize=6,
            lw=1.5,
        )
    ax.axhline(true_ate, ls="--", color="black", lw=1.3, label=f"true ATE = {true_ate:.2f}")
    for b, est in zip(bars, estimates, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, est, f"{est:.2f}", ha="center", va="bottom")
    ax.set_ylabel("estimated ATE")
    ax.set_title("Only adjustment recovers the true effect")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_covariate_balance(balance: dict, out: Path) -> Path:
    """Love plot: |SMD| per covariate before vs after IPW weighting."""
    before = balance["smd_before"]
    after = balance["smd_after"]
    idx = range(len(before))
    labels = [f"X{i}" for i in idx]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter([abs(v) for v in before], labels, color="#c0392b", label="unadjusted", zorder=3)
    ax.scatter([abs(v) for v in after], labels, color="#27ae60", label="IPW-weighted", zorder=3)
    for i in idx:
        ax.plot([abs(before[i]), abs(after[i])], [i, i], color="gray", lw=1, zorder=1)
    ax.axvline(0.1, ls="--", color="black", lw=1, label="0.1 balance threshold")
    ax.set_xlabel("|standardized mean difference|")
    ax.set_title("Weighting balances the confounders")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
