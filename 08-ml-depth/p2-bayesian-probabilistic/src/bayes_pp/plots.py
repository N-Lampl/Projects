"""Matplotlib figures written to results/figures/ (the committed evidence)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_posterior_vs_true(ds, summary: dict, out: Path) -> Path:
    """Partial-pool posterior means + CI vs true means, with the baselines."""
    truth = ds.group_true_means
    means = np.asarray(summary["theta_mean"])
    ci = np.asarray(summary["theta_ci"])  # (J, 2)
    j = len(truth)
    x = np.arange(j)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    yerr = np.abs(ci.T - means)
    ax.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="o",
        color="#27ae60",
        capsize=4,
        label=f"partial pooling ({int(summary['level'] * 100)}% CI)",
        zorder=3,
    )
    ax.scatter(x, ds.group_ybar, marker="x", color="#c0392b", label="no pooling (MLE)", zorder=2)
    ax.scatter(x, truth, marker="_", s=400, color="black", label="true mean", zorder=4)
    ax.axhline(ds.y.mean(), ls=":", color="gray", label="complete pooling")
    ax.set_xlabel("group")
    ax.set_ylabel("estimated mean")
    ax.set_title("Partial pooling: shrunk estimates with honest intervals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_shrinkage(ds, shrink: dict, out: Path) -> Path:
    """Show each per-group MLE pulled toward the global mean by pooling."""
    no_pool = np.asarray(shrink["no_pooling_estimates"])
    partial = np.asarray(shrink["partial_estimates"])
    global_mean = shrink["global_mean"]
    j = len(no_pool)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i in range(j):
        ax.plot([0, 1], [no_pool[i], partial[i]], color="gray", lw=1, zorder=1)
    ax.scatter(np.zeros(j), no_pool, color="#c0392b", label="no pooling (MLE)", zorder=3)
    ax.scatter(np.ones(j), partial, color="#27ae60", label="partial pooling", zorder=3)
    ax.axhline(global_mean, ls=":", color="black", label="global mean")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no pooling", "partial pooling"])
    ax.set_ylabel("group mean estimate")
    ax.set_title(
        f"Shrinkage toward the global mean\n"
        f"RMSE: no-pool {shrink['no_pooling_rmse']:.2f} -> "
        f"partial {shrink['partial_pooling_rmse']:.2f}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_calibration(cal: dict, out: Path) -> Path:
    """Nominal vs empirical credible-interval coverage (calibration curve)."""
    levels = cal["levels"]
    empirical = cal["empirical_coverage"]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect calibration")
    ax.plot(levels, empirical, "o-", color="#2980b9", label="credible intervals")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Credible intervals are well calibrated")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
