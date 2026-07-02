"""Matplotlib figures written to results/figures/ (the committed evidence)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _smooth(x: np.ndarray, window: int = 11) -> np.ndarray:
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_training_return(curve: list[float], random_return: float, out: Path) -> Path:
    """Per-episode return of the PG agent vs the random-policy baseline line."""
    curve_arr = np.asarray(curve, dtype=np.float64)
    episodes = np.arange(len(curve_arr))
    smooth = _smooth(curve_arr)
    smooth_x = np.arange(len(smooth)) + (len(curve_arr) - len(smooth)) // 2

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(episodes, curve_arr, color="#2980b9", alpha=0.25, lw=1, label="return (raw)")
    ax.plot(smooth_x, smooth, color="#2980b9", lw=2, label="return (smoothed)")
    ax.axhline(
        random_return,
        ls="--",
        color="#c0392b",
        label=f"random policy ({random_return:.2f})",
    )
    ax.set_xlabel("episode")
    ax.set_ylabel("episode return")
    ax.set_title("Policy gradient learns to reach the goal (vs random)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_rlhf_winrate(metrics: dict, out: Path) -> Path:
    """Base vs RLHF policy: win-rate and average true reward."""
    winrate = metrics["rlhf_winrate"]
    base_r = metrics["base_true_reward"]
    rlhf_r = metrics["rlhf_true_reward"]
    opt_r = metrics["optimal_true_reward"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.2))

    ax1.bar(["RLHF wins", "base wins"], [winrate, 1 - winrate], color=["#27ae60", "#c0392b"])
    ax1.axhline(0.5, ls=":", color="black", label="coin flip")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("fraction of contexts")
    ax1.set_title(f"Win-rate under TRUE reward: {winrate:.2f}")
    ax1.legend(fontsize=8)

    ax2.bar(
        ["base\n(uniform)", "RLHF\npolicy", "optimal"],
        [base_r, rlhf_r, opt_r],
        color=["#c0392b", "#27ae60", "#7f8c8d"],
    )
    ax2.set_ylabel("avg true reward")
    ax2.set_title("Avg true reward per context")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_reward_model_fit(metrics: dict, out: Path) -> Path:
    """Learned vs true reward scatter across all (context, action) pairs."""
    learned = np.asarray(metrics["learned_reward"]).ravel()
    true = np.asarray(metrics["true_reward"]).ravel()
    corr = metrics["reward_corr"]
    acc = metrics["reward_model_acc"]

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.scatter(true, learned, s=18, alpha=0.7, color="#2980b9", edgecolor="none")
    # Best-fit line for visual reference.
    coef = np.polyfit(true, learned, 1)
    xs = np.linspace(true.min(), true.max(), 50)
    ax.plot(xs, coef[0] * xs + coef[1], ls="--", color="black", label="best fit")
    ax.set_xlabel("true reward  r*(context, action)")
    ax.set_ylabel("learned reward  r_hat")
    ax.set_title(
        f"Reward model recovers preference ordering\ncorr={corr:.2f}, held-out pref acc={acc:.2f}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
