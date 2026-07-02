"""Matplotlib figures written to results/figures/ (the committed evidence)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_attention_induction(
    attn: np.ndarray, tokens: np.ndarray, layer: int, head: int, out: Path
) -> Path:
    """Heatmap of one head's attention on a single repeated sequence.

    A striped off-diagonal band is the induction-head signature: each query in
    the second half attends to the position just after its earlier occurrence.
    """
    t = attn.shape[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(attn, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlabel("key position (attended to)")
    ax.set_ylabel("query position")
    ax.set_title(f"Induction head attention — layer {layer}, head {head}")
    ax.set_xticks(range(t))
    ax.set_yticks(range(t))
    ax.set_xticklabels([str(int(x)) for x in tokens], fontsize=6)
    ax.set_yticklabels([str(int(x)) for x in tokens], fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attention weight")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_logit_lens(lens: dict, out: Path) -> Path:
    """Next-token accuracy and correct-token logit by residual-stream depth."""
    acc = lens["accuracy_by_layer"]
    logit = lens["correct_logit_by_layer"]
    labels = ["embed"] + [f"layer {i}" for i in range(1, len(acc))]
    x = np.arange(len(acc))

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(x, acc, "o-", color="#2980b9", label="next-token accuracy")
    ax1.set_xlabel("residual stream (depth)")
    ax1.set_ylabel("next-token accuracy", color="#2980b9")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis="y", labelcolor="#2980b9")

    ax2 = ax1.twinx()
    ax2.plot(x, logit, "s--", color="#c0392b", label="correct-token logit")
    ax2.set_ylabel("mean correct-token logit", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    ax1.set_title("Logit lens: predictions sharpen with depth")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_activation_patching(sweep: dict, out: Path) -> Path:
    """Fraction of the clean-corrupt logit gap recovered by patching each layer."""
    effects = sweep["effect_by_layer"]
    labels = ["embed"] + [f"layer {i}" for i in range(1, len(effects))]
    x = np.arange(len(effects))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = ["#27ae60" if e >= 0 else "#c0392b" for e in effects]
    ax.bar(x, effects, color=colors)
    ax.axhline(0.0, color="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of logit gap recovered")
    ax.set_title(f"Activation patching at position {sweep['position']}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
