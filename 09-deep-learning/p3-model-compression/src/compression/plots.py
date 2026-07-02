"""Matplotlib figures written to results/figures/ (the committed evidence)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe
import matplotlib.pyplot as plt  # noqa: E402

_COLORS = {
    "baseline": "#2c3e50",
    "pruned": "#27ae60",
    "quantized": "#2980b9",
    "distilled": "#c0392b",
}


def plot_pareto_accuracy_vs_size(variants: dict, out: Path) -> Path:
    """Accuracy vs serialized size (MB) scatter — the compression Pareto view."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, m in variants.items():
        color = _COLORS.get(name, "#7f8c8d")
        ax.scatter(m["size_mb"], m["accuracy"], s=90, color=color, zorder=3, label=name)
        ax.annotate(
            name,
            (m["size_mb"], m["accuracy"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_xlabel("model size (MB, serialized state_dict)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs size: the compression Pareto frontier")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_latency_vs_variant(variants: dict, out: Path) -> Path:
    """Bar chart of per-forward-pass latency (ms) for each variant."""
    names = list(variants)
    latencies = [variants[n]["latency_ms"] for n in names]
    colors = [_COLORS.get(n, "#7f8c8d") for n in names]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(names, latencies, color=colors)
    ax.set_ylabel("latency (ms / forward pass, median)")
    ax.set_title("CPU inference latency by variant")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(latencies):
        ax.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
