"""Matplotlib figures written to results/figures/ (the committed evidence)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402


def plot_accuracy(gcn_acc: float, mlp_acc: float, out: Path) -> Path:
    """Bar chart of GCN vs graph-blind MLP test accuracy."""
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    bars = ax.bar(
        ["MLP\n(graph-blind)", "GCN\n(message passing)"],
        [mlp_acc, gcn_acc],
        color=["#c0392b", "#27ae60"],
    )
    for bar, acc in zip(bars, (mlp_acc, gcn_acc), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.01,
            f"{acc:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Message passing over the graph beats the graph-blind baseline")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_tsne(embeddings: torch.Tensor, labels: torch.Tensor, out: Path, seed: int = 42) -> Path:
    """t-SNE of the GCN hidden embeddings, colored by true community label."""
    emb = embeddings.detach().cpu().numpy()
    lab = labels.detach().cpu().numpy()
    perplexity = min(30.0, max(5.0, (emb.shape[0] - 1) / 3.0))
    coords = TSNE(
        n_components=2, perplexity=perplexity, random_state=seed, init="pca"
    ).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for c in np.unique(lab):
        m = lab == c
        ax.scatter(coords[m, 0], coords[m, 1], s=12, label=f"class {c}", alpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("GCN hidden embeddings cluster by community (t-SNE)")
    ax.legend(fontsize=8, markerscale=1.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
