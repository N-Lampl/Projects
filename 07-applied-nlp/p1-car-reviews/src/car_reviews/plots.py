"""Matplotlib figure builders (headless, house palette). Each returns a Path.

Palette matches the rest of the portfolio: red ``#c0392b``, blue ``#2980b9``,
green ``#27ae60``, purple ``#8e44ad``, orange ``#d35400``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RED, BLUE, GREEN, PURPLE, ORANGE, DARK = (
    "#c0392b",
    "#2980b9",
    "#27ae60",
    "#8e44ad",
    "#d35400",
    "#2c3e50",
)
_CYCLE = [BLUE, ORANGE, GREEN, PURPLE, RED, DARK]


def plot_sentiment_vs_rating_confusion(
    confusion: list[list[int]], labels: list[int], out: Path, model_id: str
) -> Path:
    mat = np.array(confusion, dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(mat, cmap="Reds")
    ax.set_xticks(range(len(labels)), labels=[f"pred {x}" for x in labels])
    ax.set_yticks(range(len(labels)), labels=[f"true {x}" for x in labels])
    vmax = mat.max() if mat.max() > 0 else 1
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{int(mat[i, j]):,}",
                ha="center",
                va="center",
                color="white" if mat[i, j] > vmax / 2 else "black",
                fontsize=10,
            )
    ax.set_xlabel("predicted sentiment star")
    ax.set_ylabel("actual Rating")
    ax.set_title("Predicted sentiment vs. actual star rating", pad=10)
    fig.text(0.5, 0.005, model_id, ha="center", fontsize=7, color="grey")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _ranking_rows(table: pd.DataFrame, k: int) -> tuple[pd.DataFrame, list[str]]:
    tbl = table.sort_values("mean_sentiment_shrunk").reset_index(drop=True)
    if len(tbl) <= 2 * k:
        median = tbl["mean_sentiment_shrunk"].median()
        colors = [GREEN if v >= median else RED for v in tbl["mean_sentiment_shrunk"]]
        return tbl, colors
    rows = pd.concat([tbl.head(k), tbl.tail(k)]).reset_index(drop=True)
    colors = [RED] * k + [GREEN] * k
    return rows, colors


def plot_brand_ranking(table: pd.DataFrame, out: Path, k: int = 15, key: str = "make") -> Path:
    rows, colors = _ranking_rows(table, k)
    labels = [f"{r[key]}  (n={int(r['n'])})" for _, r in rows.iterrows()]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(rows))))
    y = range(len(rows))
    ax.barh(list(y), rows["mean_sentiment_shrunk"], color=colors)
    ax.set_yticks(list(y), labels=labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean sentiment (0 = negative, 1 = positive; shrunk)")
    ax.set_title("Brand sentiment ranking", pad=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_model_ranking(table: pd.DataFrame, out: Path, k: int = 15) -> Path:
    top = table.sort_values("mean_sentiment_shrunk").tail(k).reset_index(drop=True)
    labels = [f"{r['model_key']}  (n={int(r['n'])})" for _, r in top.iterrows()]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(top))))
    y = range(len(top))
    ax.barh(list(y), top["mean_sentiment_shrunk"], color=BLUE)
    ax.set_yticks(list(y), labels=labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean sentiment (0 = negative, 1 = positive; shrunk)")
    ax.set_title(f"Top {len(top)} models by sentiment", pad=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_aspect_heatmap(by_brand: pd.DataFrame, top_brands: list[str], out: Path) -> Path:
    cols = [b for b in top_brands if b in by_brand.columns]
    mat = by_brand[cols]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(mat))))
    data = mat.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(data), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), labels=cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(mat.index)), labels=list(mat.index), fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="mean sentiment")
    ax.set_title("Aspect sentiment by brand", pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_improvement(results: dict, out: Path) -> Path:
    """Grouped bars comparing baseline vs. calibrated vs. fine-tuned on the test set."""
    metrics = [
        ("exact_accuracy", "exact acc"),
        ("within_1_accuracy", "±1 acc"),
        ("spearman", "Spearman"),
        ("mae", "MAE (lower=better)"),
    ]
    stages = list(results)
    stage_color = {"baseline": RED, "calibrated": ORANGE, "fine-tuned": GREEN, "finetuned": GREEN}
    x = np.arange(len(metrics))
    width = 0.8 / max(len(stages), 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, stage in enumerate(stages):
        vals = [results[stage].get(k) or 0.0 for k, _ in metrics]
        bars = ax.bar(x + i * width, vals, width, label=stage, color=stage_color.get(stage, BLUE))
        for b, v in zip(bars, vals, strict=False):
            ax.annotate(
                f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8
            )
    ax.set_xticks(x + width * (len(stages) - 1) / 2, labels=[lbl for _, lbl in metrics])
    ax.set_ylabel("score")
    ax.set_title(
        "Making the model better: baseline → calibrated → fine-tuned (held-out test)", pad=10
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_topics_prevalence(topics: list[dict], out: Path) -> Path:
    labels = [t["label"] for t in topics]
    vals = [t["prevalence"] for t in topics]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [_CYCLE[i % len(_CYCLE)] for i in range(len(labels))]
    bars = ax.bar(range(len(labels)), vals, color=colors)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("share of reviews (dominant topic)")
    ax.set_title("Discovered topics and their prevalence", pad=10)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals, strict=False):
        ax.annotate(
            f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8
        )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
