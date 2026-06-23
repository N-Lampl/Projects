"""Generate synthetic OHLCV, detect injected manipulations, write figures + metrics.

Default path uses only numpy / pandas / scikit-learn / matplotlib.

    python scripts/run_detect.py
    python scripts/run_detect.py --budget 0.02 --window 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from market_manip import (  # noqa: E402
    build_features,
    event_metrics,
    generate,
    isolation_forest_score,
    operating_point,
    ranking_metrics,
    rolling_zscore_score,
    set_seed,
    threshold_at_budget,
)
from market_manip.evaluate import pr_curve  # noqa: E402

FIG_DIR = ROOT / "results" / "figures"


def _plot_series(df, labels, flagged, events, path: Path) -> None:
    """Price + volume with injected windows shaded and flagged bars marked."""
    dates = df["date"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(dates, df["close"], color="#1f77b4", lw=1.0, label="close")
    ax1.set_ylabel("price")
    ax1.set_title("Synthetic OHLCV with injected manipulations (shaded) and flags (red)")

    for ev in events:
        c = "#ff9896" if ev.kind == "pump_dump" else "#c5b0d5"
        ax1.axvspan(dates.iloc[ev.start], dates.iloc[ev.end], color=c, alpha=0.5)

    flag_idx = np.where(flagged)[0]
    ax1.scatter(
        dates.iloc[flag_idx], df["close"].iloc[flag_idx],
        color="red", s=14, zorder=5, label="flagged bar",
    )
    ax1.legend(loc="upper left", fontsize=8)

    ax2.bar(dates, df["volume"], color="#7f7f7f", width=1.0)
    ax2.scatter(
        dates.iloc[flag_idx], df["volume"].iloc[flag_idx],
        color="red", s=14, zorder=5,
    )
    ax2.set_ylabel("volume")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_scores(df, scores, threshold, events, path: Path) -> None:
    """Anomaly-score timeline with the operating threshold and event windows."""
    dates = df["date"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, scores, color="#2ca02c", lw=0.9, label="IsolationForest score")
    ax.axhline(threshold, color="red", ls="--", lw=1.0, label="operating threshold")
    for ev in events:
        c = "#ff9896" if ev.kind == "pump_dump" else "#c5b0d5"
        ax.axvspan(dates.iloc[ev.start], dates.iloc[ev.end], color=c, alpha=0.5)
    ax.set_ylabel("anomaly score")
    ax.set_title("Anomaly-score timeline (higher = more anomalous)")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pr(scores_if, scores_z, labels, path: Path) -> None:
    """PR curves for both detectors (the imbalance-honest money plot)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for scores, name, color in (
        (scores_if, "IsolationForest", "#1f77b4"),
        (scores_z, "rolling z-score", "#ff7f0e"),
    ):
        precision, recall = pr_curve(scores, labels)
        ax.plot(recall, precision, color=color, lw=1.5, label=name)
    base = labels.mean()
    ax.axhline(base, color="gray", ls=":", lw=1.0, label=f"prevalence={base:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall (bar-level)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=1500, help="number of bars")
    ap.add_argument("--window", type=int, default=20, help="rolling feature window")
    ap.add_argument("--budget", type=float, default=0.05, help="per-bar alert budget")
    args = ap.parse_args()

    set_seed(args.seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    series = generate(n=args.n, seed=args.seed)
    df, events, labels = series.df, series.events, series.label
    feats = build_features(df, window=args.window)

    score_if = isolation_forest_score(feats, seed=args.seed)
    score_z = rolling_zscore_score(feats)

    # Primary detector = IsolationForest; threshold at the chosen alert budget.
    thr = threshold_at_budget(score_if, args.budget)
    flagged = score_if >= thr

    rank_if = ranking_metrics(score_if, labels)
    rank_z = ranking_metrics(score_z, labels)
    op = operating_point(score_if, labels, thr)
    evm = event_metrics(score_if, events, thr)

    _plot_series(df, labels, flagged, events, FIG_DIR / "price_volume_flagged.png")
    _plot_scores(df, score_if, thr, events, FIG_DIR / "anomaly_score_timeline.png")
    _plot_pr(score_if, score_z, labels, FIG_DIR / "pr_curve.png")

    lead = evm["median_lead_time_bars"]
    summary = (
        f"IsolationForest on {args.n} synthetic OHLCV bars "
        f"({labels.sum()} manipulated, prevalence {labels.mean():.1%}): "
        f"bar PR-AUC={rank_if['pr_auc']:.3f}, ROC-AUC={rank_if['roc_auc']:.3f}; "
        f"at a {args.budget:.0%} alert budget it catches "
        f"{evm['n_detected']}/{evm['n_events']} injected events "
        f"(event recall {evm['event_recall']:.0%}) "
        f"with median lead-time {lead:+.0f} bars before the worst bar."
    )

    metrics = {
        "project": "p5-market-manipulation",
        "summary": summary,
        "source": "synthetic",
        "detector": "isolation_forest",
        "baseline_detector": "rolling_zscore",
        "seed": args.seed,
        "n_bars": int(args.n),
        "window": int(args.window),
        "alert_budget": args.budget,
        "n_manipulated_bars": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "pr_auc": rank_if["pr_auc"],
        "roc_auc": rank_if["roc_auc"],
        "precision_at_k": rank_if["precision_at_k"],
        "baseline_pr_auc": rank_z["pr_auc"],
        "baseline_roc_auc": rank_z["roc_auc"],
        "operating_threshold": op["threshold"],
        "bar_recall": op["bar_recall"],
        "bar_precision": op["bar_precision"],
        "false_positive_rate": op["false_positive_rate"],
        "confusion": op["confusion"],
        "event_recall": evm["event_recall"],
        "n_events": evm["n_events"],
        "n_detected": evm["n_detected"],
        "median_lead_time_bars": evm["median_lead_time_bars"],
        "mean_lead_time_bars": evm["mean_lead_time_bars"],
        "event_recall_by_kind": evm["event_recall_by_kind"],
        "figures": [
            "results/figures/price_volume_flagged.png",
            "results/figures/anomaly_score_timeline.png",
            "results/figures/pr_curve.png",
        ],
    }

    out = ROOT / "results" / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(summary)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
