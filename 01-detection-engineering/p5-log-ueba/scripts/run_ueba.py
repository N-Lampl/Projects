#!/usr/bin/env python3
"""Generate synthetic auth events, build UEBA features, run the detector(s), and write
SOC-relevant metrics (precision@k, time-to-detect) + figures. Run via `make detect`.

Default path uses ONLY sklearn/numpy/pandas/matplotlib (offline, synthetic data).
Pass --autoencoder to additionally run the optional torch detector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from log_ueba import (  # noqa: E402
    GenConfig,
    build_features,
    generate_auth_events,
    isolation_forest_scores,
    set_seed,
    summary_metrics,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _plot_pr_at_k(results: dict[str, dict]) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, m in results.items():
        ks = sorted(int(k) for k in m["precision_at_k"])
        prec = [m["precision_at_k"][str(k)] for k in ks]
        ax.plot(ks, prec, "o-", linewidth=2, label=f"{name}")
    ax.set_xlabel("k (alerts an analyst reviews, ranked by score)")
    ax.set_ylabel("precision@k")
    ax.set_title("Precision@k: how clean is the top of the alert queue?", pad=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "precision_at_k.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_score_dist(scores: np.ndarray, labels: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    benign = scores[labels == 0]
    anom = scores[labels == 1]
    bins = np.linspace(scores.min(), scores.max(), 40)
    ax.hist(benign, bins=bins, alpha=0.6, label="benign", color="#2980b9", density=True)
    ax.hist(anom, bins=bins, alpha=0.7, label="lateral movement", color="#c0392b", density=True)
    ax.set_xlabel("IsolationForest anomaly score (higher = more anomalous)")
    ax.set_ylabel("density")
    ax.set_title("Anomaly-score separation: benign vs injected lateral movement", pad=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "score_distribution.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_queue(scores: np.ndarray, labels: np.ndarray, top: int = 100) -> Path:
    """The triage queue: top-N events by score, true anomalies marked red."""
    order = np.argsort(scores)[::-1][:top]
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = np.where(ranked_labels == 1, "#c0392b", "#bdc3c7")
    ax.bar(np.arange(len(order)), ranked_scores, color=colors, width=1.0)
    first_hit = np.where(ranked_labels == 1)[0]
    if len(first_hit):
        ax.axvline(first_hit[0], color="#27ae60", linestyle="--", linewidth=1.5,
                   label=f"first true hit @ rank {first_hit[0] + 1}")
        ax.legend()
    ax.set_xlabel(f"alert rank (top {top} by score)")
    ax.set_ylabel("anomaly score")
    ax.set_title("Triage queue (red = real lateral movement)", pad=12)
    fig.tight_layout()
    out = FIG_DIR / "triage_queue.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=60)
    ap.add_argument("--n-days", type=int, default=14)
    ap.add_argument("--contamination", type=float, default=0.02)
    ap.add_argument("--autoencoder", action="store_true", help="also run the torch AE detector")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cfg = GenConfig(n_users=args.n_users, n_days=args.n_days)
    df = generate_auth_events(cfg)
    n_events = len(df)
    n_anom = int(df["is_anomaly"].sum())
    print(f"generated {n_events} auth events; {n_anom} injected anomalies "
          f"({n_anom / n_events:.2%}); compromised users: {df.attrs['compromised_users']}")

    feats = build_features(df)
    labels = feats["is_anomaly"].to_numpy()
    timestamps = feats["timestamp"].to_numpy()
    entities = feats["user"].to_numpy()

    results: dict[str, dict] = {}

    print("fitting IsolationForest (unsupervised)...")
    if_scores = isolation_forest_scores(feats, contamination=args.contamination)
    results["IsolationForest"] = summary_metrics(if_scores, labels, timestamps, entities)

    if args.autoencoder:
        try:
            from log_ueba import autoencoder_scores

            print("training torch autoencoder...")
            ae_scores = autoencoder_scores(feats)
            results["Autoencoder"] = summary_metrics(ae_scores, labels, timestamps, entities)
        except ImportError:
            print("torch not installed -- skipping autoencoder path")

    # ---- report ----
    for name, m in results.items():
        ttd = m["time_to_detect"]
        print(f"\n[{name}]")
        print(f"  precision@10 = {m['precision_at_k']['10']:.2f}  "
              f"precision@25 = {m['precision_at_k']['25']:.2f}")
        print(f"  recall@50    = {m['recall_at_k']['50']:.2f}")
        print(f"  avg-precision= {m['average_precision']:.3f}   roc-auc = {m['roc_auc']:.3f}")
        print(f"  first true hit at rank {ttd['rank']} "
              f"({ttd['alerts_before']} false alerts first); "
              f"time-to-detect = {ttd['ttd_seconds']}s")

    fig1 = _plot_pr_at_k(results)
    fig2 = _plot_score_dist(if_scores, labels)
    fig3 = _plot_queue(if_scores, labels)

    best = results["IsolationForest"]
    metrics = {
        "project": "p5-log-ueba",
        "summary": (
            f"UEBA on {n_events} synthetic auth events ({n_anom} injected lateral-movement "
            f"anomalies). IsolationForest precision@25={best['precision_at_k']['25']:.2f}, "
            f"first true hit at rank {best['time_to_detect']['rank']}, "
            f"ROC-AUC={best['roc_auc']:.3f}."
        ),
        "seed": 42,
        "n_events": n_events,
        "n_anomalies": n_anom,
        "anomaly_rate": n_anom / n_events,
        "compromised_users": df.attrs["compromised_users"],
        "detectors": results,
        "figures": [
            str(fig1.relative_to(PROJECT)),
            str(fig2.relative_to(PROJECT)),
            str(fig3.relative_to(PROJECT)),
        ],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {fig1.relative_to(PROJECT)}")
    print(f"wrote {fig2.relative_to(PROJECT)}")
    print(f"wrote {fig3.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
