#!/usr/bin/env python3
"""Train the NIDS baseline and write SOC metrics + ROC and confusion-matrix figures.

Pipeline (load -> build -> train -> evaluate) is the shared ``ids_pipeline``
library, imported by path. This script adds the SOC operating-point report and
the figures. Fully offline on synthetic flows (scikit-learn only). Run via
``make detect``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nids_baseline import (  # noqa: E402
    get_pipeline_api,
    set_seed,
    soc_report,
    sweep_operating_points,
    threshold_for_alert_rate,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"

ALERT_BUDGETS = (0.01, 0.05, 0.10, 0.25)


def _plot_confusion(conf: dict) -> Path:
    cm = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["benign", "attack"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("NIDS confusion matrix (synthetic test set)", pad=10)
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > vmax * 0.5 else "black",
                fontsize=13, fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = FIG_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_roc(y_true, y_score, auc: float) -> Path:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (detection rate)")
    ax.set_title("NIDS ROC curve (synthetic test set)", pad=10)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "roc_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_budget_tradeoff(rows: list[dict]) -> Path:
    """Detection rate vs alert precision as the alert budget widens."""
    budgets = [r["target_alert_rate"] * 100 for r in rows]
    det = [r["detection_rate"] * 100 for r in rows]
    prec = [r["alert_precision"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(budgets, det, "o-", color="#27ae60", lw=2, label="detection rate (recall)")
    ax.plot(budgets, prec, "s-", color="#c0392b", lw=2, label="alert precision")
    ax.set_xlabel("alert budget (% of flows triaged)")
    ax.set_ylabel("percent")
    ax.set_title("SOC trade-off: detection vs precision per alert budget", pad=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    for b, d in zip(budgets, det):
        ax.annotate(f"{d:.0f}", (b, d), textcoords="offset points", xytext=(0, 8), fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "alert_budget_tradeoff.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="NIDS baseline on synthetic (or real NSL-KDD) flows.")
    ap.add_argument("--n-samples", type=int, default=12000)
    ap.add_argument("--attack-fraction", type=float, default=0.25)
    ap.add_argument("--classifier", choices=["rf", "xgb"], default="rf")
    ap.add_argument("--alert-budget", type=float, default=0.10,
                    help="fraction of flows to flag as alerts (sets the operating threshold)")
    ap.add_argument("--flows-per-day", type=int, default=1_000_000,
                    help="for extrapolating the daily false-alert load")
    ap.add_argument("--real", action="store_true", help="use real NSL-KDD (see data/README.md)")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    api = get_pipeline_api()

    print(f"loading data (synthetic={not args.real}) via shared ids_pipeline...")
    ds = api.load_data(
        synthetic=not args.real,
        n_samples=args.n_samples,
        attack_fraction=args.attack_fraction,
    )
    print(f"  source={ds.source}  train={len(ds.X_train)}  test={len(ds.X_test)}  features={ds.n_features}")

    print(f"building + training pipeline (classifier={args.classifier})...")
    pipe = api.build_pipeline(ds, classifier=args.classifier)
    api.train(pipe, ds)

    y_true = ds.y_test
    y_score = api.predict_proba(pipe, ds.X_test)

    # Operating point chosen by an alert budget, not an arbitrary 0.5 threshold.
    op_threshold = threshold_for_alert_rate(y_score, args.alert_budget)
    rep = soc_report(y_true, y_score, threshold=op_threshold, flows_per_day=args.flows_per_day)
    print(
        f"operating point @ {args.alert_budget:.0%} alert budget (thr={op_threshold:.3f}): "
        f"detection={rep['detection_rate']:.3f}  precision={rep['alert_precision']:.3f}  "
        f"ROC-AUC={rep['roc_auc']:.3f}"
    )

    # Also the shared library's own threshold-free metrics (precision@k etc).
    lib_metrics = api.evaluate(pipe, ds, threshold=op_threshold)

    sweep = sweep_operating_points(y_true, y_score, ALERT_BUDGETS)

    cm_fig = _plot_confusion(rep["confusion"])
    roc_fig = _plot_roc(y_true, y_score, rep["roc_auc"])
    tr_fig = _plot_budget_tradeoff(sweep)

    metrics = {
        "project": "p1-nids-baseline",
        "summary": (
            f"RandomForest NIDS on {ds.source} flows via shared ids_pipeline: "
            f"ROC-AUC={rep['roc_auc']:.3f}; at a {args.alert_budget:.0%} alert budget it "
            f"detects {rep['detection_rate']:.0%} of attacks at "
            f"{rep['alert_precision']:.0%} alert precision."
        ),
        "source": ds.source,
        "classifier": args.classifier,
        "seed": 42,
        "n_train": int(len(ds.X_train)),
        "n_features": ds.n_features,
        "alert_budget": args.alert_budget,
        "operating_threshold": op_threshold,
        "roc_auc": rep["roc_auc"],
        "detection_rate": rep["detection_rate"],
        "alert_precision": rep["alert_precision"],
        "miss_rate": rep["miss_rate"],
        "false_positive_rate": rep["false_positive_rate"],
        "alerts_per_day_est": rep["alerts_per_day_est"],
        "false_alerts_per_day_est": rep["false_alerts_per_day_est"],
        "confusion": rep["confusion"],
        "precision_at_k": lib_metrics["precision_at_k"],
        "alert_budget_sweep": [
            {
                "alert_budget": r["target_alert_rate"],
                "threshold": r["threshold"],
                "detection_rate": r["detection_rate"],
                "alert_precision": r["alert_precision"],
            }
            for r in sweep
        ],
        "figures": [
            str(roc_fig.relative_to(PROJECT)),
            str(cm_fig.relative_to(PROJECT)),
            str(tr_fig.relative_to(PROJECT)),
        ],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {roc_fig.relative_to(PROJECT)}")
    print(f"wrote {cm_fig.relative_to(PROJECT)}")
    print(f"wrote {tr_fig.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
