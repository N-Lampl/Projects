#!/usr/bin/env python3
"""End-to-end IDS demo on SYNTHETIC flows: load -> build -> train -> evaluate.

Writes results/metrics.json and a confusion-matrix figure. Fully offline
(scikit-learn only). Run via `make run`.
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

from ids_pipeline import (  # noqa: E402
    build_pipeline,
    evaluate,
    load_data,
    predict_proba,
    set_seed,
    train,
)
from sklearn.metrics import roc_curve  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _plot_confusion(cm: list[list[int]]) -> Path:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["benign", "attack"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("IDS confusion matrix (synthetic test set)", pad=10)
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
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (recall)")
    ax.set_title("IDS ROC curve (synthetic test set)", pad=10)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "roc_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic IDS demo.")
    ap.add_argument("--n-samples", type=int, default=12000)
    ap.add_argument("--attack-fraction", type=float, default=0.25)
    ap.add_argument("--classifier", choices=["rf", "xgb"], default="rf")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--real", action="store_true", help="use real NSL-KDD (see data/README.md)")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading data (synthetic={not args.real})...")
    ds = load_data(
        synthetic=not args.real,
        n_samples=args.n_samples,
        attack_fraction=args.attack_fraction,
    )
    print(f"  source={ds.source}  train={len(ds.X_train)}  test={len(ds.X_test)}  features={ds.n_features}")

    print(f"building + training pipeline (classifier={args.classifier})...")
    pipe = build_pipeline(ds, classifier=args.classifier)
    train(pipe, ds)

    print("evaluating on the held-out test set...")
    m = evaluate(pipe, ds, threshold=args.threshold)
    print(f"  precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}")
    print(f"  roc_auc={m['roc_auc']:.3f}  precision@k={m['precision_at_k']}")

    cm_fig = _plot_confusion(m["confusion_matrix"])
    y_score = predict_proba(pipe, ds.X_test)
    roc_fig = _plot_roc(ds.y_test, y_score, m["roc_auc"])

    metrics = {
        "project": "shared/ids_pipeline",
        "summary": (
            f"RandomForest IDS on {ds.source} flows: "
            f"F1={m['f1']:.3f}, ROC-AUC={m['roc_auc']:.3f}, "
            f"recall={m['recall']:.3f} at threshold {args.threshold}."
        ),
        "source": ds.source,
        "classifier": args.classifier,
        "seed": 42,
        "n_train": int(len(ds.X_train)),
        "n_features": ds.n_features,
        **{k: m[k] for k in ("n_test", "n_attack_test", "threshold", "precision", "recall", "f1", "roc_auc")},
        "precision_at_k": m["precision_at_k"],
        "confusion_matrix": m["confusion_matrix"],
        "confusion_matrix_labelled": m["confusion_matrix_labelled"],
        "figures": [
            str(cm_fig.relative_to(PROJECT)),
            str(roc_fig.relative_to(PROJECT)),
        ],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {cm_fig.relative_to(PROJECT)}")
    print(f"wrote {roc_fig.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
