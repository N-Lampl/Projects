#!/usr/bin/env python3
"""Train the DGA detector on synthetic domains, write figures + metrics.json.

Default path: scikit-learn LogisticRegression on char n-grams + entropy/length
stats, compared against a naive entropy-threshold baseline. Pass --lstm to also
train the optional torch char-LSTM. Run via `make detect`.
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
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dga_detection import (  # noqa: E402
    DGAClassifier,
    EntropyBaseline,
    evaluate,
    make_dataset,
    set_seed,
    train_test_split_df,
)
from dga_detection.features import stats_frame  # noqa: E402
from dga_detection.model import pr_points, roc_points  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _plot_roc(curves: dict[str, tuple], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_points(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc_score(y_true, y_score):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("DGA detection ROC: model vs entropy baseline", pad=12)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_pr(curves: dict[str, tuple], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_score) in curves.items():
        prec, rec, _ = pr_points(y_true, y_score)
        ax.plot(rec, prec, linewidth=2, label=name)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("DGA detection precision-recall", pad=12)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_entropy(domains: list[str], y: np.ndarray, threshold: float, out: Path) -> Path:
    """Entropy distribution: benign vs DGA, with the baseline threshold. Shows
    the dictionary-DGA overlap that defeats the naive detector."""
    ent = stats_frame(domains)["entropy"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(ent.min(), ent.max(), 40)
    ax.hist(ent[y == 0], bins=bins, alpha=0.6, label="benign", color="#2980b9")
    ax.hist(ent[y == 1], bins=bins, alpha=0.6, label="DGA", color="#c0392b")
    ax.axvline(threshold, color="black", linestyle="--", label=f"baseline thr={threshold:.2f}")
    ax.set_xlabel("Shannon entropy (bits/char) of domain label")
    ax.set_ylabel("count")
    ax.set_title("Why entropy alone is not enough (note the overlap)", pad=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _per_family_recall(df, y_pred) -> dict[str, float]:
    """Recall on each DGA family — exposes the baseline's blind spot."""
    out = {}
    for fam in sorted(df["family"].unique()):
        if fam == "benign":
            continue
        idx = (df["family"] == fam).to_numpy()
        out[fam] = float(y_pred[idx].mean())  # fraction flagged as DGA
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=4000)
    ap.add_argument("--lstm", action="store_true", help="also train the torch char-LSTM")
    ap.add_argument("--epochs", type=int, default=4, help="LSTM epochs")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = make_dataset(n_per_class=args.n_per_class)
    train_df, test_df = train_test_split_df(df)
    x_tr, y_tr = train_df["domain"].tolist(), train_df["label"].to_numpy()
    x_te, y_te = test_df["domain"].tolist(), test_df["label"].to_numpy()
    print(f"train={len(x_tr)}  test={len(x_te)}  (balanced benign/DGA)")

    # default model
    clf = DGAClassifier().fit(x_tr, y_tr)
    score_clf = clf.predict_proba(x_te)
    pred_clf = (score_clf >= 0.5).astype(int)
    m_clf = evaluate(y_te, score_clf, pred_clf)

    # baseline
    base = EntropyBaseline().fit(x_tr, y_tr)
    score_base = base.score(x_te)
    pred_base = base.predict(x_te)
    m_base = evaluate(y_te, score_base, pred_base)

    curves = {"logreg (ngram+stats)": (y_te, score_clf), "entropy baseline": (y_te, score_base)}

    print("\n=== LogisticRegression (default) ===")
    for k in ("roc_auc", "pr_auc", "accuracy", "precision_dga", "recall_dga", "false_positive_rate"):
        print(f"  {k:22s} {m_clf[k]:.4f}")
    print("\n=== Entropy baseline ===")
    for k in ("roc_auc", "accuracy", "recall_dga", "false_positive_rate"):
        print(f"  {k:22s} {m_base[k]:.4f}")

    fam_clf = _per_family_recall(test_df, pred_clf)
    fam_base = _per_family_recall(test_df, pred_base)
    print("\nper-family recall (model / baseline):")
    for fam in fam_clf:
        print(f"  {fam:8s}  {fam_clf[fam]:.3f} / {fam_base[fam]:.3f}")

    roc_fig = _plot_roc(curves, FIG_DIR / "roc_curve.png")
    pr_fig = _plot_pr(curves, FIG_DIR / "pr_curve.png")
    ent_fig = _plot_entropy(x_te, y_te, base.threshold, FIG_DIR / "entropy_distribution.png")
    figures = [roc_fig, pr_fig, ent_fig]

    lstm_metrics = None
    if args.lstm:
        from dga_detection.lstm import lstm_proba, train_lstm

        print(f"\ntraining char-LSTM for {args.epochs} epochs (CPU)...")
        lstm = train_lstm(x_tr, y_tr, epochs=args.epochs)
        score_lstm = lstm_proba(lstm, x_te)
        pred_lstm = (score_lstm >= 0.5).astype(int)
        lstm_metrics = evaluate(y_te, score_lstm, pred_lstm)
        curves["char-LSTM"] = (y_te, score_lstm)
        roc_fig = _plot_roc(curves, FIG_DIR / "roc_curve.png")  # redraw with LSTM
        print(f"  char-LSTM roc_auc={lstm_metrics['roc_auc']:.4f}  recall_dga={lstm_metrics['recall_dga']:.4f}")

    metrics = {
        "project": "p4-dga-detection",
        "summary": (
            f"LogReg on char n-grams + entropy/length stats detects synthetic DGA domains "
            f"at ROC-AUC {m_clf['roc_auc']:.3f} (recall {m_clf['recall_dga']:.3f}, "
            f"FPR {m_clf['false_positive_rate']:.3f}); the naive entropy baseline reaches only "
            f"ROC-AUC {m_base['roc_auc']:.3f} and misses the dictionary-DGA family."
        ),
        "seed": 42,
        "n_train": len(x_tr),
        "n_test": len(x_te),
        "model": m_clf,
        "entropy_baseline": m_base,
        "per_family_recall": {"model": fam_clf, "baseline": fam_base},
        "lstm": lstm_metrics,
        "figures": [str(f.relative_to(PROJECT)) for f in figures],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    for f in figures:
        print(f"wrote {f.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
