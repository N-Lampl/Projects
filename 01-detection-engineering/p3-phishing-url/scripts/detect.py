#!/usr/bin/env python3
"""Train the phishing-URL detector and write figures + metrics.json.

Default path is fully offline: synthetic URLs -> lexical features -> sklearn
LogisticRegression. Use --data phiusiil for the optional real-data path (needs
ucimlrepo) and --model cnn for the optional torch char-CNN. Run via `make detect`.
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

from phishing_url import (  # noqa: E402
    FEATURE_NAMES,
    build_classifier,
    evaluate,
    extract_features,
    make_synthetic,
    set_seed,
    top_feature_weights,
    train_test_split_df,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _load_dataframe(source: str, n: int, seed: int):
    if source == "synthetic":
        return make_synthetic(n=n, phish_frac=0.5, seed=seed), "synthetic"
    if source == "phiusiil":
        from phishing_url import load_phiusiil

        return load_phiusiil(max_rows=n, seed=seed), "PhiUSIIL (ucimlrepo)"
    raise ValueError(f"unknown data source: {source!r}")


def _plot_roc(fpr, tpr, auc: float) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#c0392b", linewidth=2, label=f"detector (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#7f8c8d", linewidth=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (recall)")
    ax.set_title("Phishing-URL detector — ROC curve", pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "roc_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_weights(weights: list[tuple[str, float]]) -> Path:
    names = [w[0] for w in weights][::-1]
    vals = [w[1] for w in weights][::-1]
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in vals]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("signed weight  (red -> phishing, blue -> benign)")
    ax.set_title("Most influential lexical features", pad=10)
    fig.tight_layout()
    out = FIG_DIR / "feature_importance.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "phiusiil"], default="synthetic")
    ap.add_argument("--model", choices=["logreg", "rf", "cnn"], default="logreg")
    ap.add_argument("--n", type=int, default=4000, help="dataset size (or max rows for phiusiil)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df, source_name = _load_dataframe(args.data, args.n, args.seed)
    train_df, test_df = train_test_split_df(df, test_frac=0.25, seed=args.seed)
    print(f"data source: {source_name} | train={len(train_df)} test={len(test_df)} "
          f"| phishing frac (test)={test_df['label'].mean():.2f}")

    weights_fig = None
    top_weights: list[tuple[str, float]] = []

    if args.model in ("logreg", "rf"):
        X_tr = extract_features(train_df["url"])
        X_te = extract_features(test_df["url"])
        y_tr = train_df["label"].to_numpy()
        y_te = test_df["label"].to_numpy()
        pipe = build_classifier(kind=args.model, seed=args.seed)
        pipe.fit(X_tr, y_tr)
        results = evaluate(pipe, X_te, y_te)
        top_weights = top_feature_weights(pipe, FEATURE_NAMES, k=10)
        weights_fig = _plot_weights(top_weights)
        model_name = f"sklearn {args.model} over {len(FEATURE_NAMES)} lexical features"
    else:  # char-CNN (optional torch path)
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
        )

        from phishing_url.char_cnn import (
            build_cnn,
            encode_urls,
            predict_proba_cnn,
            train_cnn,
        )

        X_tr = encode_urls(train_df["url"].tolist())
        X_te = encode_urls(test_df["url"].tolist())
        y_tr = train_df["label"].to_numpy()
        y_te = test_df["label"].to_numpy()
        model = build_cnn()
        train_cnn(model, X_tr, y_tr, epochs=4)
        proba = predict_proba_cnn(model, X_te)
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_te, proba)
        results = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }
        model_name = "char-CNN (torch) over raw URL strings"

    print(f"model: {model_name}")
    for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {m:9s} = {results[m]:.4f}")

    roc_fig = _plot_roc(np.array(results["fpr"]), np.array(results["tpr"]), results["roc_auc"])

    figures = [str(roc_fig.relative_to(PROJECT))]
    if weights_fig is not None:
        figures.append(str(weights_fig.relative_to(PROJECT)))

    metrics = {
        "project": "p3-phishing-url",
        "summary": (
            f"{model_name} on {source_name} data: "
            f"ROC-AUC {results['roc_auc']:.3f}, F1 {results['f1']:.3f} "
            f"on {len(test_df)} held-out URLs."
        ),
        "data_source": source_name,
        "model": model_name,
        "seed": args.seed,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
        "roc_auc": results["roc_auc"],
        "top_features": [{"name": n, "weight": w} for n, w in top_weights],
        "figures": figures,
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {roc_fig.relative_to(PROJECT)}")
    if weights_fig is not None:
        print(f"wrote {weights_fig.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
