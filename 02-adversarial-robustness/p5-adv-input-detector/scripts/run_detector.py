#!/usr/bin/env python3
"""Train the target CNN (if needed), manufacture FGSM examples, train the runtime
adversarial-input detector, and write ROC + PR figures, a feature plot, and
metrics.json. Run via `make detect`.
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
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adv_detector import (  # noqa: E402
    FEATURE_NAMES,
    SmallCNN,
    build_feature_dataset,
    evaluate,
    get_device,
    get_loaders,
    set_seed,
    train_detector,
)
from adv_detector.train import load_model, save_model, train  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"
WEIGHTS = PROJECT / "models" / "smallcnn.pt"


def _get_model(device: torch.device, epochs: int, dataset: str) -> SmallCNN:
    if WEIGHTS.exists():
        print(f"loading weights <- {WEIGHTS.relative_to(PROJECT)}")
        return load_model(WEIGHTS, device)
    print("no weights found - training a fresh target model...")
    train_loader, test_loader = get_loaders(dataset=dataset)
    model = SmallCNN()
    train(model, train_loader, epochs=epochs, device=device)
    acc = evaluate(model, test_loader, device=device)
    print(f"target clean accuracy: {acc * 100:.1f}%")
    save_model(model, WEIGHTS)
    return model.eval()


def _plot_roc(y: np.ndarray, scores: np.ndarray, op_fpr: float, op_tpr: float) -> Path:
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"detector (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
    ax.scatter([op_fpr], [op_tpr], color="#c0392b", zorder=5,
               label=f"operating point\n(FPR={op_fpr:.2f}, TPR={op_tpr:.2f})")
    ax.set_xlabel("false positive rate (clean flagged)")
    ax.set_ylabel("true positive rate (FGSM caught)")
    ax.set_title("Adversarial-input detector — ROC", pad=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "detector_roc.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_pr(y: np.ndarray, scores: np.ndarray) -> Path:
    prec, rec, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec, prec, color="#27ae60", lw=2, label=f"AP={ap:.3f}")
    ax.set_xlabel("recall (fraction of attacks caught)")
    ax.set_ylabel("precision")
    ax.set_title("Adversarial-input detector — Precision/Recall", pad=10)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "detector_pr.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_features(X: np.ndarray, y: np.ndarray) -> Path:
    """Show how the two flagship features separate clean vs adversarial."""
    fig, ax = plt.subplots(figsize=(6, 5))
    clean = y == 0
    ax.scatter(X[clean, 2], X[clean, 3], s=8, alpha=0.4, color="#2980b9", label="clean")
    ax.scatter(X[~clean, 2], X[~clean, 3], s=8, alpha=0.4, color="#c0392b", label="FGSM adv")
    ax.set_xlabel(f"{FEATURE_NAMES[2]} (softmax shift under squeezing)")
    ax.set_ylabel(f"{FEATURE_NAMES[3]} (input total variation)")
    ax.set_title("Why it works: squeezing shift vs total variation", pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "feature_separation.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "mnist"], default="synthetic",
                    help="synthetic (offline default) or real MNIST (optional, downloads ~11MB)")
    ap.add_argument("--epsilon", type=float, default=0.2, help="FGSM L-inf budget for adv examples")
    ap.add_argument("--bits", type=int, default=2, help="bit-depth-reduction squeezer precision")
    ap.add_argument("--kernel", type=int, default=3, help="median-blur kernel size")
    ap.add_argument("--target-fpr", type=float, default=0.05, help="max clean false-positive rate")
    ap.add_argument("--epochs", type=int, default=2, help="epochs if auto-training the target")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model = _get_model(device, args.epochs, args.dataset)

    # Separate train/test splits so the detector is evaluated on UNSEEN inputs.
    train_loader, test_loader = get_loaders(dataset=args.dataset)

    print(f"building detector features (FGSM eps={args.epsilon}, bits={args.bits}, "
          f"median k={args.kernel})...")
    X_tr, y_tr = build_feature_dataset(model, train_loader, epsilon=args.epsilon,
                                       bits=args.bits, kernel=args.kernel, device=device)
    X_te, y_te = build_feature_dataset(model, test_loader, epsilon=args.epsilon,
                                       bits=args.bits, kernel=args.kernel, device=device)
    print(f"  train rows: {len(y_tr)} ({int(y_tr.sum())} adv)  "
          f"test rows: {len(y_te)} ({int(y_te.sum())} adv)")

    bundle = train_detector(X_tr, y_tr, target_fpr=args.target_fpr)

    scores = bundle.score(X_te)
    preds = (scores >= bundle.threshold).astype(int)
    roc_auc = float(roc_auc_score(y_te, scores))
    ap_score = float(average_precision_score(y_te, scores))
    prec = float(precision_score(y_te, preds, zero_division=0))
    rec = float(recall_score(y_te, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_te, preds, labels=[0, 1]).ravel()
    op_fpr = float(fp / max(fp + tn, 1))
    op_tpr = float(tp / max(tp + fn, 1))

    print(f"  ROC-AUC          : {roc_auc:.3f}")
    print(f"  Average precision: {ap_score:.3f}")
    print(f"  operating point  : threshold={bundle.threshold:.3f} "
          f"-> precision={prec:.3f} recall={rec:.3f} FPR={op_fpr:.3f}")

    roc_png = _plot_roc(y_te, scores, op_fpr, op_tpr)
    pr_png = _plot_pr(y_te, scores)
    feat_png = _plot_features(X_te, y_te)

    coeffs = dict(zip(FEATURE_NAMES, [round(float(c), 4) for c in bundle.clf.coef_[0]]))

    metrics = {
        "project": "p5-adv-input-detector",
        "summary": (
            f"Feature-squeezing + statistical detector catches {op_tpr * 100:.0f}% of "
            f"successful FGSM (eps={args.epsilon}) attacks at a {op_fpr * 100:.0f}% clean "
            f"false-positive rate (ROC-AUC {roc_auc:.3f})."
        ),
        "dataset": args.dataset,
        "seed": 42,
        "attack": f"FGSM (L-inf, single-step, eps={args.epsilon})",
        "detector": "LogisticRegression on 7 squeeze+statistical features",
        "squeezers": {"bit_depth_bits": args.bits, "median_kernel": args.kernel},
        "n_train_rows": int(len(y_tr)),
        "n_test_rows": int(len(y_te)),
        "n_test_adv": int(y_te.sum()),
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(ap_score, 4),
        "operating_point": {
            "target_fpr": args.target_fpr,
            "threshold": round(bundle.threshold, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "false_positive_rate": round(op_fpr, 4),
            "true_positive_rate": round(op_tpr, 4),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "detector_coefficients": coeffs,
        "figures": [
            str(roc_png.relative_to(PROJECT)),
            str(pr_png.relative_to(PROJECT)),
            str(feat_png.relative_to(PROJECT)),
        ],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {roc_png.relative_to(PROJECT)}")
    print(f"wrote {pr_png.relative_to(PROJECT)}")
    print(f"wrote {feat_png.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
