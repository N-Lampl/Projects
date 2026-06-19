#!/usr/bin/env python3
"""Run the online LiRA membership-inference attack end to end and write
results/figures/*.png + results/metrics.json. Run via `make attack`.

Pipeline:
  1. build a population pool (default: synthetic; --dataset fashion_mnist for the
     optional torchvision path),
  2. warm-start checkpoint -> target model -> 8-16 warm-started shadows,
  3. per-example LiRA likelihood-ratio scores + a naive-confidence baseline,
  4. ROC, AUC, TPR@1%FPR, and the score-distribution plot.
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

from lira_mia import (  # noqa: E402
    auc,
    build_target,
    collect_shadow_signals,
    global_threshold_baseline,
    lira_scores,
    make_synthetic_pool,
    make_warm_start,
    roc_from_scores,
    set_seed,
    target_confidences,
    tpr_at_fpr,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _load_pool(name: str, n_samples: int):
    if name == "synthetic":
        return make_synthetic_pool(n_samples=n_samples)
    if name == "fashion_mnist":
        from lira_mia.data import load_fashion_mnist_pool

        return load_fashion_mnist_pool(root=str(PROJECT / "data"), n_samples=n_samples)
    raise ValueError(f"unknown dataset {name!r}")


def _plot_roc(curves: dict[str, tuple], aucs: dict[str, float], tprs: dict[str, float]) -> Path:
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = {"LiRA (online)": "#c0392b", "confidence baseline": "#2980b9"}
    for name, (fpr, tpr) in curves.items():
        c = colors.get(name, "#555")
        label = f"{name}  AUC={aucs[name]:.3f}  TPR@1%={tprs[name]:.3f}"
        ax_lin.plot(fpr, tpr, color=c, lw=2, label=label)
        ax_log.plot(np.clip(fpr, 1e-4, 1), np.clip(tpr, 1e-4, 1), color=c, lw=2, label=name)
    ax_lin.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_lin.set(xlabel="false positive rate", ylabel="true positive rate",
               title="LiRA ROC (linear)")
    ax_lin.legend(fontsize=8, loc="lower right")
    ax_lin.grid(alpha=0.3)

    ax_log.plot([1e-4, 1], [1e-4, 1], "k--", lw=1, alpha=0.5)
    ax_log.axvline(0.01, color="gray", ls=":", lw=1)
    ax_log.set(xscale="log", yscale="log", xlim=(1e-3, 1), ylim=(1e-3, 1),
               xlabel="false positive rate (log)", ylabel="true positive rate (log)",
               title="ROC (log-log) -- the low-FPR regime that matters")
    ax_log.legend(fontsize=8, loc="lower right")
    ax_log.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = FIG_DIR / "lira_roc.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_score_hist(scores: np.ndarray, is_member: np.ndarray) -> Path:
    # clip to robust percentiles for display: a few tail examples produce extreme
    # log-ratios that would otherwise swamp the histogram (the ROC uses raw scores).
    lo, hi = np.percentile(scores, [1, 99])
    s = np.clip(scores, lo, hi)
    bins = np.linspace(lo, hi, 41)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(s[is_member], bins=bins, alpha=0.6, color="#c0392b", label="members")
    ax.hist(s[~is_member], bins=bins, alpha=0.6, color="#2980b9", label="non-members")
    ax.set(xlabel="LiRA log-likelihood-ratio score (clipped to 1-99 pct)", ylabel="count",
           title="Members score higher: separation is the leakage")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "lira_score_hist.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "fashion_mnist"], default="synthetic")
    ap.add_argument("--pool-size", type=int, default=3000)
    ap.add_argument("--n-shadows", type=int, default=16, help="keep small for CPU (8-16)")
    ap.add_argument("--n-query", type=int, default=500)
    ap.add_argument("--target-epochs", type=int, default=100)
    ap.add_argument("--shadow-epochs", type=int, default=100)
    ap.add_argument("--no-warm-start", action="store_true")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] building pool ({args.dataset}, n={args.pool_size})...")
    pool = _load_pool(args.dataset, args.pool_size)

    warm = None if args.no_warm_start else make_warm_start(pool)
    print(f"[2/4] training target (warm_start={'off' if warm is None else 'on'})...")
    tw = build_target(pool, n_query=args.n_query, target_epochs=args.target_epochs,
                      warm_start=warm)
    print(f"      target train_acc={tw.train_acc:.3f}  test_acc={tw.test_acc:.3f}  "
          f"(gap={tw.train_acc - tw.test_acc:.3f} -> the memorisation MIA exploits)")

    print(f"[3/4] training {args.n_shadows} warm-started shadows...")
    sig = collect_shadow_signals(pool, tw.query_idx, n_shadows=args.n_shadows,
                                 shadow_epochs=args.shadow_epochs, warm_start=warm,
                                 progress=True)

    print("[4/4] scoring + metrics...")
    phi_target = target_confidences(tw.model, pool, tw.query_idx)
    lira = lira_scores(phi_target, sig)
    base = global_threshold_baseline(phi_target)

    curves, aucs, tprs = {}, {}, {}
    for name, sc in {"LiRA (online)": lira, "confidence baseline": base}.items():
        fpr, tpr, _ = roc_from_scores(sc, tw.is_member)
        curves[name] = (fpr, tpr)
        aucs[name] = auc(fpr, tpr)
        tprs[name] = tpr_at_fpr(fpr, tpr, 0.01)
        print(f"      {name:22s} AUC={aucs[name]:.3f}  TPR@1%FPR={tprs[name]:.3f}")

    roc_png = _plot_roc(curves, aucs, tprs)
    hist_png = _plot_score_hist(lira, tw.is_member)

    metrics = {
        "project": "p3-membership-inference",
        "summary": (
            f"Online LiRA with {args.n_shadows} warm-started shadows on a self-trained "
            f"{args.dataset} MLP: AUC={aucs['LiRA (online)']:.3f}, "
            f"TPR@1%FPR={tprs['LiRA (online)']:.3f} "
            f"(baseline AUC={aucs['confidence baseline']:.3f})."
        ),
        "dataset": args.dataset,
        "seed": 42,
        "n_shadows": args.n_shadows,
        "n_query": args.n_query,
        "warm_start": warm is not None,
        "target_train_acc": round(tw.train_acc, 4),
        "target_test_acc": round(tw.test_acc, 4),
        "generalization_gap": round(tw.train_acc - tw.test_acc, 4),
        "lira_auc": round(aucs["LiRA (online)"], 4),
        "lira_tpr_at_1pct_fpr": round(tprs["LiRA (online)"], 4),
        "baseline_auc": round(aucs["confidence baseline"], 4),
        "baseline_tpr_at_1pct_fpr": round(tprs["confidence baseline"], 4),
        "figures": [
            str(roc_png.relative_to(PROJECT)),
            str(hist_png.relative_to(PROJECT)),
        ],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {roc_png.relative_to(PROJECT)}")
    print(f"wrote {hist_png.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
