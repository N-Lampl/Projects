#!/usr/bin/env python3
"""Train the target at several epsilons, re-run MIA + model extraction against each,
and write the tradeoff figures + metrics.json. Run via `make run`.

DEFAULT (offline): manual DP-SGD + synthetic data + a shared non-private shadow
set -- no Opacus, no downloads. The smoke default is 2 epsilon points + a tiny
pool so it finishes in well under a minute on a CPU. Pass --full for the 3-point
{inf, 3, 1} flagship sweep (still CPU-friendly, a few minutes).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dp_defenses import (  # noqa: E402
    build_shared_world,
    evaluate_epsilon,
    make_synthetic_pool,
    set_seed,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _eps_label(e: float) -> str:
    return "inf" if math.isinf(e) else f"{e:g}"


def _plot_tradeoff(results: list) -> Path:
    """The money plot: utility AND attack success vs the privacy budget epsilon.

    x-axis is epsilon on a log scale with inf pinned at the right; lower epsilon =
    stronger privacy. We expect test accuracy to fall and both attack metrics to
    fall as epsilon shrinks -- the classic privacy-utility tradeoff.
    """
    # order weakest-privacy (inf) ... strongest (smallest eps); plot left=strong
    res = sorted(results, key=lambda r: (math.inf if math.isinf(r.epsilon) else r.epsilon))
    labels = [_eps_label(r.epsilon) for r in res]
    xs = list(range(len(res)))

    test_acc = [r.test_acc * 100 for r in res]
    gap = [r.gen_gap * 100 for r in res]
    mia = [r.mia_auc * 100 for r in res]
    steal = [r.steal_fidelity * 100 for r in res]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(xs, test_acc, "o-", color="#27ae60", linewidth=2, label="target test acc")
    ax.plot(xs, gap, "s--", color="#7f8c8d", linewidth=1.6, label="train-test gap")
    ax.plot(xs, mia, "^-", color="#c0392b", linewidth=2, label="MIA AUC")
    ax.plot(xs, steal, "D-", color="#2980b9", linewidth=2, label="extraction fidelity")
    ax.axhline(50, color="#c0392b", alpha=0.25, linestyle=":")
    ax.annotate("MIA chance (50%)", (0, 50), fontsize=8, color="#c0392b", va="bottom")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"eps={lbl}" for lbl in labels])
    ax.set_xlabel("privacy budget (left = stronger privacy, more DP noise)")
    ax.set_ylabel("percent")
    ax.set_title("DP-SGD privacy-utility tradeoff: utility vs attack success", pad=12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "privacy_utility_tradeoff.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_roc(results: list) -> Path:
    """MIA ROC curves, one per epsilon -- DP should pull the curve toward chance."""
    res = sorted(results, key=lambda r: (math.inf if math.isinf(r.epsilon) else r.epsilon))
    fig, ax = plt.subplots(figsize=(5.2, 5))
    colors = ["#c0392b", "#e67e22", "#2980b9", "#8e44ad", "#16a085"]
    for i, r in enumerate(res):
        fpr, tpr = r.roc
        ax.plot(fpr, tpr, linewidth=2, color=colors[i % len(colors)],
                label=f"eps={_eps_label(r.epsilon)}  (AUC={r.mia_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("LiRA membership-inference ROC vs DP budget", pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "mia_roc_by_epsilon.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--epsilons", type=str, nargs="+", default=None,
        help="epsilons to sweep, e.g. inf 3 1  (default: smoke = inf 1)",
    )
    ap.add_argument("--full", action="store_true",
                    help="flagship sweep {inf, 3, 1} + larger pool")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="target training epochs")
    ap.add_argument("--n-shadows", type=int, default=None)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    args = ap.parse_args()

    # presets: smoke (fast, 2 points) vs full (flagship, 3 points)
    if args.full:
        epsilons = args.epsilons or ["inf", "3", "1"]
        n_samples = args.n_samples or 2400
        epochs = args.epochs or 60
        n_shadows = args.n_shadows or 12
    else:
        epsilons = args.epsilons or ["inf", "1"]
        n_samples = args.n_samples or 1200
        epochs = args.epochs or 60
        n_shadows = args.n_shadows or 8

    eps_vals = [math.inf if e.lower() in ("inf", "infinity") else float(e) for e in epsilons]

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"building shared world (n_samples={n_samples}, n_shadows={n_shadows})...")
    pool = make_synthetic_pool(n_samples=n_samples)
    world = build_shared_world(pool, n_shadows=n_shadows, shadow_epochs=max(15, epochs - 5))

    results = []
    for e in eps_vals:
        print(f"\n=== epsilon = {_eps_label(e)} ===")
        r = evaluate_epsilon(
            world, e, epochs=epochs,
            max_grad_norm=args.max_grad_norm, delta=args.delta,
        )
        print(f"  sigma={r.noise_multiplier:.3f}  accounted_eps={r.accounted_epsilon:.3f}")
        print(f"  utility:  train_acc={r.train_acc:.3f}  test_acc={r.test_acc:.3f}  "
              f"gap={r.gen_gap:.3f}")
        print(f"  MIA:      AUC={r.mia_auc:.3f}  TPR@1%FPR={r.mia_tpr_at_1pct:.3f}")
        print(f"  steal:    acc={r.steal_acc:.3f}  fidelity={r.steal_fidelity:.3f}")
        results.append(r)

    tradeoff = _plot_tradeoff(results)
    roc = _plot_roc(results)

    by_eps = {}
    for r in results:
        by_eps[_eps_label(r.epsilon)] = {
            "requested_epsilon": None if math.isinf(r.epsilon) else r.epsilon,
            "accounted_epsilon": None if math.isinf(r.accounted_epsilon)
            else round(r.accounted_epsilon, 4),
            "noise_multiplier": round(r.noise_multiplier, 4),
            "train_acc": round(r.train_acc, 4),
            "test_acc": round(r.test_acc, 4),
            "train_test_gap": round(r.gen_gap, 4),
            "mia_auc": round(r.mia_auc, 4),
            "mia_tpr_at_1pct_fpr": round(r.mia_tpr_at_1pct, 4),
            "extraction_acc": round(r.steal_acc, 4),
            "extraction_fidelity": round(r.steal_fidelity, 4),
        }

    inf_key = "inf"
    summary = (
        "DP-SGD trades test accuracy for privacy: as epsilon shrinks, the train-test "
        "gap, membership-inference AUC, and extraction fidelity all fall toward their "
        "no-information baselines."
    )
    metrics = {
        "project": "CAPSTONE-dp-defenses",
        "summary": summary,
        "track": "03-ml-privacy",
        "default_backend": "manual DP-SGD (torch); Opacus optional",
        "accountant": "subsampled-Gaussian RDP -> (eps, delta)",
        "delta": args.delta,
        "max_grad_norm": args.max_grad_norm,
        "n_shadows": n_shadows,
        "n_pool": n_samples,
        "epsilons": [_eps_label(e) for e in eps_vals],
        "by_epsilon": by_eps,
        "headline": {
            "non_private_mia_auc": by_eps.get(inf_key, {}).get("mia_auc"),
            "non_private_test_acc": by_eps.get(inf_key, {}).get("test_acc"),
            "private_min_eps": min(
                (v["requested_epsilon"] for v in by_eps.values()
                 if v["requested_epsilon"] is not None), default=None),
        },
        "figures": [
            str(tradeoff.relative_to(PROJECT)),
            str(roc.relative_to(PROJECT)),
        ],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {tradeoff.relative_to(PROJECT)}")
    print(f"wrote {roc.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
