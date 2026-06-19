#!/usr/bin/env python3
"""Run the full adversarial-IDS capstone end-to-end and write the report card.

Pipeline:
  1. train the shared RandomForest IDS on synthetic flows (offline);
  2. fit a differentiable logistic substitute to the target's decisions;
  3. craft constrained-FGSM evasions of the TEST attack flows (mutable features
     only, attacker-feasible directions, projected to a valid flow);
  4. measure transfer attack-success-rate on the deployed target;
  5. HARDEN (adversarial training, default) and re-measure;
  6. emit results/report_card.md, results/figures/*.png, results/metrics.json.

Fully offline on synthetic data with scikit-learn/numpy/pandas/matplotlib.
``--use-art`` switches the attack engine to IBM ART if installed (else it
transparently falls back to the hand-rolled FGSM). Run via ``make attack``.
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

from adversarial_ids import (  # noqa: E402
    adversarially_train,
    attack_success_rate,
    build_constraints,
    build_robust_ensemble,
    craft_adversarial,
    fit_surrogate,
    get_pipeline_api,
    refit_surrogate_for,
    set_seed,
    write_report_card,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"
REPORT = PROJECT / "results" / "report_card.md"


def _clean_metrics(api, pipe, ds) -> dict:
    m = api.evaluate(pipe, ds)
    return {"accuracy": _accuracy(api, pipe, ds), "roc_auc": m["roc_auc"], "recall": m["recall"]}


def _accuracy(api, pipe, ds) -> float:
    from sklearn.metrics import accuracy_score

    y_pred = pipe.predict(ds.X_test)
    return float(accuracy_score(ds.y_test, y_pred))


def _plot_report(metrics: dict) -> Path:
    """Two-panel money figure: ASR before/after + clean metrics retained."""
    s = metrics["summary"]
    clean = metrics["clean"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel 1: attack success rate before vs after
    ax = axes[0]
    bars = ax.bar(
        ["before\nhardening", "after\nhardening"],
        [s["asr_before"] * 100, s["asr_after"] * 100],
        color=["#c0392b", "#27ae60"],
        width=0.55,
    )
    ax.set_ylabel("attack success rate (%)")
    ax.set_title("Constrained evasion: ASR collapses after hardening", pad=10)
    ax.set_ylim(0, max(100, s["asr_before"] * 100 + 12))
    for b, v in zip(bars, [s["asr_before"], s["asr_after"]]):
        ax.annotate(
            f"{v:.0%}", (b.get_x() + b.get_width() / 2, b.get_height()),
            textcoords="offset points", xytext=(0, 6), ha="center", fontsize=11, fontweight="bold",
        )
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: clean metrics retained (no accuracy collapse from the defense)
    ax = axes[1]
    labels = ["accuracy", "ROC-AUC", "recall"]
    before = [clean["accuracy_before"], clean["roc_auc_before"], clean["recall_before"]]
    after = [clean["accuracy_after"], clean["roc_auc_after"], clean["recall_after"]]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, before, w, label="before", color="#7f8c8d")
    ax.bar(x + w / 2, after, w, label="after", color="#2980b9")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Clean performance is retained by the defense", pad=10)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("IDS Robustness Report Card", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "report_card.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_perturbation(cspec, X_clean, X_adv, numeric_features) -> Path:
    """Show WHICH features the attack moved (mutable vs immutable)."""
    mean_abs = np.abs(X_adv - X_clean).mean(axis=0)
    # normalise per-feature by std so scales are comparable
    std = X_clean.std(axis=0) + 1e-9
    norm_move = mean_abs / std

    colors = ["#c0392b" if cspec.mutable_mask[i] > 0 else "#bdc3c7"
              for i in range(len(numeric_features))]
    order = np.argsort(norm_move)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        [numeric_features[i] for i in order][::-1],
        [norm_move[i] for i in order][::-1],
        color=[colors[i] for i in order][::-1],
    )
    ax.set_xlabel("mean |perturbation| (in feature-std units)")
    ax.set_title("Attack only moves MUTABLE features (red); immutable held fixed (grey)", pad=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "perturbation_by_feature.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilon", type=float, default=2.0,
                    help="L-inf budget as a fraction of each feature's std")
    ap.add_argument("--steps", type=int, default=10, help="constrained-FGSM steps (BIM/PGD)")
    ap.add_argument("--n-samples", type=int, default=12000, help="synthetic flows to generate")
    ap.add_argument("--defense", choices=["advtrain", "ensemble"], default="advtrain")
    ap.add_argument("--use-art", action="store_true",
                    help="use IBM ART FastGradientMethod if installed (else fall back)")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    api = get_pipeline_api()

    # 1. data + baseline IDS ------------------------------------------------
    print("loading synthetic flows + training the baseline IDS...")
    ds = api.load_data(synthetic=True, n_samples=args.n_samples)
    base = api.build_pipeline(ds)
    api.train(base, ds)
    num = ds.numeric_features
    cspec = build_constraints(num)
    print(f"  mutable features: {[c for c in num if cspec.mutable_mask[num.index(c)] > 0]}")

    clean_base = _clean_metrics(api, base, ds)
    print(f"  baseline clean: acc={clean_base['accuracy']:.3f} "
          f"auc={clean_base['roc_auc']:.3f} recall={clean_base['recall']:.3f}")

    # 2. substitute + 3. craft attack on TEST attack flows ------------------
    print("fitting differentiable substitute + crafting constrained evasions...")
    X_test_num = ds.X_test[num].to_numpy(dtype=np.float64)
    y_test = ds.y_test.to_numpy()
    sur = fit_surrogate(base, ds.X_train[num].to_numpy(dtype=np.float64), num)

    atk_mask = y_test == 1
    X_atk = X_test_num[atk_mask]
    y_atk = y_test[atk_mask]
    X_adv, engine = craft_adversarial(
        sur, cspec, X_atk, y_atk,
        epsilon=args.epsilon, steps=args.steps, use_art=args.use_art,
    )
    print(f"  attack engine: {engine}")

    # 4. ASR on the deployed target -----------------------------------------
    asr_before = attack_success_rate(base, cspec, num, X_atk, X_adv, y_atk)
    print(f"  ASR before hardening: {asr_before['attack_success_rate']:.1%} "
          f"({asr_before['n_evaded']}/{asr_before['n_detected_before']} detected attacks evaded)")

    # 5. harden + re-measure -------------------------------------------------
    if args.defense == "advtrain":
        print("hardening via ADVERSARIAL TRAINING + re-measuring...")
        hardened, _ = adversarially_train(
            api, ds, cspec, sur, epsilon=args.epsilon, steps=args.steps
        )
        defense_label = "adversarial training (constrained-FGSM augmentation)"
    else:
        print("hardening via DIVERSE ENSEMBLE + re-measuring...")
        hardened = build_robust_ensemble(api, ds)
        defense_label = "diverse RandomForest ensemble (deeper, sqrt feature subsample)"

    clean_hard = _clean_metrics(api, hardened, ds)
    # re-fit a substitute against the hardened target, re-craft, re-measure
    sur2 = refit_surrogate_for(api, hardened, ds)
    X_adv2, _ = craft_adversarial(
        sur2, cspec, X_atk, y_atk,
        epsilon=args.epsilon, steps=args.steps, use_art=args.use_art,
    )
    asr_after = attack_success_rate(hardened, cspec, num, X_atk, X_adv2, y_atk)
    print(f"  ASR after hardening: {asr_after['attack_success_rate']:.1%} "
          f"({asr_after['n_evaded']}/{asr_after['n_detected_before']} detected attacks evaded)")

    # 6. assemble metrics + figures + report card ---------------------------
    metrics = {
        "project": "CAPSTONE-adversarial-ids",
        "source": ds.source,
        "attack_engine": engine,
        "epsilon": args.epsilon,
        "steps": args.steps,
        "defense": defense_label,
        "seed": 42,
        "summary": {
            "clean_accuracy": clean_base["accuracy"],
            "asr_before": asr_before["attack_success_rate"],
            "asr_after": asr_after["attack_success_rate"],
            "asr_reduction": asr_before["attack_success_rate"] - asr_after["attack_success_rate"],
        },
        "clean": {
            "accuracy_before": clean_base["accuracy"],
            "accuracy_after": clean_hard["accuracy"],
            "roc_auc_before": clean_base["roc_auc"],
            "roc_auc_after": clean_hard["roc_auc"],
            "recall_before": clean_base["recall"],
            "recall_after": clean_hard["recall"],
        },
        "attack_before": asr_before,
        "attack_after": asr_after,
    }

    fig_card = _plot_report(metrics)
    fig_pert = _plot_perturbation(cspec, X_atk, X_adv, num)
    metrics["figures"] = [
        str(fig_card.relative_to(PROJECT)),
        str(fig_pert.relative_to(PROJECT)),
        str(REPORT.relative_to(PROJECT)),
    ]

    write_report_card(metrics, REPORT)
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\nwrote {REPORT.relative_to(PROJECT)}")
    print(f"wrote {fig_card.relative_to(PROJECT)}")
    print(f"wrote {fig_pert.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
