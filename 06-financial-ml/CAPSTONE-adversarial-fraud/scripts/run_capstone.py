"""End-to-end capstone: train fraud model -> attack -> harden -> re-measure.

Produces (all paths relative to the project root):
    results/figures/robustness_before_after.png   the money plot
    results/figures/score_shift.png               attack pushes scores below thr
    results/metrics.json                           dashboard-discoverable metrics
    results/REPORT_CARD.md                         one-page robustness report card

Run:  python scripts/run_capstone.py [--n 12000] [--model logreg] [--steps 40]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adv_fraud import (  # noqa: E402
    AttackConfig,
    adversarially_train,
    attack_success_rate,
    detection_report,
    make_dataset,
    make_model,
    set_seed,
)


def _grade(asr_after: float) -> str:
    if asr_after < 0.10:
        return "A"
    if asr_after < 0.25:
        return "B"
    if asr_after < 0.50:
        return "C"
    return "D"


def main() -> None:
    ap = argparse.ArgumentParser(description="Adversarial fraud capstone")
    ap.add_argument("--n", type=int, default=12000, help="dataset size")
    ap.add_argument("--model", choices=["logreg", "gboost"], default="logreg")
    ap.add_argument("--steps", type=int, default=40, help="evasion search steps")
    ap.add_argument("--rounds", type=int, default=3, help="adv-training rounds")
    ap.add_argument(
        "--linear-defense",
        action="store_true",
        help="keep the hardened model linear (logreg) instead of gboost",
    )
    ap.add_argument("--fpr-budget", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = AttackConfig(steps=args.steps)

    # --- data ---
    ds = make_dataset(n=args.n, seed=args.seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=0.3, stratify=ds.y, random_state=args.seed
    )
    print(f"[data] train={len(X_tr)} test={len(X_te)} fraud_rate={ds.y.mean():.3f}")

    # --- baseline model ---
    base = make_model(kind=args.model, seed=args.seed)
    base.fit(X_tr, y_tr)
    s_base = base.predict_proba(X_te)[:, 1]
    clean = detection_report(y_te, s_base, fpr_budget=args.fpr_budget)
    thr = clean["operating_threshold"]
    print(
        f"[clean] PR-AUC={clean['pr_auc']:.3f} ROC-AUC={clean['roc_auc']:.3f} "
        f"recall@{args.fpr_budget:.0%}FPR={clean['recall_at_fpr_budget']:.3f}"
    )

    # frauds the baseline actually catches in the test set = attack surface
    caught = (y_te == 1) & (s_base >= thr)
    X_attack = X_te[caught]
    print(f"[attack] caught frauds available to attack: {len(X_attack)}")

    atk_before = attack_success_rate(base, X_attack, thr, cfg)
    print(
        f"[attack] ASR_before={atk_before['asr']:.3f} "
        f"feasibility={atk_before['feasibility_rate']:.3f} "
        f"mean_score_drop={atk_before['mean_score_drop']:.3f}"
    )

    # --- harden ---
    # Defender uses a non-linear head by default (it can isolate the bounded
    # fraud region a single hyperplane cannot) + iterative adversarial training.
    harden_kind = "logreg" if args.model == "logreg" and args.linear_defense else "gboost"
    hardened = adversarially_train(
        base, X_tr, y_tr, fpr_budget=args.fpr_budget, kind=harden_kind,
        seed=args.seed, config=cfg, rounds=args.rounds,
    )
    s_hard = hardened.predict_proba(X_te)[:, 1]
    clean_hard = detection_report(y_te, s_hard, fpr_budget=args.fpr_budget)
    thr_hard = clean_hard["operating_threshold"]

    # re-attack the hardened model on the frauds IT catches (apples-to-apples
    # threat model) AND on the original attack set for a direct comparison
    caught_hard = (y_te == 1) & (s_hard >= thr_hard)
    X_attack_hard = X_te[caught_hard]
    atk_after = attack_success_rate(hardened, X_attack_hard, thr_hard, cfg)
    # also: did the SAME crafted evasions still fool the hardened model?
    s_evasions_on_hard = hardened.predict_proba(atk_before["X_adv"])[:, 1]
    transfer_evade = float(np.mean(s_evasions_on_hard < thr_hard))
    print(
        f"[harden] PR-AUC={clean_hard['pr_auc']:.3f} "
        f"ASR_after={atk_after['asr']:.3f} "
        f"old-evasions-still-evade={transfer_evade:.3f}"
    )

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- FIGURE 1: before/after robustness bars ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["ASR\n(attack success)", "Clean PR-AUC", "Recall @ FPR budget"]
    before_vals = [
        atk_before["asr"],
        clean["pr_auc"],
        clean["recall_at_fpr_budget"],
    ]
    after_vals = [
        atk_after["asr"],
        clean_hard["pr_auc"],
        clean_hard["recall_at_fpr_budget"],
    ]
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, before_vals, w, label="Baseline", color="#c0392b")
    ax.bar(x + w / 2, after_vals, w, label="Hardened (adv-trained)", color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("Fraud model: before vs. after adversarial training")
    for xi, (b, a) in enumerate(zip(before_vals, after_vals, strict=True)):
        ax.text(xi - w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "robustness_before_after.png", dpi=130)
    plt.close(fig)

    # --- FIGURE 2: score shift under attack (baseline) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(atk_before["score_before"], bins=30, alpha=0.6, label="before attack",
            color="#2980b9")
    ax.hist(atk_before["score_after"], bins=30, alpha=0.6, label="after attack",
            color="#e67e22")
    ax.axvline(thr, color="k", ls="--", lw=1.5, label=f"threshold={thr:.2f}")
    ax.set_xlabel("model P(fraud) on caught frauds")
    ax.set_ylabel("count")
    ax.set_title("Evasion pushes fraud scores below the alert threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "score_shift.png", dpi=130)
    plt.close(fig)

    figures = [
        "results/figures/robustness_before_after.png",
        "results/figures/score_shift.png",
    ]

    grade = _grade(atk_after["asr"])
    summary = (
        f"Logistic fraud model (PR-AUC={clean['pr_auc']:.3f}). A feasibility-"
        f"constrained greedy evasion (only mutable txn fields, in-bounds, "
        f"consistency-preserving) achieved ASR={atk_before['asr']:.0%} against "
        f"the baseline; adversarial training cut ASR to {atk_after['asr']:.0%} "
        f"(grade {grade}) while keeping clean PR-AUC at {clean_hard['pr_auc']:.3f}."
    )

    metrics = {
        "project": "CAPSTONE-adversarial-fraud",
        "summary": summary,
        "source": "synthetic",
        "classifier": args.model,
        "hardened_classifier": harden_kind,
        "adv_training_rounds": int(args.rounds),
        "seed": args.seed,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "fraud_rate": float(ds.y.mean()),
        "fpr_budget": float(args.fpr_budget),
        "robustness_grade": grade,
        "clean_baseline": {
            "pr_auc": clean["pr_auc"],
            "roc_auc": clean["roc_auc"],
            "recall_at_fpr_budget": clean["recall_at_fpr_budget"],
            "precision_at_threshold": clean["precision_at_threshold"],
            "precision_at_k": clean["precision_at_k"],
            "confusion": clean["confusion"],
            "operating_threshold": clean["operating_threshold"],
        },
        "clean_hardened": {
            "pr_auc": clean_hard["pr_auc"],
            "roc_auc": clean_hard["roc_auc"],
            "recall_at_fpr_budget": clean_hard["recall_at_fpr_budget"],
            "confusion": clean_hard["confusion"],
        },
        "attack": {
            "n_frauds_attacked_baseline": int(len(X_attack)),
            "n_frauds_attacked_hardened": int(len(X_attack_hard)),
            "asr_before": atk_before["asr"],
            "asr_after": atk_after["asr"],
            "feasibility_rate_before": atk_before["feasibility_rate"],
            "feasibility_rate_after": atk_after["feasibility_rate"],
            "mean_score_drop_before": atk_before["mean_score_drop"],
            "old_evasions_still_evade_hardened": transfer_evade,
        },
        "figures": figures,
    }

    results_dir = ROOT / "results"
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- REPORT CARD ---
    report = f"""# Fraud Model Robustness Report Card

**Project:** CAPSTONE-adversarial-fraud  ·  **Baseline:** {args.model}  ·  **Hardened:** {harden_kind} ({args.rounds}-round adv-training)  ·  **Data:** synthetic (seed {args.seed})

## Overall grade: {grade}

{summary}

## Clean performance (no attacker)

| Metric | Baseline | Hardened |
|---|---|---|
| PR-AUC (primary) | {clean['pr_auc']:.3f} | {clean_hard['pr_auc']:.3f} |
| ROC-AUC | {clean['roc_auc']:.3f} | {clean_hard['roc_auc']:.3f} |
| Recall @ {args.fpr_budget:.0%} FPR budget | {clean['recall_at_fpr_budget']:.3f} | {clean_hard['recall_at_fpr_budget']:.3f} |
| Precision @ threshold | {clean['precision_at_threshold']:.3f} | - |
| p@100 | {clean['precision_at_k']['p@100']:.3f} | - |

## Adversarial robustness (feasibility-constrained evasion)

| Metric | Value |
|---|---|
| Frauds attacked (baseline) | {len(X_attack)} |
| **Attack Success Rate - before** | **{atk_before['asr']:.1%}** |
| **Attack Success Rate - after hardening** | **{atk_after['asr']:.1%}** |
| ASR reduction | {atk_before['asr'] - atk_after['asr']:+.1%} |
| Evasion feasibility rate | {atk_before['feasibility_rate']:.1%} |
| Mean P(fraud) drop under attack | {atk_before['mean_score_drop']:.3f} |
| Old evasions still fooling hardened model | {transfer_evade:.1%} |

## Threat model

- **Mutable** (attacker-controlled): amount, hour, merchant_risk, distance_from_home, n_items
- **Immutable** (server-side / account history): account_age_days, avg_amount_30d, txn_count_30d, home_country_risk, card_present
- **Constraints enforced:** per-feature plausibility bounds; integer fields rounded; amount kept >= 5% of the account's 30-day average (consistency). Feasibility rate above audits that every counted evasion obeys the contract.

## Figures

- `results/figures/robustness_before_after.png`
- `results/figures/score_shift.png`

*Synthetic data and self-trained models only. Authorized use only - see ../../ETHICS.md.*
"""
    (results_dir / "REPORT_CARD.md").write_text(report)

    print(f"\n[done] grade={grade}  ASR {atk_before['asr']:.1%} -> {atk_after['asr']:.1%}")
    print(f"[done] wrote {results_dir/'metrics.json'}")
    print(f"[done] wrote {results_dir/'REPORT_CARD.md'}")


if __name__ == "__main__":
    main()
