"""The one-shot money target: train the detector, replay the p5 attacks against
the undefended p4 RAG and the DefendedRAG, and write the before/after ASR plot.

Fully offline by default (synthetic data + mock LLM). Produces:
  results/figures/asr_before_after.png   -- the MONEY PLOT (ASR delta by family)
  results/figures/detector_roc.png       -- detector ROC curve
  results/metrics.json                   -- all numbers (dashboard-discoverable)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defend_rag import (  # noqa: E402
    DefendedRAG,
    InjectionDetector,
    asr,
    asr_by_family,
    build_attacks,
    build_undefended_target,
    generate_dataset,
    run_battery,
    set_seed,
    train_detector,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
METRICS = ROOT / "results" / "metrics.json"
MODEL_PATH = ROOT / "models" / "injection_detector.joblib"


def _get_detector(n_per_class: int) -> tuple[InjectionDetector, dict]:
    """Load a saved detector if present, else train one. Return (det, eval_dict)."""
    ds = generate_dataset(n_per_class=n_per_class, seed=42)
    n_test = int(len(ds) * 0.25)
    test_x, test_y = ds.texts[:n_test], ds.labels[:n_test]
    if MODEL_PATH.exists():
        det = InjectionDetector.load(MODEL_PATH)
    else:
        det = train_detector(ds.texts[n_test:], ds.labels[n_test:])
        det.save(MODEL_PATH)
    report = det.evaluate(test_x, test_y)
    return det, {"report": report, "n_train": len(ds) - n_test, "n_test": n_test}


def _plot_money(before_fam: dict, after_fam: dict, before_all: float, after_all: float) -> Path:
    families = sorted(set(before_fam) | set(after_fam))
    labels = [*families, "OVERALL"]
    before = [before_fam.get(f, 0.0) for f in families] + [before_all]
    after = [after_fam.get(f, 0.0) for f in families] + [after_all]

    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], before, width, label="undefended (p4)", color="#c0392b")
    ax.bar([i + width / 2 for i in x], after, width, label="defended (p7)", color="#27ae60")
    ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Prompt-injection ASR: before vs after layered guardrails")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - width / 2, b + 0.02, f"{b:.0%}", ha="center", fontsize=8)
        ax.text(i + width / 2, a + 0.02, f"{a:.0%}", ha="center", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "asr_before_after.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_roc(report) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(report.fpr, report.tpr, color="#2980b9", lw=2, label=f"AUC = {report.roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Injection detector ROC (TF-IDF + LogReg)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "detector_roc.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Defend the RAG; produce ASR delta.")
    parser.add_argument("--n-per-class", type=int, default=600)
    parser.add_argument("--use-nemo", action="store_true", help="enable optional NeMo output rail")
    args = parser.parse_args()

    set_seed(42)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    det, det_info = _get_detector(args.n_per_class)
    report = det_info["report"]

    attacks = build_attacks()

    # BEFORE: undefended p4 RAG.
    undef = build_undefended_target(k=3)
    before = run_battery(undef, attacks, defended=False)

    # AFTER: same RAG wrapped in the four-layer defense.
    defended_rag = DefendedRAG(build_undefended_target(k=3), det, use_nemo=args.use_nemo)
    after = run_battery(defended_rag, attacks, defended=True)

    before_all, after_all = asr(before), asr(after)
    before_fam, after_fam = asr_by_family(before), asr_by_family(after)

    money = _plot_money(before_fam, after_fam, before_all, after_all)
    roc = _plot_roc(report)

    metrics = {
        "project": "p7-defend-rag",
        "summary": (
            f"Layered guardrails cut prompt-injection ASR from {before_all:.0%} to "
            f"{after_all:.0%} across {len(attacks)} attacks; the from-scratch "
            f"TF-IDF+LogReg detector scores ROC-AUC {report.roc_auc:.3f}."
        ),
        "asr_before": round(before_all, 4),
        "asr_after": round(after_all, 4),
        "asr_reduction": round(before_all - after_all, 4),
        "asr_before_by_family": {k: round(v, 4) for k, v in before_fam.items()},
        "asr_after_by_family": {k: round(v, 4) for k, v in after_fam.items()},
        "n_attacks": len(attacks),
        "detector": {
            **report.as_dict(),
            "n_train": det_info["n_train"],
            "n_test": det_info["n_test"],
            "model": "tfidf(word 1-2gram) + logreg",
        },
        "attacks": [
            {
                "id": o.attack_id,
                "family": o.family,
                "leaked_before": b.succeeded,
                "leaked_after": o.succeeded,
            }
            for b, o in zip(before, after)
        ],
        "figures": [
            f"results/figures/{money.name}",
            f"results/figures/{roc.name}",
        ],
    }
    METRICS.write_text(json.dumps(metrics, indent=2))

    print(metrics["summary"])
    print(f"  ASR  before={before_all:.0%}  after={after_all:.0%}")
    print(f"wrote {money}")
    print(f"wrote {roc}")
    print(f"wrote {METRICS}")


if __name__ == "__main__":
    main()
