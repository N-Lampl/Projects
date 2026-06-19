#!/usr/bin/env python3
"""Run the attribute-inference attack on synthetic tabular data, sweeping how
strongly the sensitive attribute drives the label, and write a figure + metrics.
Run via `make attribute`.
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

from inversion_attribute import (  # noqa: E402
    make_attribute_dataset,
    run_attribute_inference,
    set_seed,
    train_target,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics_attribute.json"


def _plot_sweep(signals: list[float], attack: list[float], base: list[float]) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(signals, [a * 100 for a in attack], "o-", color="#c0392b", lw=2, label="attack")
    ax.plot(signals, [b * 100 for b in base], "s--", color="#7f8c8d", lw=2, label="baseline (guess majority)")
    ax.set_xlabel("strength of sensitive attribute -> label (s_signal)")
    ax.set_ylabel("sensitive-attribute recovery (%)")
    ax.set_title("Attribute inference: leakage grows with how much\nthe model relies on the sensitive feature", fontsize=11)
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "attribute_inference_sweep.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    ap.add_argument("--signals", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.6, 2.4, 3.2])
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    attack_accs, base_accs, rows = [], [], []
    for sig in args.signals:
        data = make_attribute_dataset(n=args.n, s_signal=sig)
        clf, X_te, _ = train_target(data, model=args.model)
        res = run_attribute_inference(data, clf, X_te)
        attack_accs.append(res["attack_accuracy"])
        base_accs.append(res["baseline_accuracy"])
        rows.append({"s_signal": sig, **res})
        print(f"  s_signal={sig:<4} attack={res['attack_accuracy'] * 100:5.1f}%  "
              f"baseline={res['baseline_accuracy'] * 100:5.1f}%  "
              f"lift={res['lift_over_baseline'] * 100:+5.1f}pp")

    fig = _plot_sweep(args.signals, attack_accs, base_accs)

    # headline at the strongest tested signal
    head = rows[-1]
    metrics = {
        "project": "p4-inversion-attribute",
        "task": "attribute-inference",
        "summary": (
            f"On synthetic tabular data, MAP attribute inference recovers the sensitive "
            f"attribute with {head['attack_accuracy'] * 100:.0f}% accuracy vs a "
            f"{head['baseline_accuracy'] * 100:.0f}% majority baseline "
            f"(+{head['lift_over_baseline'] * 100:.0f}pp) when the model relies on it."
        ),
        "seed": 42,
        "target_model": args.model,
        "sweep": rows,
        "max_attack_accuracy": max(attack_accs),
        "max_lift_over_baseline": max(r["lift_over_baseline"] for r in rows),
        "figures": [str(fig.relative_to(PROJECT))],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {fig.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
