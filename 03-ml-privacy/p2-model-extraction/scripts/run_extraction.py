#!/usr/bin/env python3
"""Run the model-stealing experiment end to end and write figures + metrics.json.

1. Train (or load) the victim and expose it as a black-box label-only API.
2. Sweep query budgets with NO defense -> fidelity-vs-budget curve.
3. Repeat with a rate-limit / query-budget defense -> show the thief capped.
4. Write results/figures/*.png and results/metrics.json.

Run via `make attack`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_extraction import (  # noqa: E402
    StealResult,
    evaluate,
    fidelity_vs_budget,
    get_device,
    get_splits,
    loader,
    make_victim,
    set_seed,
)
from model_extraction.train import save_model, train  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"
WEIGHTS = PROJECT / "models" / "victim.pt"

DEFAULT_BUDGETS = [250, 500, 1000, 2000, 4000, 8000]


def _get_victim(splits, device, epochs):
    from model_extraction.train import load_victim

    if WEIGHTS.exists():
        try:
            print(f"loading victim <- {WEIGHTS.relative_to(PROJECT)}")
            return load_victim(WEIGHTS, device)
        except Exception:  # noqa: BLE001 - architecture/shape mismatch -> retrain
            print("victim weights incompatible with current dataset; retraining...")
    print("training a fresh victim...")
    victim = make_victim(splits.img_size, splits.n_classes)
    train(victim, loader(splits.victim_x, splits.victim_y, shuffle=True), epochs=epochs,
          device=device)
    save_model(victim, WEIGHTS, splits.img_size, splits.n_classes)
    return victim.eval()


def _plot_curve(no_def: list[StealResult], defended: list[StealResult], cap: int,
                victim_acc: float) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bx = [r.budget_requested for r in no_def]
    fid = [r.thief_fidelity * 100 for r in no_def]
    fid_def = [r.thief_fidelity * 100 for r in defended]

    ax.plot(bx, fid, "o-", color="#c0392b", linewidth=2, label="no defense")
    ax.plot(bx, fid_def, "s--", color="#2471a3", linewidth=2,
            label=f"rate-limit (cap={cap} queries)")
    ax.axvline(cap, color="#2471a3", alpha=0.3, linestyle=":")
    ax.set_xlabel("attacker query budget (images labelled by the victim API)")
    ax.set_ylabel("thief fidelity vs victim (% test-set agreement)")
    ax.set_title("Model extraction: fidelity rises with query budget;\nrate-limiting caps the thief",
                 fontsize=11, pad=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "fidelity_vs_budget.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_acc_fid(no_def: list[StealResult], victim_acc: float) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bx = [r.budget_requested for r in no_def]
    ax.plot(bx, [r.thief_test_acc * 100 for r in no_def], "o-", color="#27ae60",
            linewidth=2, label="thief task accuracy")
    ax.plot(bx, [r.thief_fidelity * 100 for r in no_def], "s-", color="#c0392b",
            linewidth=2, label="thief fidelity (agreement w/ victim)")
    ax.axhline(victim_acc * 100, color="#7f8c8d", linestyle="--",
               label=f"victim accuracy ({victim_acc * 100:.0f}%)")
    ax.set_xlabel("attacker query budget")
    ax.set_ylabel("percent")
    ax.set_title("Stolen model: accuracy and fidelity vs query budget", fontsize=11, pad=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "accuracy_and_fidelity.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "mnist"], default="synthetic")
    ap.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    ap.add_argument("--defense-cap", type=int, default=1000,
                    help="max queries the rate-limited API will serve")
    ap.add_argument("--epochs", type=int, default=8, help="thief training epochs")
    ap.add_argument("--victim-epochs", type=int, default=12)
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    splits = get_splits(args.dataset)
    print(f"dataset={args.dataset}  victim={len(splits.victim_x)}  "
          f"attack-pool={splits.n_attack}  test={len(splits.test_x)}")

    victim = _get_victim(splits, device, args.victim_epochs)
    victim_acc = evaluate(victim, loader(splits.test_x, splits.test_y), device=device)
    print(f"victim test accuracy: {victim_acc * 100:.1f}%\n")

    print("=== extraction with NO defense ===")
    no_def = fidelity_vs_budget(victim, splits, args.budgets, epochs=args.epochs, device=device)
    for r in no_def:
        print(f"  budget={r.budget_requested:>5}  used={r.queries_used:>5}  "
              f"thief_acc={r.thief_test_acc * 100:5.1f}%  fidelity={r.thief_fidelity * 100:5.1f}%")

    print(f"\n=== extraction WITH rate-limit defense (cap={args.defense_cap}) ===")
    defended = fidelity_vs_budget(
        victim, splits, args.budgets, api_max_queries=args.defense_cap,
        epochs=args.epochs, device=device,
    )
    for r in defended:
        flag = " (THROTTLED)" if r.rejected else ""
        print(f"  budget={r.budget_requested:>5}  used={r.queries_used:>5}  "
              f"fidelity={r.thief_fidelity * 100:5.1f}%{flag}")

    curve = _plot_curve(no_def, defended, args.defense_cap, victim_acc)
    accfid = _plot_acc_fid(no_def, victim_acc)

    best = no_def[-1]
    def_best = defended[-1]
    metrics = {
        "project": "p2-model-extraction",
        "summary": (
            f"A label-only thief trained on {best.queries_used} victim queries reaches "
            f"{best.thief_fidelity * 100:.0f}% fidelity to the victim "
            f"(victim acc {victim_acc * 100:.0f}%). A query-budget rate limit of "
            f"{args.defense_cap} caps the thief at {def_best.thief_fidelity * 100:.0f}% fidelity."
        ),
        "dataset": args.dataset,
        "seed": 42,
        "attack": "label-only model extraction (hard-label query, no attack library)",
        "victim_test_accuracy": victim_acc,
        "defense_cap_queries": args.defense_cap,
        "budgets": args.budgets,
        "no_defense": [
            {
                "budget": r.budget_requested,
                "queries_used": r.queries_used,
                "thief_test_accuracy": r.thief_test_acc,
                "thief_fidelity": r.thief_fidelity,
            }
            for r in no_def
        ],
        "with_defense": [
            {
                "budget": r.budget_requested,
                "queries_used": r.queries_used,
                "throttled": r.rejected,
                "thief_fidelity": r.thief_fidelity,
            }
            for r in defended
        ],
        "best_fidelity_no_defense": best.thief_fidelity,
        "best_fidelity_with_defense": def_best.thief_fidelity,
        "figures": [str(curve.relative_to(PROJECT)), str(accfid.relative_to(PROJECT))],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {curve.relative_to(PROJECT)}")
    print(f"wrote {accfid.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
