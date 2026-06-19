#!/usr/bin/env python3
"""Transfer + black-box attack study. Trains two different models (if needed),
crafts PGD adversarials on the CNN surrogate and measures transfer to the MLP
target, then runs hand-rolled query-based Square + Boundary attacks under a
capped query budget. Writes figures + metrics.json. Run via `make attack`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transfer_blackbox import (  # noqa: E402
    QueryOracle,
    boundary_attack,
    build_model,
    evaluate,
    get_device,
    get_loaders,
    set_seed,
    square_attack,
    transfer_accuracy,
)
from transfer_blackbox.train import load_model, save_model, train  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"
MODELS = PROJECT / "models"

DEFAULT_EPS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]


def _get_models(device, train_loader, test_loader, epochs):
    models = {}
    for kind, seed in (("cnn", 42), ("mlp", 7)):
        path = MODELS / f"{kind}.pt"
        if path.exists():
            print(f"loading {kind} <- {path.relative_to(PROJECT)}")
            m = load_model(kind, path, device)
        else:
            print(f"training {kind} (different seed {seed})...")
            set_seed(seed)
            m = build_model(kind)
            train(m, train_loader, epochs=epochs, device=device)
            save_model(m, path)
            m.eval()
        acc = evaluate(m, test_loader, device=device)
        print(f"  {kind.upper()} clean accuracy = {acc * 100:5.1f}%")
        models[kind] = (m, acc)
    return models


def _plot_transfer(res, surrogate_self, out: Path) -> Path:
    eps = sorted(res["target"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eps, [res["surrogate"][e] * 100 for e in eps], "o--", color="#7f8c8d",
            label="surrogate (white-box)")
    ax.plot(eps, [res["target"][e] * 100 for e in eps], "o-", color="#c0392b",
            label="target (transfer / black-box)")
    ax.set_xlabel("epsilon (L-inf budget for PGD on surrogate)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Transfer attack: PGD on CNN surrogate -> MLP target")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_queries(sq_curve, bd_curve, budgets, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(budgets, [s * 100 for s in sq_curve], "o-", color="#2980b9",
            label="Square (score-based, L-inf)")
    ax.plot(budgets, [b * 100 for b in bd_curve], "s-", color="#27ae60",
            label="Boundary (decision-based, L2)")
    ax.set_xlabel("query budget (target calls per image)")
    ax.set_ylabel("attack success rate (%)")
    ax.set_title("Query-based black-box attacks vs. query budget")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _success_rate(oracle_factory, attack_fn, x, y, budget, **kw):
    oracle = oracle_factory()
    res = attack_fn(oracle, x, y, query_budget=budget, **kw)
    return float(res.success.float().mean().item()), res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use real MNIST (downloads, optional)")
    ap.add_argument("--epsilons", type=float, nargs="+", default=DEFAULT_EPS)
    ap.add_argument("--pgd-steps", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-blackbox", type=int, default=40, help="images for query attacks")
    ap.add_argument("--bb-epsilon", type=float, default=0.3, help="L-inf eps for Square")
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    train_loader, test_loader = get_loaders(real=args.real)
    models = _get_models(device, train_loader, test_loader, args.epochs)
    surrogate, surr_acc = models["cnn"]
    target, targ_acc = models["mlp"]

    # ---- 1. Transfer study ------------------------------------------------- #
    print("\n[1] transfer: crafting PGD on CNN surrogate, evaluating on MLP target...")
    res = transfer_accuracy(surrogate, target, test_loader, args.epsilons,
                            steps=args.pgd_steps, device=device)
    for e in sorted(res["target"]):
        print(f"  eps={e:<5} surrogate={res['surrogate'][e] * 100:5.1f}%  "
              f"target(transfer)={res['target'][e] * 100:5.1f}%")
    transfer_fig = _plot_transfer(res, surr_acc, FIG_DIR / "transfer_vs_epsilon.png")

    # ---- 2. Query-based black-box attacks (capped budget) ------------------ #
    print("\n[2] query-based black-box on the MLP target (Square + Boundary)...")
    # take correctly-classified images so "success" = a genuine flip
    xs, ys = [], []
    for x, y in test_loader:
        with torch.no_grad():
            keep = (target(x).argmax(1) == y)
        xs.append(x[keep])
        ys.append(y[keep])
        if sum(t.shape[0] for t in xs) >= args.n_blackbox:
            break
    x_bb = torch.cat(xs)[: args.n_blackbox]
    y_bb = torch.cat(ys)[: args.n_blackbox]

    sq_curve, bd_curve = [], []
    sq_last = bd_last = None
    for b in args.budgets:
        sr_sq, sq_last = _success_rate(lambda: QueryOracle(target), square_attack, x_bb, y_bb, b,
                                       epsilon=args.bb_epsilon, seed=0)
        sr_bd, bd_last = _success_rate(lambda: QueryOracle(target), boundary_attack, x_bb, y_bb, b,
                                       seed=0)
        sq_curve.append(sr_sq)
        bd_curve.append(sr_bd)
        print(f"  budget={b:<4}  Square SR={sr_sq * 100:5.1f}%   Boundary SR={sr_bd * 100:5.1f}%")
    query_fig = _plot_queries(sq_curve, bd_curve, args.budgets, FIG_DIR / "blackbox_vs_budget.png")

    avg_q_sq = float(np.mean(sq_last.queries_per_sample)) if sq_last else 0.0
    avg_q_bd = float(np.mean(bd_last.queries_per_sample)) if bd_last else 0.0

    metrics = {
        "project": "p4-transfer-blackbox",
        "summary": (
            f"PGD on a CNN surrogate transfers to a different MLP target "
            f"(target acc drops {targ_acc * 100:.0f}% -> "
            f"{res['target'][max(res['target'])] * 100:.0f}% at eps="
            f"{max(res['target'])}); hand-rolled Square reaches "
            f"{max(sq_curve) * 100:.0f}% and Boundary {max(bd_curve) * 100:.0f}% "
            f"success under <= {max(args.budgets)} queries/image."
        ),
        "data": "real-MNIST" if args.real else "synthetic-glyphs",
        "seed": 42,
        "surrogate_arch": "SmallCNN",
        "target_arch": "SmallMLP",
        "surrogate_clean_accuracy": surr_acc,
        "target_clean_accuracy": targ_acc,
        "transfer": {
            "epsilons": [float(e) for e in sorted(res["target"])],
            "surrogate_accuracy_by_epsilon": {str(e): res["surrogate"][e]
                                              for e in sorted(res["surrogate"])},
            "target_accuracy_by_epsilon": {str(e): res["target"][e]
                                           for e in sorted(res["target"])},
        },
        "blackbox": {
            "n_images": int(x_bb.shape[0]),
            "square_epsilon_linf": args.bb_epsilon,
            "budgets": list(args.budgets),
            "square_success_rate_by_budget": dict(zip(map(str, args.budgets), sq_curve)),
            "boundary_success_rate_by_budget": dict(zip(map(str, args.budgets), bd_curve)),
            "square_avg_queries_used": avg_q_sq,
            "boundary_avg_queries_used": avg_q_bd,
        },
        "figures": [
            str(transfer_fig.relative_to(PROJECT)),
            str(query_fig.relative_to(PROJECT)),
        ],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {transfer_fig.relative_to(PROJECT)}")
    print(f"wrote {query_fig.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
