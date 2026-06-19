#!/usr/bin/env python3
"""Compare standard vs PGD-adversarially-trained robustness across an epsilon sweep.

Trains both models if weights are missing, runs a PGD epsilon sweep on each,
writes the robustness-curve figure + a clean-vs-robust bar chart + metrics.json.
Run via `make run`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adv_training import (  # noqa: E402
    SmallCNN,
    accuracy_under_attack,
    get_device,
    get_loaders,
    set_seed,
)
from adv_training.train import load_model, save_model, train  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"
STD_W = PROJECT / "models" / "standard.pt"
ADV_W = PROJECT / "models" / "adversarial.pt"

DEFAULT_EPS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def _ensure_models(device, args) -> tuple[SmallCNN, SmallCNN]:
    if STD_W.exists() and ADV_W.exists():
        print(f"loading weights <- {STD_W.name}, {ADV_W.name}")
        return load_model(STD_W, device), load_model(ADV_W, device)

    print("weights missing - training both models...")
    train_loader, _ = get_loaders(batch_size=args.batch_size, real=args.real)

    set_seed()
    std = SmallCNN()
    train(std, train_loader, epochs=args.epochs, device=device)
    save_model(std, STD_W)

    set_seed()
    adv = SmallCNN()
    train(
        adv,
        train_loader,
        epochs=args.epochs,
        adv_epsilon=args.train_eps,
        adv_steps=args.train_steps,
        device=device,
    )
    save_model(adv, ADV_W)
    return std.eval(), adv.eval()


def _plot_curves(std_acc: dict, adv_acc: dict, train_eps: float) -> Path:
    eps = sorted(std_acc)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot([e for e in eps], [std_acc[e] * 100 for e in eps], "o-",
            color="#c0392b", linewidth=2, label="standard training")
    ax.plot([e for e in eps], [adv_acc[e] * 100 for e in eps], "s-",
            color="#27ae60", linewidth=2, label="PGD adversarial training")
    ax.axvline(train_eps, color="gray", linestyle="--", alpha=0.6)
    ax.annotate(f"AT train eps={train_eps}", (train_eps, 8),
                fontsize=8, color="gray", rotation=90, va="bottom", ha="right")
    ax.set_xlabel("PGD epsilon (L-inf perturbation budget)")
    ax.set_ylabel("test accuracy under PGD (%)")
    ax.set_title("Robustness curves: adversarial training holds up under attack", pad=12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "robustness_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_bars(std_acc: dict, adv_acc: dict, eval_eps: float) -> Path:
    labels = ["clean (eps=0)", f"PGD (eps={eval_eps})"]
    std_vals = [std_acc[0.0] * 100, std_acc[eval_eps] * 100]
    adv_vals = [adv_acc[0.0] * 100, adv_acc[eval_eps] * 100]
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w / 2 for i in x], std_vals, w, color="#c0392b", label="standard")
    ax.bar([i + w / 2 for i in x], adv_vals, w, color="#27ae60", label="adversarial")
    for i, (s, a) in enumerate(zip(std_vals, adv_vals)):
        ax.annotate(f"{s:.0f}", (i - w / 2, s), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
        ax.annotate(f"{a:.0f}", (i + w / 2, a), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("The robustness trade-off: clean accuracy vs PGD robustness", pad=12)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "clean_vs_robust_bars.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilons", type=float, nargs="+", default=DEFAULT_EPS)
    ap.add_argument("--eval-steps", type=int, default=7, help="PGD steps at eval time")
    ap.add_argument("--eval-eps", type=float, default=0.1, help="epsilon for the bar chart")
    ap.add_argument("--test-subset", type=int, default=1000, help="images for the sweep")
    ap.add_argument("--epochs", type=int, default=4, help="epochs if auto-training")
    ap.add_argument("--train-eps", type=float, default=0.1, help="PGD epsilon used during AT")
    ap.add_argument("--train-steps", type=int, default=7, help="PGD steps used during AT")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--real", action="store_true", help="use real MNIST (needs torchvision)")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    std, adv = _ensure_models(device, args)

    _, sweep_loader = get_loaders(batch_size=256, test_subset=args.test_subset, real=args.real)
    n_eval = len(sweep_loader.dataset)

    print(f"running PGD sweep over {args.epsilons} on {n_eval} images...")
    std_acc = accuracy_under_attack(std, sweep_loader, args.epsilons, steps=args.eval_steps,
                                    device=device)
    adv_acc = accuracy_under_attack(adv, sweep_loader, args.epsilons, steps=args.eval_steps,
                                    device=device)

    print(f"{'eps':>6} {'standard':>10} {'adversarial':>12}")
    for e in sorted(std_acc):
        print(f"{e:>6} {std_acc[e] * 100:>9.1f}% {adv_acc[e] * 100:>11.1f}%")

    curve = _plot_curves(std_acc, adv_acc, args.train_eps)
    bars = _plot_bars(std_acc, adv_acc, args.eval_eps)

    ee = args.eval_eps
    metrics = {
        "project": "p6-adv-training",
        "summary": (
            f"PGD adversarial training (eps={args.train_eps}) lifts robust accuracy at "
            f"PGD eps={ee} from {std_acc[ee] * 100:.1f}% (standard) to "
            f"{adv_acc[ee] * 100:.1f}%, while clean accuracy moves from "
            f"{std_acc[0.0] * 100:.1f}% to {adv_acc[0.0] * 100:.1f}%."
        ),
        "attack": "PGD (L-inf, multi-step, random start)",
        "data": "real MNIST" if args.real else "synthetic (offline default)",
        "seed": 42,
        "n_eval_images": n_eval,
        "train_epsilon": args.train_eps,
        "train_pgd_steps": args.train_steps,
        "eval_pgd_steps": args.eval_steps,
        "clean_accuracy": {"standard": std_acc[0.0], "adversarial": adv_acc[0.0]},
        "robust_accuracy_at_eval_eps": {
            "eval_epsilon": ee,
            "standard": std_acc.get(ee),
            "adversarial": adv_acc.get(ee),
            "robustness_gain": (
                (adv_acc.get(ee, 0) - std_acc.get(ee, 0)) if ee in std_acc else None
            ),
        },
        "accuracy_by_epsilon": {
            "standard": {str(e): std_acc[e] for e in sorted(std_acc)},
            "adversarial": {str(e): adv_acc[e] for e in sorted(adv_acc)},
        },
        "figures": [str(curve.relative_to(PROJECT)), str(bars.relative_to(PROJECT))],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {curve.relative_to(PROJECT)}")
    print(f"wrote {bars.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
