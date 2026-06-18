#!/usr/bin/env python3
"""Run the FGSM epsilon sweep, write the money plot + a clean-vs-adversarial grid
+ metrics.json. Auto-trains the model if weights are missing. Run via `make attack`.
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

from fgsm_mnist import (  # noqa: E402
    SmallCNN,
    accuracy_under_attack,
    evaluate,
    fgsm_perturb,
    get_device,
    get_loaders,
    set_seed,
)
from fgsm_mnist.train import load_model, save_model, train  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"
WEIGHTS = PROJECT / "models" / "smallcnn_mnist.pt"

DEFAULT_EPS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def _get_model(device: torch.device, epochs: int) -> SmallCNN:
    if WEIGHTS.exists():
        print(f"loading weights <- {WEIGHTS.relative_to(PROJECT)}")
        return load_model(WEIGHTS, device)
    print("no weights found - training a fresh model...")
    train_loader, _ = get_loaders()
    model = SmallCNN()
    train(model, train_loader, epochs=epochs, device=device)
    save_model(model, WEIGHTS)
    return model.eval()


def _plot_curve(acc_by_eps: dict[float, float]) -> Path:
    eps = sorted(acc_by_eps)
    accs = [acc_by_eps[e] * 100 for e in eps]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eps, accs, "o-", color="#c0392b", linewidth=2)
    ax.set_xlabel("epsilon (L-inf perturbation budget)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("FGSM on MNIST: accuracy collapses as epsilon grows", pad=12)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    for e, a in zip(eps, accs):
        ax.annotate(f"{a:.0f}", (e, a), textcoords="offset points", xytext=(0, 8), fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "accuracy_vs_epsilon.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_grid(model: SmallCNN, device: torch.device, eps_demo: float, n: int = 6) -> Path:
    """Show n correctly-classified digits, clean vs adversarial, with predictions."""
    _, test_loader = get_loaders(batch_size=256, test_subset=256)
    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        clean_pred = model(x).argmax(1)
    keep = (clean_pred == y).nonzero(as_tuple=True)[0][:n]
    x, y = x[keep], y[keep]

    x_adv = fgsm_perturb(model, x, y, eps_demo)
    with torch.no_grad():
        adv_pred = model(x_adv).argmax(1)

    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.6))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"pred {clean_pred[keep][i].item()}", fontsize=9)
        axes[1, i].imshow(x_adv[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        color = "green" if adv_pred[i] == y[i] else "red"
        axes[1, i].set_title(f"pred {adv_pred[i].item()}", fontsize=9, color=color)
        for ax in (axes[0, i], axes[1, i]):
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("clean", fontsize=10)
    axes[1, 0].set_ylabel(f"FGSM eps={eps_demo}", fontsize=10)
    fig.suptitle(f"Same digits + a small L-inf FGSM perturbation (eps={eps_demo}) -> flipped predictions", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "clean_vs_adversarial.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilons", type=float, nargs="+", default=DEFAULT_EPS)
    ap.add_argument("--test-subset", type=int, default=1000, help="images for the sweep (--full for all)")
    ap.add_argument("--full", action="store_true", help="evaluate on the full 10k test set")
    ap.add_argument("--eps-demo", type=float, default=0.2, help="epsilon for the example grid")
    ap.add_argument("--epochs", type=int, default=2, help="epochs if auto-training")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model = _get_model(device, args.epochs)

    subset = None if args.full else args.test_subset
    _, sweep_loader = get_loaders(batch_size=256, test_subset=subset)
    n_eval = len(sweep_loader.dataset)

    print(f"running FGSM sweep over {args.epsilons} on {n_eval} images...")
    acc_by_eps = accuracy_under_attack(model, sweep_loader, args.epsilons, device=device)
    for e in sorted(acc_by_eps):
        print(f"  eps={e:<5} accuracy={acc_by_eps[e] * 100:5.1f}%")

    curve = _plot_curve(acc_by_eps)
    grid = _plot_grid(model, device, args.eps_demo)

    metrics = {
        "project": "p1-fgsm-mnist",
        "attack": "FGSM (L-inf, single-step)",
        "seed": 42,
        "n_eval_images": n_eval,
        "clean_accuracy": acc_by_eps.get(0.0),
        "accuracy_by_epsilon": {str(e): acc_by_eps[e] for e in sorted(acc_by_eps)},
        "eps_demo": args.eps_demo,
        "figures": [str(curve.relative_to(PROJECT)), str(grid.relative_to(PROJECT))],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {curve.relative_to(PROJECT)}")
    print(f"wrote {grid.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
