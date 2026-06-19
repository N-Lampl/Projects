#!/usr/bin/env python3
"""Train the target classifier (if needed), reconstruct one image per class via
gradient-ascent model inversion, and write the reconstruction figures + the
inversion metrics. Run via `make invert`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inversion_attribute import (  # noqa: E402
    SmallCNN,
    class_prototypes,
    evaluate,
    get_device,
    get_loaders,
    invert_all_classes,
    reconstruction_quality,
    set_seed,
)
from inversion_attribute.train import load_model, save_model, train  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics_inversion.json"
WEIGHTS = PROJECT / "models" / "smallcnn_target.pt"


def _get_model(device: torch.device, epochs: int, use_mnist: bool) -> SmallCNN:
    train_loader, test_loader = get_loaders(use_mnist=use_mnist)
    if WEIGHTS.exists():
        print(f"loading weights <- {WEIGHTS.relative_to(PROJECT)}")
        model = load_model(WEIGHTS, device)
    else:
        print("no weights found - training a fresh target model...")
        model = SmallCNN()
        train(model, train_loader, epochs=epochs, device=device)
        save_model(model, WEIGHTS)
        model.eval()
    acc = evaluate(model, test_loader, device=device)
    print(f"target test accuracy = {acc * 100:.1f}%")
    return model, acc


def _plot_reconstructions(recon: torch.Tensor, protos: torch.Tensor, confs: list[float]) -> Path:
    n = recon.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.6))
    for c in range(n):
        axes[0, c].imshow(protos[c], cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"class {c}", fontsize=9)
        axes[1, c].imshow(recon[c, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, c].set_title(f"conf {confs[c]:.2f}", fontsize=8)
        for ax in (axes[0, c], axes[1, c]):
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("true\nprototype", fontsize=9)
    axes[1, 0].set_ylabel("inverted\n(recovered)", fontsize=9)
    fig.suptitle("Model inversion: gradient ascent reconstructs each class signature", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "inversion_reconstructions.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300, help="gradient-ascent steps per class")
    ap.add_argument("--step-size", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=5, help="epochs if auto-training the target")
    ap.add_argument("--mnist", action="store_true", help="use real MNIST (downloads) instead of synthetic")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, acc = _get_model(device, args.epochs, args.mnist)

    print(f"running model inversion ({args.steps} steps/class)...")
    recon, confs = invert_all_classes(
        model, n_classes=10, device=device, steps=args.steps, step_size=args.step_size
    )
    for c, conf in enumerate(confs):
        print(f"  class {c}: recovered with confidence {conf:.3f}")

    protos = torch.tensor(class_prototypes())
    quality = reconstruction_quality(recon, protos)
    print(f"reconstruction quality: {quality}")

    fig = _plot_reconstructions(recon, protos.numpy(), confs)

    metrics = {
        "project": "p4-inversion-attribute",
        "task": "model-inversion",
        "summary": (
            f"Gradient-ascent model inversion recovers all 10 class signatures from a "
            f"{acc * 100:.0f}%-accurate classifier (top-1 prototype match "
            f"{quality['top1_match_rate'] * 100:.0f}%)."
        ),
        "seed": 42,
        "data": "synthetic" if not args.mnist else "mnist",
        "target_test_accuracy": acc,
        "inversion_steps": args.steps,
        "mean_recovery_confidence": sum(confs) / len(confs),
        "per_class_confidence": {str(c): confs[c] for c in range(10)},
        "mean_own_class_correlation": quality["mean_own_class_correlation"],
        "top1_match_rate": quality["top1_match_rate"],
        "figures": [str(fig.relative_to(PROJECT))],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {fig.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
