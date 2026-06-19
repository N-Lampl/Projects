#!/usr/bin/env python3
"""Certify a smoothed classifier on a small test subset and write:

  results/figures/certified_accuracy_vs_radius.png   <- the money plot
  results/figures/radius_histogram.png               <- distribution of certified radii
  results/metrics.json

Cohen et al. (2019), Algorithm 1: per-point CERTIFY with N MC samples and a
Clopper-Pearson lower bound. Auto-trains the (noise-augmented) base model if its
weights are missing. Run via `make certify`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rand_smoothing import (  # noqa: E402
    ABSTAIN,
    SmallCNN,
    SmoothedClassifier,
    certified_accuracy_at,
    get_device,
    get_loaders,
    set_seed,
)
from rand_smoothing.smoothing import _HAVE_SCIPY  # noqa: E402
from rand_smoothing.train import load_model, save_model, train  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _get_model(dataset: str, sigma: float, device: torch.device, epochs: int) -> SmallCNN:
    weights = PROJECT / "models" / f"base_{dataset}_sigma{sigma}.pt"
    if weights.exists():
        print(f"loading weights <- {weights.relative_to(PROJECT)}")
        return load_model(weights, device)
    print("no weights found - training a noise-augmented base model...")
    train_loader, _ = get_loaders(dataset=dataset)
    model = SmallCNN()
    train(model, train_loader, sigma=sigma, epochs=epochs, device=device)
    save_model(model, weights)
    return model.eval()


def _plot_curve(grid: np.ndarray, cert_acc: np.ndarray, sigma: float, n: int) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, cert_acc * 100, "-", color="#1f6feb", linewidth=2)
    ax.fill_between(grid, cert_acc * 100, alpha=0.12, color="#1f6feb")
    ax.set_xlabel("L2 radius r (pixel units)")
    ax.set_ylabel("certified accuracy (%)")
    ax.set_title(
        f"Randomized smoothing: certified accuracy vs L2 radius\n"
        f"(sigma={sigma}, N={n} MC samples)",
        fontsize=11,
    )
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "certified_accuracy_vs_radius.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_hist(radii: np.ndarray, correct: np.ndarray) -> Path:
    cert = radii[(radii > 0) & correct]
    fig, ax = plt.subplots(figsize=(6, 4))
    if cert.size:
        ax.hist(cert, bins=20, color="#2da44e", edgecolor="white")
    ax.set_xlabel("certified L2 radius")
    ax.set_ylabel("# correctly-certified points")
    ax.set_title("Distribution of certified radii", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "radius_histogram.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "mnist"], default="synthetic")
    ap.add_argument("--sigma", type=float, default=0.5, help="smoothing noise level")
    ap.add_argument("--n0", type=int, default=100, help="selection samples")
    ap.add_argument("--n", type=int, default=1000, help="estimation (MC) samples")
    ap.add_argument("--alpha", type=float, default=0.001, help="1 - confidence")
    ap.add_argument("--num-points", type=int, default=100, help="test points to certify")
    ap.add_argument("--batch", type=int, default=200, help="noise batch size per forward pass")
    ap.add_argument("--epochs", type=int, default=3, help="epochs if auto-training")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model = _get_model(args.dataset, args.sigma, device, args.epochs)
    smoothed = SmoothedClassifier(model, sigma=args.sigma, num_classes=10, device=device)

    # collect the points to certify
    _, test_loader = get_loaders(dataset=args.dataset, batch_size=256, test_subset=args.num_points)
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x)
        ys.append(y)
    xs = torch.cat(xs)[: args.num_points]
    ys = torch.cat(ys)[: args.num_points]

    print(f"certifying {len(xs)} points  (sigma={args.sigma}, n0={args.n0}, "
          f"N={args.n}, alpha={args.alpha}, scipy={'yes' if _HAVE_SCIPY else 'no/fallback'})")

    radii = np.zeros(len(xs))
    preds = np.full(len(xs), ABSTAIN)
    for i in range(len(xs)):
        c, r = smoothed.certify(
            xs[i : i + 1], n0=args.n0, n=args.n, alpha=args.alpha, batch=args.batch
        )
        preds[i] = c
        radii[i] = r
        if (i + 1) % 20 == 0:
            print(f"  certified {i + 1}/{len(xs)}")

    labels = ys.numpy()
    correct = (preds == labels) & (preds != ABSTAIN)
    abstained = int((preds == ABSTAIN).sum())

    # certified-accuracy-vs-radius curve
    r_max = float(radii.max()) if radii.max() > 0 else args.sigma
    grid = np.linspace(0.0, max(r_max, 0.5), 60)
    cert_acc = np.array([certified_accuracy_at(radii, correct, r) for r in grid])

    curve = _plot_curve(grid, cert_acc, args.sigma, args.n)
    hist = _plot_hist(radii, correct)

    report_radii = [0.0, 0.25, 0.5, 0.75, 1.0]
    cert_at = {f"{r:.2f}": certified_accuracy_at(radii, correct, r) for r in report_radii}

    metrics = {
        "project": "p7-randomized-smoothing",
        "summary": (
            f"Cohen-2019 randomized smoothing on a {args.dataset} subset: certified clean "
            f"accuracy {cert_at['0.00'] * 100:.1f}%, mean certified L2 radius "
            f"{float(radii[correct].mean()) if correct.any() else 0.0:.3f} "
            f"(sigma={args.sigma}, N={args.n}, alpha={args.alpha})."
        ),
        "method": (
            "Randomized smoothing (Gaussian); Clopper-Pearson lower bound; "
            "R = sigma * Phi^-1(pA)"
        ),
        "dataset": args.dataset,
        "seed": 42,
        "sigma": args.sigma,
        "n0": args.n0,
        "n_samples": args.n,
        "alpha": args.alpha,
        "confidence": 1 - args.alpha,
        "num_points": int(len(xs)),
        "scipy_used": bool(_HAVE_SCIPY),
        "abstain_count": abstained,
        "abstain_rate": abstained / len(xs),
        "certified_clean_accuracy": cert_at["0.00"],
        "mean_certified_radius_correct": float(radii[correct].mean()) if correct.any() else 0.0,
        "max_certified_radius": float(radii.max()),
        "certified_accuracy_at_radius": cert_at,
        "figures": [str(curve.relative_to(PROJECT)), str(hist.relative_to(PROJECT))],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")

    print("\ncertified accuracy by radius:")
    for r, a in cert_at.items():
        print(f"  r>={r}: {a * 100:5.1f}%")
    print(f"abstained: {abstained}/{len(xs)}")
    print(f"\nwrote {curve.relative_to(PROJECT)}")
    print(f"wrote {hist.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
