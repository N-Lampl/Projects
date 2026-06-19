#!/usr/bin/env python3
"""Train the base classifier with Gaussian noise augmentation and save weights.

Run via `make train`. The certify script auto-trains too, so this is optional.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rand_smoothing import (  # noqa: E402
    SmallCNN,
    evaluate,
    get_device,
    get_loaders,
    set_seed,
    train,
)
from rand_smoothing.train import save_model  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "mnist"], default="synthetic")
    ap.add_argument("--sigma", type=float, default=0.5, help="noise sigma for augmentation")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-subset", type=int, default=6000)
    args = ap.parse_args()

    set_seed()
    device = get_device()
    train_loader, test_loader = get_loaders(dataset=args.dataset, train_subset=args.train_subset)
    model = SmallCNN()
    train(model, train_loader, sigma=args.sigma, epochs=args.epochs, device=device)
    acc = evaluate(model, test_loader, device=device)
    print(f"base (clean) test accuracy: {acc * 100:.1f}%")

    weights = PROJECT / "models" / f"base_{args.dataset}_sigma{args.sigma}.pt"
    save_model(model, weights)
    print(f"wrote {weights.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
