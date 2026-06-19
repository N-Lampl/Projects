#!/usr/bin/env python3
"""Train BOTH small classifiers (CNN surrogate + MLP target) and save weights.
Run via `make train`. The attack script auto-trains too, so this is optional.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transfer_blackbox import build_model, evaluate, get_device, get_loaders, set_seed  # noqa: E402
from transfer_blackbox.train import save_model, train  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use real MNIST instead of synthetic")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    set_seed()
    device = get_device()
    train_loader, test_loader = get_loaders(real=args.real)

    for kind in ("cnn", "mlp"):
        set_seed(42 if kind == "cnn" else 7)  # different seeds -> different solutions
        model = build_model(kind)
        train(model, train_loader, epochs=args.epochs, device=device)
        acc = evaluate(model, test_loader, device=device)
        path = MODELS / f"{kind}.pt"
        save_model(model, path)
        print(f"{kind.upper():4s} clean accuracy = {acc * 100:5.1f}%  -> {path.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
