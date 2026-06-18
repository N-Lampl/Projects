#!/usr/bin/env python3
"""Train the SmallCNN on MNIST and save weights. Run via `make train`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgsm_mnist import SmallCNN, evaluate, get_device, get_loaders, set_seed  # noqa: E402
from fgsm_mnist.train import save_model  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", type=Path, default=PROJECT / "models" / "smallcnn_mnist.pt")
    args = ap.parse_args()

    set_seed()
    device = get_device()
    train_loader, test_loader = get_loaders(batch_size=args.batch_size)

    model = SmallCNN()
    from fgsm_mnist import train

    train(model, train_loader, epochs=args.epochs, lr=args.lr, device=device)
    acc = evaluate(model, test_loader, device=device)
    print(f"clean test accuracy: {acc:.4f}")

    save_model(model, args.out)
    print(f"saved weights -> {args.out.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
