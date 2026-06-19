#!/usr/bin/env python3
"""Train the SmallCNN target classifier and save weights. Run via `make train`.

Default source is the offline SYNTHETIC dataset (no download). Pass
`--source cifar10` or `--source mnist` for the optional real-data path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attack_zoo import SmallCNN, evaluate, get_device, get_loaders, set_seed  # noqa: E402
from attack_zoo.train import save_model, train  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "cifar10", "mnist"])
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", type=Path, default=PROJECT / "models" / "smallcnn.pt")
    args = ap.parse_args()

    set_seed()
    device = get_device()
    train_loader, test_loader, meta = get_loaders(
        source=args.source, batch_size=args.batch_size, num_classes=args.num_classes
    )

    model = SmallCNN(in_channels=meta["in_channels"], num_classes=meta["num_classes"])
    train(model, train_loader, epochs=args.epochs, lr=args.lr, device=device)
    acc = evaluate(model, test_loader, device=device)
    print(f"clean test accuracy: {acc:.4f}")

    meta["source"] = args.source
    save_model(model, args.out, meta=meta)
    print(f"saved weights -> {args.out.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
