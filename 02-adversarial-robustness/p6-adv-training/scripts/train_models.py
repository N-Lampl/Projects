#!/usr/bin/env python3
"""Train the two models (standard + PGD-adversarial) and save weights.

Run via `make train`. Called automatically by `make run` if weights are missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adv_training import SmallCNN, evaluate, get_device, get_loaders, set_seed  # noqa: E402
from adv_training.train import save_model, train  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
STD_W = PROJECT / "models" / "standard.pt"
ADV_W = PROJECT / "models" / "adversarial.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--train-eps", type=float, default=0.1, help="PGD epsilon used during AT")
    ap.add_argument("--train-steps", type=int, default=7, help="PGD steps used during AT")
    ap.add_argument("--real", action="store_true", help="use real MNIST (needs torchvision)")
    args = ap.parse_args()

    set_seed()
    device = get_device()
    train_loader, test_loader = get_loaders(batch_size=args.batch_size, real=args.real)

    print("== standard training ==")
    set_seed()
    std = SmallCNN()
    train(std, train_loader, epochs=args.epochs, lr=args.lr, device=device, log_every=0)
    print(f"  standard clean acc: {evaluate(std, test_loader, device):.4f}")
    save_model(std, STD_W)

    print(f"== PGD adversarial training (eps={args.train_eps}, steps={args.train_steps}) ==")
    set_seed()
    adv = SmallCNN()
    train(
        adv,
        train_loader,
        epochs=args.epochs,
        lr=args.lr,
        adv_epsilon=args.train_eps,
        adv_steps=args.train_steps,
        device=device,
        log_every=0,
    )
    print(f"  adversarial clean acc: {evaluate(adv, test_loader, device):.4f}")
    save_model(adv, ADV_W)

    print(f"saved -> {STD_W.relative_to(PROJECT)}, {ADV_W.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
