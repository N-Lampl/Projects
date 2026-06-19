#!/usr/bin/env python3
"""Train the victim classifier and save its weights. Run via `make train`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_extraction import (  # noqa: E402
    evaluate,
    get_device,
    get_splits,
    loader,
    make_victim,
    set_seed,
)
from model_extraction.train import save_model, train  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
WEIGHTS = PROJECT / "models" / "victim.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "mnist"], default="synthetic")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    set_seed()
    device = get_device()
    splits = get_splits(args.dataset)

    victim = make_victim(splits.img_size, splits.n_classes)
    train(victim, loader(splits.victim_x, splits.victim_y, shuffle=True), epochs=args.epochs,
          device=device)
    acc = evaluate(victim, loader(splits.test_x, splits.test_y), device=device)
    print(f"victim test accuracy: {acc * 100:.1f}%")

    save_model(victim, WEIGHTS, splits.img_size, splits.n_classes)
    print(f"wrote {WEIGHTS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
