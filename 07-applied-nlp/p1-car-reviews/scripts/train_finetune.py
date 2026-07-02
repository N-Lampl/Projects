#!/usr/bin/env python3
"""Fine-tune DistilBERT on the car-review star ratings and save it to
models/finetuned (git-ignored). This is the slow, CPU-bound improvement step
(~1-2 h on a laptop); run it, then `make improve` folds it into the comparison.

The held-out test split (seeded, shared with `make improve`) is NEVER trained on,
so the before -> after numbers stay honest.

Run via `make finetune` (or `make finetune ARGS='--train-sample 25000 --epochs 3'`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_reviews import load_reviews, make_splits, set_seed, stratified_sample  # noqa: E402
from car_reviews.finetune import finetune_distilbert  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-sample", type=int, default=15000, help="capped train reviews")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default="models/finetuned")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    df = load_reviews(full=True, seed=args.seed)
    train, val, test = make_splits(df, seed=args.seed)
    train = stratified_sample(train, args.train_sample, seed=args.seed)
    print(
        f"fine-tuning DistilBERT on {len(train)} reviews "
        f"({args.epochs} epochs, max_len {args.max_length}); "
        f"held-out test={len(test)} stays untouched"
    )

    out = Path(args.out) if Path(args.out).is_absolute() else PROJECT / args.out
    t0 = time.time()
    finetune_distilbert(
        train,
        val_df=val,
        out_dir=out,
        epochs=args.epochs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")
    print("now run:  make improve   (adds the fine-tuned column to the comparison)")


if __name__ == "__main__":
    main()
