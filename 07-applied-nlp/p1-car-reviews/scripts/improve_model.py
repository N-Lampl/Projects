#!/usr/bin/env python3
"""Compare the baseline sentiment model against two improvements — calibration and
(if present) a fine-tuned DistilBERT — on the SAME held-out test set, and write
results/improvement.json + results/figures/baseline_vs_improved.png.

Run via `make improve`. Before `make finetune` it reports baseline vs. calibrated;
after, it also folds in the fine-tuned model. All three are scored on the identical
seeded test split, so the before -> after deltas are apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_reviews import (  # noqa: E402
    Calibrator,
    build_text,
    get_sentiment_backend,
    load_reviews,
    make_splits,
    predictions_from_star_probs,
    set_seed,
    stratified_sample,
    validate_against_rating,
)
from car_reviews.finetune import load_finetuned  # noqa: E402
from car_reviews.plots import plot_improvement  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
OUT = PROJECT / "results" / "improvement.json"


def _line(name: str, m: dict) -> str:
    return (
        f"{name}: exact {m['exact_accuracy']:.2f}, +/-1 {m['within_1_accuracy']:.2f}, "
        f"MAE {m['mae']:.2f}, Spearman {m.get('spearman')}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-sample", type=int, default=8000, help="reviews to fit the calibrator")
    ap.add_argument("--finetuned", default="models/finetuned", help="fine-tuned model dir")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_reviews(full=True, seed=args.seed)
    train, _val, test = make_splits(df, seed=args.seed)
    print(f"splits: train={len(train)} test={len(test)} (held-out)")

    backend = get_sentiment_backend(
        "hf", model="nlptown", max_length=args.max_length, batch_size=args.batch_size
    )
    test_texts = build_text(test)
    print("[baseline] scoring test set with nlptown...")
    test_probs, _ = backend.predict_proba(test_texts)
    baseline = validate_against_rating(
        predictions_from_star_probs(test_probs), test["Rating"], "1-5"
    )

    print("[calibrate] scoring calibrator-fit subset with nlptown...")
    fit_df = stratified_sample(train, args.calib_sample, seed=args.seed)
    fit_probs, _ = backend.predict_proba(build_text(fit_df))
    cal = Calibrator(seed=args.seed).fit(fit_probs, fit_df["Rating"])
    calibrated = validate_against_rating(cal.predict(test_probs), test["Rating"], "1-5")

    results = {"baseline": baseline, "calibrated": calibrated}

    ft_dir = args.finetuned if Path(args.finetuned).is_absolute() else PROJECT / args.finetuned
    if (Path(ft_dir) / "config.json").exists():
        print(f"[fine-tuned] scoring test set with {ft_dir}...")
        ft = load_finetuned(ft_dir, max_length=args.max_length, batch_size=args.batch_size)
        ft_probs, _ = ft.predict_proba(test_texts)
        results["fine-tuned"] = validate_against_rating(
            predictions_from_star_probs(ft_probs), test["Rating"], "1-5"
        )
    else:
        print(f"[fine-tuned] none at {ft_dir}; run `make finetune`, then re-run `make improve`.")

    fig = plot_improvement(results, FIG_DIR / "baseline_vs_improved.png")
    summary = " | ".join(_line(k, v) for k, v in results.items())
    payload = {
        "project": "p1-car-reviews",
        "task": "model improvement — baseline vs. calibrated vs. fine-tuned",
        "test_n": int(len(test)),
        "seed": args.seed,
        "results": results,
        "figure": str(fig.relative_to(PROJECT)),
        "summary": summary,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n" + summary)
    for p in (fig, OUT):
        print(f"wrote {p.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
