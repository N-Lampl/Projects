#!/usr/bin/env python3
"""Analyze car reviews by brand & model with a HuggingFace sentiment model, then
write five figures + metrics.json.

Pipeline: load reviews -> parse Vehicle_Title into make/model -> score sentiment
(HF, CPU) -> validate against the 1-5 Rating -> rank brands & models -> aspect
sentiment -> distinctive keywords -> topics -> per-model summaries.

Run via `make run` (fast --sample) or `make run-full` (all 36,984 reviews, the
committed artifact). Tests/CI use `--backend stub` (offline, no downloads).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_reviews import (  # noqa: E402
    MODEL_REGISTRY,
    UNKNOWN,
    add_parsed_columns,
    aspect_sentiment,
    attach_scores,
    brand_sentiment,
    fit_topics,
    get_sentiment_backend,
    load_reviews,
    make_coverage,
    model_sentiment,
    set_seed,
    summarize_top_models,
    tfidf_distinctive_terms,
    to_records,
    unmatched_make_examples,
    validate_against_rating,
)
from car_reviews.aggregate import cap_per_group  # noqa: E402
from car_reviews.plots import (  # noqa: E402
    plot_aspect_heatmap,
    plot_brand_ranking,
    plot_model_ranking,
    plot_sentiment_vs_rating_confusion,
    plot_topics_prevalence,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _sentiment_texts(df: pd.DataFrame) -> list[str]:
    """Review title + body (title first — it is short and strongly polar)."""
    titles = df["Review_Title"].fillna("").astype(str).str.strip()
    reviews = df["Review"].astype(str).str.strip()
    return [f"{t}. {r}" if t else r for t, r in zip(titles, reviews, strict=False)]


def _headline(model_id: str, n: int, n_total: int, full: bool, val: dict, brand_tbl) -> str:
    where = "reviews (full dataset)" if full else f"sampled reviews of {n_total:,}"
    if val["scale"] == "1-5":
        rho = val.get("spearman")
        rho_str = f"{rho:.2f}" if rho is not None else "n/a"
        core = (
            f"predicted stars match the Rating within +/-1 for "
            f"{val['within_1_accuracy'] * 100:.0f}% of reviews "
            f"(exact {val['exact_accuracy'] * 100:.0f}%, MAE {val['mae']:.2f}, "
            f"Spearman {rho_str})"
        )
    else:
        core = f"sentiment-vs-Rating accuracy {val['accuracy'] * 100:.0f}%"
    top = brand_tbl.iloc[0] if len(brand_tbl) else None
    bottom = brand_tbl.iloc[-1] if len(brand_tbl) else None
    brands = ""
    if top is not None:
        ts, bs = top["mean_sentiment_shrunk"], bottom["mean_sentiment_shrunk"]
        brands = (
            f" Top brand = {top['make']} ({ts:.2f}, n={int(top['n'])}), "
            f"bottom = {bottom['make']} ({bs:.2f}, n={int(bottom['n'])})."
        )
    return f"{model_id} on {n:,} {where}: {core}.{brands}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sample", type=int, default=3000, help="rows to analyze (stratified by Rating)"
    )
    ap.add_argument("--full", action="store_true", help="use all reviews (the committed artifact)")
    ap.add_argument(
        "--backend", choices=["hf", "stub"], default="hf", help="stub = offline lexicon"
    )
    ap.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="nlptown")
    ap.add_argument("--per-model-cap", type=int, default=200, help="cap reviews/model for ranking")
    ap.add_argument("--min-reviews-per-model", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--aspect-sample", type=int, default=5000, help="cap reviews for aspect pass")
    ap.add_argument("--n-topics", type=int, default=8)
    ap.add_argument("--topic-model", choices=["nmf", "lda"], default="nmf")
    ap.add_argument(
        "--summaries", choices=["extractive", "abstractive", "none"], default="extractive"
    )
    ap.add_argument("--summary-top-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- load + parse --------------------------------------------------------
    df = load_reviews(sample=args.sample, full=args.full, seed=args.seed)
    df = add_parsed_columns(df)
    source = df.attrs.get("source", "unknown")
    n_total = df.attrs.get("n_total", len(df))
    coverage = make_coverage(df)
    model_id = MODEL_REGISTRY[args.model].model_id if args.backend == "hf" else "stub-lexicon"
    print(f"{source}: {len(df):,} reviews (of {n_total:,}) | make coverage {coverage * 100:.1f}%")

    # --- sentiment inference (the main CPU cost) -----------------------------
    backend = get_sentiment_backend(
        args.backend, model=args.model, max_length=args.max_length, batch_size=args.batch_size
    )
    preds = backend.predict(_sentiment_texts(df))
    scored = attach_scores(df, preds)
    val = validate_against_rating(preds, scored["Rating"], backend.scale)

    # --- brand & model rankings ---------------------------------------------
    brand_tbl = brand_sentiment(scored, min_reviews=1)
    capped = cap_per_group(scored, "model_key", args.per_model_cap, seed=args.seed)
    model_tbl = model_sentiment(capped, min_reviews=args.min_reviews_per_model)
    n_models_total = int(scored.loc[scored["make"] != UNKNOWN, "model_key"].nunique())

    # --- aspects, keywords, topics, summaries --------------------------------
    aspects = aspect_sentiment(scored, backend, max_reviews=args.aspect_sample, seed=args.seed)
    keywords = tfidf_distinctive_terms(scored, min_reviews=args.min_reviews_per_model)
    topics = fit_topics(scored, n_topics=args.n_topics, method=args.topic_model, seed=args.seed)
    summaries = summarize_top_models(
        scored,
        method=args.summaries,
        top_n=args.summary_top_n,
        min_reviews=args.min_reviews_per_model,
        seed=args.seed,
    )

    # --- figures -------------------------------------------------------------
    top_brands = (
        scored.loc[scored["make"] != UNKNOWN, "make"].value_counts().head(10).index.tolist()
    )
    figures = []
    if "confusion" in val:
        figures.append(
            plot_sentiment_vs_rating_confusion(
                val["confusion"],
                val["confusion_labels"],
                FIG_DIR / "sentiment_vs_rating_confusion.png",
                model_id,
            )
        )
    figures.append(plot_brand_ranking(brand_tbl, FIG_DIR / "brand_sentiment_ranking.png"))
    if len(model_tbl):
        figures.append(plot_model_ranking(model_tbl, FIG_DIR / "model_sentiment_ranking.png"))
    if len(aspects["by_brand"].columns):
        figures.append(
            plot_aspect_heatmap(
                aspects["by_brand"], top_brands, FIG_DIR / "aspect_sentiment_heatmap.png"
            )
        )
    figures.append(plot_topics_prevalence(topics["topics"], FIG_DIR / "topics_prevalence.png"))

    # --- metrics.json --------------------------------------------------------
    summary = _headline(model_id, len(scored), n_total, args.full, val, brand_tbl)
    metrics = {
        "project": "p1-car-reviews",
        "summary": summary,
        "data_source": source,
        "backend": args.backend,
        "model_id": model_id,
        "seed": args.seed,
        "full": bool(args.full),
        "n_reviews": int(len(scored)),
        "n_reviews_total": int(n_total),
        "n_brands": int(len(brand_tbl)),
        "n_models": n_models_total,
        "n_models_ranked": int(len(model_tbl)),
        "min_reviews_per_model": args.min_reviews_per_model,
        "per_model_cap": args.per_model_cap,
        "parse_make_coverage": round(coverage, 4),
        "parse_unmatched_make_examples": unmatched_make_examples(df),
        "sentiment_vs_rating": val,
        "top_brands": to_records(brand_tbl, "make", k=15),
        "bottom_brands": to_records(brand_tbl.iloc[::-1], "make", k=15),
        "top_models": to_records(model_tbl, "model_key", k=15),
        "bottom_models": to_records(model_tbl.iloc[::-1], "model_key", k=15),
        "aspects": aspects["overall"],
        "keywords_by_model": keywords,
        "topics": topics,
        "summaries": {"method": args.summaries, "top_n": args.summary_top_n, "by_model": summaries},
        "figures": [str(p.relative_to(PROJECT)) for p in figures],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")

    print("\n" + summary)
    for p in (*figures, METRICS):
        print(f"wrote {p.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
