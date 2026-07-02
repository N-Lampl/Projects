"""NLP analysis of car reviews by brand and specific model (HuggingFace, CPU).

Public API:
    set_seed, get_device              -- reproducibility helpers (CPU-only)
    load_reviews, make_reviews        -- HF dataset loader + offline synthetic fallback
    parse_vehicle_title, add_parsed_columns
                                      -- Vehicle_Title -> year/make/model/model_key/trim
    get_sentiment_backend, MODEL_REGISTRY
                                      -- stub (offline) / HF sentiment backends
    validate_against_rating           -- predicted sentiment vs. ground-truth 1-5 Rating
    brand_sentiment, model_sentiment  -- ranked, shrinkage-corrected aggregates
    aspect_sentiment                  -- aspect-based sentiment (performance/comfort/...)
    tfidf_distinctive_terms           -- distinctive keywords per model
    fit_topics                        -- NMF/LDA topic modeling
    summarize_top_models              -- extractive (default) / abstractive summaries
"""

from .aggregate import (
    attach_scores,
    brand_sentiment,
    cap_per_group,
    model_sentiment,
    rank_top_bottom,
    to_records,
)
from .aspects import ASPECT_LEXICON, aspect_sentiment, split_sentences
from .data import HF_DATASET, load_reviews
from .evaluate import validate_against_rating
from .finetune import finetune_distilbert, load_finetuned
from .improve import (
    Calibrator,
    build_text,
    make_splits,
    predictions_from_star_probs,
    stratified_sample,
)
from .keywords import tfidf_distinctive_terms
from .parsing import (
    CANONICAL_MAKES,
    UNKNOWN,
    ParsedTitle,
    add_parsed_columns,
    make_coverage,
    parse_vehicle_title,
    unmatched_make_examples,
)
from .sentiment import MODEL_REGISTRY, HFBackend, Prediction, StubBackend, get_sentiment_backend
from .summarize import extractive_summary, summarize_top_models
from .synthetic import make_reviews
from .topics import fit_topics
from .utils import configure_torch_threads, get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "configure_torch_threads",
    "load_reviews",
    "HF_DATASET",
    "make_reviews",
    "parse_vehicle_title",
    "add_parsed_columns",
    "make_coverage",
    "unmatched_make_examples",
    "ParsedTitle",
    "CANONICAL_MAKES",
    "UNKNOWN",
    "get_sentiment_backend",
    "MODEL_REGISTRY",
    "Prediction",
    "StubBackend",
    "HFBackend",
    "validate_against_rating",
    "attach_scores",
    "brand_sentiment",
    "model_sentiment",
    "rank_top_bottom",
    "cap_per_group",
    "to_records",
    "aspect_sentiment",
    "ASPECT_LEXICON",
    "split_sentences",
    "tfidf_distinctive_terms",
    "fit_topics",
    "summarize_top_models",
    "extractive_summary",
    "make_splits",
    "stratified_sample",
    "build_text",
    "Calibrator",
    "predictions_from_star_probs",
    "finetune_distilbert",
    "load_finetuned",
]
