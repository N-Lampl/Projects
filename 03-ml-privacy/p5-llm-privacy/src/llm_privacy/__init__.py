"""LLM training-data memorization: insert canaries, measure leakage via exposure.

Public API:
    set_seed, get_device          -- reproducibility helpers
    build_corpus, Canary          -- synthetic corpus with inserted secrets
    CharLM                        -- the tiny char-level language model (GRU)
    train, save_model, load_model -- next-char training + checkpoints
    sequence_log_perplexity       -- per-string perplexity (the membership signal)
    estimate_exposure             -- the canary-exposure metric (Secret Sharer)
    empirical_rank                -- distribution-free rank check
"""

from .corpus import Canary, build_corpus, make_canary, random_secret
from .exposure import (
    RANDOMNESS_SPACE,
    ExposureResult,
    empirical_rank,
    estimate_exposure,
    sequence_log_perplexity,
)
from .model import CharLM
from .train import load_model, save_model, train
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "build_corpus",
    "Canary",
    "make_canary",
    "random_secret",
    "CharLM",
    "train",
    "save_model",
    "load_model",
    "sequence_log_perplexity",
    "estimate_exposure",
    "empirical_rank",
    "ExposureResult",
    "RANDOMNESS_SPACE",
]
