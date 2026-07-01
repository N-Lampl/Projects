"""Pluggable sentiment backends.

Default (tests / CI): a deterministic **stub** lexicon scorer — no torch, no
transformers, no network — so the whole pipeline runs offline. The real run
uses the **HF** backend: a HuggingFace ``transformers`` sentiment pipeline loaded
lazily (weights download on first use), selected with ``--backend hf``.

Every backend returns :class:`Prediction` objects on a common footing:
``score`` in ``[0, 1]`` (higher = more positive) and ``star`` on the 1-5 scale,
so brand/model aggregation and the ``Rating`` validation are model-agnostic.

Authorized use only: models are public pretrained checkpoints used for analysis.
See ../../ETHICS.md.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

# --- model registry ----------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    scale: str  # "1-5" | "binary" | "3-class"
    kind: str  # "star5" | "binary" | "cardiff3"


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "nlptown": ModelSpec("nlptown/bert-base-multilingual-uncased-sentiment", "1-5", "star5"),
    "sst2": ModelSpec("distilbert-base-uncased-finetuned-sst-2-english", "binary", "binary"),
    "cardiff": ModelSpec("cardiffnlp/twitter-roberta-base-sentiment-latest", "3-class", "cardiff3"),
}


@dataclass(frozen=True)
class Prediction:
    score: float  # sentiment in [0, 1], higher = more positive
    star: float  # predicted star on the 1-5 scale (continuous expected value)
    label: str  # human-readable label


def _label_from_score(score: float) -> str:
    if score >= 0.6:
        return "positive"
    if score <= 0.4:
        return "negative"
    return "neutral"


# --- raw HF output -> Prediction converters ----------------------------------


def _as_map(scores: list[dict]) -> dict[str, float]:
    return {str(d["label"]).strip().lower(): float(d["score"]) for d in scores}


def _convert_star5(scores: list[dict]) -> Prediction:
    m = _as_map(scores)
    exp = 0.0
    for label, p in m.items():
        digit = re.match(r"(\d)", label)
        if digit:
            exp += int(digit.group(1)) * p
    star = exp if exp else 3.0
    return Prediction(score=(star - 1) / 4, star=star, label=_label_from_score((star - 1) / 4))


def _convert_binary(scores: list[dict]) -> Prediction:
    m = _as_map(scores)
    p_pos = m.get("positive", m.get("label_1", 0.0))
    return Prediction(score=p_pos, star=1 + 4 * p_pos, label=_label_from_score(p_pos))


def _convert_cardiff3(scores: list[dict]) -> Prediction:
    m = _as_map(scores)
    p_neu = m.get("neutral", m.get("label_1", 0.0))
    p_pos = m.get("positive", m.get("label_2", 0.0))
    score = p_neu * 0.5 + p_pos  # weights: neg=0, neu=0.5, pos=1
    return Prediction(score=score, star=1 + 4 * score, label=_label_from_score(score))


_CONVERTERS = {
    "star5": _convert_star5,
    "binary": _convert_binary,
    "cardiff3": _convert_cardiff3,
}


# --- backends ----------------------------------------------------------------


class SentimentBackend(Protocol):
    name: str
    scale: str

    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...


_POS_WORDS = {
    "love",
    "loved",
    "fantastic",
    "excellent",
    "great",
    "good",
    "recommend",
    "happy",
    "reliable",
    "dependable",
    "comfortable",
    "quiet",
    "quick",
    "power",
    "powerful",
    "smooth",
    "value",
    "saves",
    "safe",
    "strong",
    "stability",
    "well",
    "plenty",
    "amazing",
    "perfect",
    "best",
    "solid",
    "impressive",
    "enjoy",
    "efficient",
    "roomy",
    "spacious",
}
_NEG_WORDS = {
    "regret",
    "terrible",
    "awful",
    "disappointed",
    "disappointing",
    "sluggish",
    "weak",
    "cramped",
    "noisy",
    "harsh",
    "unreliable",
    "broke",
    "broken",
    "expensive",
    "repair",
    "overpriced",
    "poor",
    "unsafe",
    "problem",
    "problems",
    "issue",
    "issues",
    "worst",
    "bad",
    "hate",
    "junk",
    "cheap",
    "uncomfortable",
    "recall",
    "failure",
    "leak",
}
_WORD_RE = re.compile(r"[a-z']+")


class StubBackend:
    """Deterministic lexicon scorer. No torch/transformers/network — the CI path."""

    name = "stub"
    scale = "1-5"

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        out: list[Prediction] = []
        for text in texts:
            words = _WORD_RE.findall(str(text).lower())
            pos = sum(w in _POS_WORDS for w in words)
            neg = sum(w in _NEG_WORDS for w in words)
            total = pos + neg
            score = 0.5 if total == 0 else pos / total
            out.append(Prediction(score=score, star=1 + 4 * score, label=_label_from_score(score)))
        return out


class HFBackend:
    """HuggingFace ``transformers`` sentiment pipeline (lazy weight download)."""

    def __init__(
        self,
        spec: ModelSpec,
        max_length: int = 256,
        batch_size: int = 16,
        char_cap: int = 2000,
        chunk: int = 1024,
        verbose: bool = True,
    ):
        self.spec = spec
        self.name = "hf"
        self.scale = spec.scale
        self.model_id = spec.model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.char_cap = char_cap
        self.chunk = chunk
        self.verbose = verbose
        self._pipe = None
        self._convert = _CONVERTERS[spec.kind]

    def _build(self):
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - optional heavy path
            raise SystemExit(
                "transformers not installed. The HF backend is the real-run path.\n"
                "  pip install 'transformers>=4.44' 'torch>=2.2' datasets\n"
                "The offline path is `--backend stub` (no transformers needed)."
            ) from exc
        from .utils import configure_torch_threads

        configure_torch_threads()
        if self.verbose:
            print(f"[sentiment] loading {self.model_id} (downloads on first use)...")
        self._pipe = pipeline(
            "sentiment-analysis",
            model=self.model_id,
            tokenizer=self.model_id,
            top_k=None,  # return every class score, not just the argmax
            device=-1,  # CPU
        )

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        if self._pipe is None:
            self._build()
        prepped = [str(t)[: self.char_cap] for t in texts]
        n = len(prepped)
        # Length-sorted batching minimises pad waste; re-map to original order.
        order = sorted(range(n), key=lambda i: len(prepped[i]))
        results: list[Prediction | None] = [None] * n
        done = 0
        for start in range(0, n, self.chunk):
            idx = order[start : start + self.chunk]
            batch = [prepped[i] for i in idx]
            raw = self._pipe(
                batch, batch_size=self.batch_size, truncation=True, max_length=self.max_length
            )
            for local, gi in enumerate(idx):
                results[gi] = self._convert(raw[local])
            done += len(idx)
            if self.verbose and n > self.chunk:
                print(f"[sentiment] {done}/{n} reviews scored", flush=True)
        return [r for r in results if r is not None]


def get_sentiment_backend(
    name: str | None = None,
    model: str = "nlptown",
    max_length: int = 256,
    batch_size: int = 16,
    verbose: bool = True,
) -> SentimentBackend:
    """Factory. ``"stub"`` (offline default) or ``"hf"`` (real transformers).

    The backend can also be forced with the ``CAR_REVIEWS_BACKEND`` env var.
    """
    name = (name or os.environ.get("CAR_REVIEWS_BACKEND") or "stub").lower()
    if name == "stub":
        return StubBackend()
    if name == "hf":
        if model not in MODEL_REGISTRY:
            raise ValueError(f"unknown model {model!r}; choose from {sorted(MODEL_REGISTRY)}")
        return HFBackend(
            MODEL_REGISTRY[model],
            max_length=max_length,
            batch_size=batch_size,
            verbose=verbose,
        )
    raise ValueError(f"unknown backend {name!r}; choose 'stub' or 'hf'")
