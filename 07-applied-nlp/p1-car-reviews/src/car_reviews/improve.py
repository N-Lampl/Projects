"""Improving the sentiment model, measured against the SAME validation harness.

Two levers, cheap -> heavy:

  1. **Calibration (this module).** The confusion matrix shows the baseline nlptown
     model is *miscalibrated* to this very-positive dataset (86% of reviews are 4-5
     stars): it is more conservative than car owners and systematically under-rates.
     We fit a multinomial logistic model that maps nlptown's 5 class-probabilities to
     the actual ``Rating`` on a TRAIN split, then evaluate on a held-out TEST split.
     Near-zero compute; fixes the systematic bias (exact accuracy, MAE). It re-weights
     the model's own signal, so it can't lift rank-correlation (Spearman) much - for
     that you need new representations.

  2. **Fine-tuning (``finetune.py``).** Learn car-domain representations end-to-end.

Both are scored on the same held-out TEST set (see :func:`make_splits`) so the
before -> after comparison is apples-to-apples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .sentiment import Prediction, _label_from_score

STAR_LABELS = (1, 2, 3, 4, 5)


def build_text(df: pd.DataFrame) -> list[str]:
    """Review title + body (same composition the baseline scores), so every model
    in the comparison sees identical input and the delta is the model, not the text.
    """
    titles = df["Review_Title"].fillna("").astype(str).str.strip()
    reviews = df["Review"].astype(str).str.strip()
    return [f"{t}. {r}" if t else r for t, r in zip(titles, reviews, strict=False)]


def make_splits(
    df: pd.DataFrame, test_size: int = 5000, val_size: int = 2000, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified (by ``Rating``) train/val/test split.

    Same ``df`` + ``seed`` always yields the same TEST set, so the fine-tune script
    and the comparison script evaluate on identical held-out data.
    """
    idx = np.arange(len(df))
    strat = df["Rating"].to_numpy()
    test_n = min(test_size, len(df) // 5)
    train_idx, test_idx = train_test_split(idx, test_size=test_n, stratify=strat, random_state=seed)
    val_n = min(val_size, len(train_idx) // 5)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_n, stratify=strat[train_idx], random_state=seed
    )
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42, by: str = "Rating") -> pd.DataFrame:
    """Down-sample to ~``n`` rows keeping per-``by`` proportions (seeded)."""
    if n >= len(df):
        return df.reset_index(drop=True)
    out = df.groupby(by, group_keys=False, observed=True).sample(
        frac=n / len(df), random_state=seed
    )
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.reset_index(drop=True)


def predictions_from_star_probs(probs: np.ndarray, star_labels=STAR_LABELS) -> list[Prediction]:
    """Turn a 5-class probability matrix into Predictions via the expected star."""
    lab = np.asarray(star_labels, dtype=float)
    stars = probs @ lab
    scores = np.clip((stars - 1) / 4, 0, 1)
    return [
        Prediction(score=float(s), star=float(st), label=_label_from_score(float(s)))
        for st, s in zip(stars, scores, strict=False)
    ]


class Calibrator:
    """Multinomial-logistic recalibration of a 5-class sentiment head to the Rating.

    Learns ``P(Rating | model_class_probs)`` - i.e. it stacks a light classifier on
    top of the frozen sentiment model to undo its domain bias.
    """

    def __init__(self, seed: int = 42):
        self.clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        self.classes_: np.ndarray | None = None

    def fit(self, probs_train: np.ndarray, y_train) -> Calibrator:
        self.clf.fit(probs_train, np.asarray(y_train))
        self.classes_ = self.clf.classes_.astype(float)
        return self

    def predict(self, probs: np.ndarray) -> list[Prediction]:
        cal = self.clf.predict_proba(probs)  # columns ordered by self.classes_
        stars = cal @ self.classes_
        scores = np.clip((stars - 1) / 4, 0, 1)
        return [
            Prediction(score=float(s), star=float(st), label=_label_from_score(float(s)))
            for st, s in zip(stars, scores, strict=False)
        ]
