"""Validate predicted sentiment against the ground-truth 1-5 ``Rating`` column.

The ``Rating`` is a real label, so this is an honest check that the model's
sentiment tracks what reviewers actually scored. The headline (nlptown / stub,
1-5 scale) is exact accuracy, **±1 accuracy**, MAE, and Spearman ρ — car reviews
skew positive, so ±1 and ρ tell the story better than exact match alone.
Binary (SST-2) and 3-class (cardiff) mappings are provided for the other models.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from .sentiment import Prediction


def _round_star(stars: np.ndarray) -> np.ndarray:
    return np.clip(np.round(stars), 1, 5).astype(int)


def validate_against_rating(
    preds: Sequence[Prediction], ratings: Sequence[int], scale: str = "1-5"
) -> dict:
    """Return validation metrics appropriate to the model's ``scale``."""
    stars = np.array([p.star for p in preds], dtype=float)
    scores = np.array([p.score for p in preds], dtype=float)
    ratings = np.asarray(ratings, dtype=int)

    if scale == "1-5":
        pred = _round_star(stars)
        rho = spearmanr(stars, ratings).correlation
        cm = confusion_matrix(ratings, pred, labels=[1, 2, 3, 4, 5])
        return {
            "scale": "1-5",
            "exact_accuracy": round(float(np.mean(pred == ratings)), 4),
            "within_1_accuracy": round(float(np.mean(np.abs(pred - ratings) <= 1)), 4),
            "mae": round(float(np.mean(np.abs(stars - ratings))), 4),
            "spearman": None if np.isnan(rho) else round(float(rho), 4),
            "confusion": cm.tolist(),
            "confusion_labels": [1, 2, 3, 4, 5],
        }

    if scale == "binary":
        mask = ratings != 3
        y_true = (ratings[mask] >= 4).astype(int)
        y_pred = (scores[mask] >= 0.5).astype(int)
        return {
            "scale": "binary",
            "n_dropped_neutral": int((~mask).sum()),
            "accuracy": round(float(np.mean(y_true == y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        }

    if scale == "3-class":
        y_true = np.where(ratings <= 2, 0, np.where(ratings == 3, 1, 2))
        y_pred = np.where(scores < 0.4, 0, np.where(scores <= 0.6, 1, 2))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        return {
            "scale": "3-class",
            "accuracy": round(float(np.mean(y_true == y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "confusion": cm.tolist(),
            "confusion_labels": ["neg", "neu", "pos"],
        }

    raise ValueError(f"unknown scale {scale!r}")
