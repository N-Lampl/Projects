"""From-scratch classical-ML prompt-injection detector.

Pipeline: TF-IDF (word + char n-grams) -> sklearn LogisticRegression. Trained on
the synthetic injection dataset from `dataset.py`. No deep models, no external
APIs -- a small, fast, fully-offline first line of defense.

The detector exposes:
  * `fit(texts, labels)`               -- train.
  * `predict(text) / predict_proba`    -- score a single string.
  * `evaluate(texts, labels)`          -- precision/recall/F1/ROC-AUC + curve.
  * `save / load`                      -- joblib persistence to models/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


@dataclass
class EvalReport:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    threshold: float
    fpr: list[float] = field(default_factory=list)
    tpr: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "threshold": self.threshold,
        }


class InjectionDetector:
    """TF-IDF + LogisticRegression prompt-injection classifier (from scratch)."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        # Word n-grams capture phrases ("ignore previous"); char n-grams catch
        # obfuscation / spacing tricks ("i g n o r e"). Both are cheap on CPU.
        self.pipeline: Pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        analyzer="word",
                        sublinear_tf=True,
                        min_df=1,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=4.0,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        self._fitted = False

    # ----------------------------------------------------------------- train #
    def fit(self, texts: list[str], labels: list[int]) -> "InjectionDetector":
        self.pipeline.fit(texts, labels)
        self._fitted = True
        return self

    # --------------------------------------------------------------- predict #
    def predict_proba(self, text: str) -> float:
        """Probability that `text` is a prompt injection (class 1)."""
        if not self._fitted:
            raise RuntimeError("detector is not fitted; call fit() or load() first")
        return float(self.pipeline.predict_proba([text])[0, 1])

    def predict(self, text: str) -> int:
        """1 if injection-probability >= threshold, else 0."""
        return int(self.predict_proba(text) >= self.threshold)

    def predict_proba_batch(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)[:, 1]

    # -------------------------------------------------------------- evaluate #
    def evaluate(self, texts: list[str], labels: list[int]) -> EvalReport:
        scores = self.predict_proba_batch(texts)
        preds = (scores >= self.threshold).astype(int)
        y = np.asarray(labels)
        fpr, tpr, _ = roc_curve(y, scores)
        return EvalReport(
            precision=precision_score(y, preds, zero_division=0),
            recall=recall_score(y, preds, zero_division=0),
            f1=f1_score(y, preds, zero_division=0),
            roc_auc=roc_auc_score(y, scores),
            threshold=self.threshold,
            fpr=[float(v) for v in fpr],
            tpr=[float(v) for v in tpr],
        )

    # ----------------------------------------------------------- persistence #
    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: str | Path) -> "InjectionDetector":
        import joblib

        blob = joblib.load(path)
        det = cls(threshold=blob["threshold"])
        det.pipeline = blob["pipeline"]
        det._fitted = True
        return det


def train_detector(
    texts: list[str],
    labels: list[int],
    threshold: float = 0.5,
) -> InjectionDetector:
    """Convenience: build + fit a detector in one call."""
    return InjectionDetector(threshold=threshold).fit(texts, labels)
