"""Detectors: a LogisticRegression (default) and an entropy-threshold baseline."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .features import FeatureExtractor, extract_stats


class DGAClassifier:
    """char n-gram + stats -> LogisticRegression. The default detector."""

    def __init__(self, C: float = 4.0, max_iter: int = 1000):
        self.fe = FeatureExtractor()
        self.clf = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")

    def fit(self, domains: list[str], y: np.ndarray) -> DGAClassifier:
        x = self.fe.fit_transform(domains)
        self.clf.fit(x, y)
        return self

    def predict_proba(self, domains: list[str]) -> np.ndarray:
        return self.clf.predict_proba(self.fe.transform(domains))[:, 1]

    def predict(self, domains: list[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(domains) >= threshold).astype(int)


class EntropyBaseline:
    """A naive detector: flag a domain as DGA if its label entropy exceeds a
    threshold chosen from the training set. Demonstrates why entropy alone is
    not enough (it misses the dictionary-DGA family)."""

    ENTROPY_COL = 1  # index of 'entropy' in extract_stats

    def __init__(self) -> None:
        self.threshold = 0.0

    def fit(self, domains: list[str], y: np.ndarray) -> EntropyBaseline:
        ent = extract_stats(domains)[:, self.ENTROPY_COL]
        # midpoint between the mean benign and mean DGA entropy
        self.threshold = float((ent[y == 0].mean() + ent[y == 1].mean()) / 2)
        return self

    def score(self, domains: list[str]) -> np.ndarray:
        return extract_stats(domains)[:, self.ENTROPY_COL]

    def predict(self, domains: list[str]) -> np.ndarray:
        return (self.score(domains) >= self.threshold).astype(int)


def evaluate(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict:
    """Bundle the metrics a SOC actually cares about."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(report["accuracy"]),
        "precision_dga": float(report["1"]["precision"]),
        "recall_dga": float(report["1"]["recall"]),
        "f1_dga": float(report["1"]["f1-score"]),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def roc_points(y_true: np.ndarray, y_score: np.ndarray):
    return roc_curve(y_true, y_score)


def pr_points(y_true: np.ndarray, y_score: np.ndarray):
    return precision_recall_curve(y_true, y_score)
