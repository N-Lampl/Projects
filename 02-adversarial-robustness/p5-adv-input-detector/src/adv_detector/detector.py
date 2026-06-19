"""The runtime adversarial-input detector.

Pipeline:
  1. Take clean inputs; make matched FGSM adversarial inputs from the SAME images.
  2. For every input, extract feature-squeezing + statistical features
     (src/adv_detector/squeeze.py).
  3. Train a scikit-learn LogisticRegression (default path; no xgboost/lightgbm)
     on label clean=0 / adversarial=1.
  4. Score new inputs at runtime; threshold the predicted P(adversarial).

We report ROC-AUC, precision/recall, and pick a deployable OPERATING POINT
(the threshold that hits a target false-positive rate on clean traffic).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from .attack import fgsm_perturb
from .squeeze import FEATURE_NAMES, detector_features


@dataclass
class DetectorBundle:
    """A fitted detector: feature scaler + classifier + chosen threshold."""

    scaler: StandardScaler
    clf: LogisticRegression
    threshold: float
    feature_names: list[str]

    def score(self, feats: np.ndarray) -> np.ndarray:
        """Return P(adversarial) for raw feature rows."""
        return self.clf.predict_proba(self.scaler.transform(feats))[:, 1]

    def predict(self, feats: np.ndarray) -> np.ndarray:
        """Return 1 (adversarial) / 0 (clean) at the chosen operating threshold."""
        return (self.score(feats) >= self.threshold).astype(int)


def build_feature_dataset(
    model: nn.Module,
    loader: DataLoader,
    *,
    epsilon: float = 0.2,
    bits: int = 2,
    kernel: int = 3,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) where each clean image contributes one clean + one FGSM row.

    Only images the model classifies CORRECTLY when clean are used, and only FGSM
    examples that actually FLIP the prediction count as adversarial — this is the
    realistic detection problem (catch successful attacks, don't flag failures).
    """
    device = device or torch.device("cpu")
    model.to(device).eval()
    feats: list[np.ndarray] = []
    labels: list[int] = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            clean_pred = model(x).argmax(1)
        correct = clean_pred == y
        if correct.sum() == 0:
            continue
        x_c, y_c = x[correct], y[correct]

        x_adv = fgsm_perturb(model, x_c, y_c, epsilon)
        with torch.no_grad():
            adv_pred = model(x_adv).argmax(1)
        flipped = adv_pred != y_c  # successful attacks only

        f_clean = detector_features(model, x_c, bits=bits, kernel=kernel).cpu().numpy()
        feats.append(f_clean)
        labels.extend([0] * len(f_clean))

        if flipped.sum() > 0:
            f_adv = detector_features(model, x_adv[flipped], bits=bits, kernel=kernel)
            f_adv = f_adv.cpu().numpy()
            feats.append(f_adv)
            labels.extend([1] * len(f_adv))

    X = np.concatenate(feats, axis=0)
    y_arr = np.array(labels, dtype=int)
    return X, y_arr


def pick_threshold_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.05
) -> float:
    """Smallest threshold whose false-positive rate on clean data is <= target_fpr."""
    clean_scores = np.sort(scores[y_true == 0])
    n = clean_scores.size
    if n == 0:
        return 0.5
    # We flag input as adversarial when score >= threshold. To keep the clean
    # false-positive rate <= target_fpr, allow at most k clean scores at/above thr.
    k = int(np.floor(target_fpr * n))  # max clean positives permitted
    if k <= 0:
        # set threshold just above the largest clean score -> 0 false positives
        return float(np.nextafter(clean_scores[-1], np.inf))
    # threshold = the (k)-th largest clean score's successor, so exactly the top
    # (k-1) clean scores stay strictly below it... pick the score at index n-k and
    # nudge above it to exclude ties.
    return float(np.nextafter(clean_scores[n - k], np.inf))


def train_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_fpr: float = 0.05,
    seed: int = 42,
) -> DetectorBundle:
    """Fit scaler + logistic regression and choose an operating threshold."""
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(scaler.transform(X_train), y_train)
    train_scores = clf.predict_proba(scaler.transform(X_train))[:, 1]
    thr = pick_threshold_at_fpr(y_train, train_scores, target_fpr=target_fpr)
    return DetectorBundle(scaler=scaler, clf=clf, threshold=thr, feature_names=list(FEATURE_NAMES))
