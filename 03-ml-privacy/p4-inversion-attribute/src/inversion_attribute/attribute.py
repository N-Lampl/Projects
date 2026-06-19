"""Attribute inference on a synthetic tabular dataset (scikit-learn only).

Threat model: a "model release" exposes a classifier trained to predict some
benign label `y` from features that include a *sensitive* attribute S (e.g. a
protected group). An adversary knows the other (non-sensitive) features of a
target record AND the model's prediction, but NOT the sensitive attribute. Can
they infer S?

Attack (Fredrikson-style attribute inference): for each candidate value of S,
plug it into the released model together with the known features, and pick the
value that best explains the observed prediction. We compare this against a
*marginal baseline* that just guesses the most common value of S -- the lift
over the baseline is the privacy leakage.

Everything here is synthetic and generated locally; no real data, no network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SENSITIVE_VALUES = (0, 1)  # binary sensitive attribute


@dataclass
class AttributeData:
    X: np.ndarray  # full feature matrix, sensitive attr in column `s_idx`
    y: np.ndarray  # benign target label
    s_idx: int  # index of the sensitive column
    feature_names: list[str]


def make_attribute_dataset(
    n: int = 3000,
    n_other: int = 5,
    s_signal: float = 1.6,
    seed: int = 42,
) -> AttributeData:
    """Synthetic dataset where the sensitive attribute S genuinely influences y.

    `s_signal` controls how strongly S drives the label: large -> the released
    model leaks more about S (the whole point of the demo).
    """
    rng = np.random.default_rng(seed)
    other = rng.normal(0, 1, size=(n, n_other))
    s = rng.integers(0, 2, size=n).astype(float)  # sensitive attribute
    w_other = rng.normal(0, 1, size=n_other)
    logit = other @ w_other + s_signal * (s - 0.5) + rng.normal(0, 0.5, size=n)
    y = (logit > 0).astype(int)

    # assemble feature matrix with the sensitive column appended last
    X = np.concatenate([other, s[:, None]], axis=1)
    s_idx = n_other
    names = [f"x{i}" for i in range(n_other)] + ["sensitive"]
    return AttributeData(X=X, y=y, s_idx=s_idx, feature_names=names)


def train_target(
    data: AttributeData, model: str = "logreg", seed: int = 42
) -> tuple[object, np.ndarray, np.ndarray]:
    """Train the released model on (features incl. sensitive) -> y.

    Returns (fitted_model, X_test, y_test). The test split is the population the
    adversary attacks.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.X, data.y, test_size=0.4, random_state=seed, stratify=data.y
    )
    if model == "rf":
        clf = RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=1)
    else:
        clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(X_tr, y_tr)
    return clf, X_te, y_te


def infer_sensitive(
    clf: object,
    X: np.ndarray,
    s_idx: int,
    s_prior: np.ndarray,
) -> np.ndarray:
    """Infer the sensitive attribute for each row of X.

    For each candidate value v of S, set column s_idx = v and ask the model for
    P(y=1). The adversary observes the *true* model output for the real record;
    here we use the model's own predicted probability on the real (sensitive)
    value as the "observed" output, then pick the v whose induced prediction is
    closest -- weighted by the marginal prior P(S=v). This is the standard MAP
    attribute-inference estimator.
    """
    proba = getattr(clf, "predict_proba")
    observed = proba(X)[:, 1]  # the released prediction for the real record

    best_v = np.zeros(len(X), dtype=int)
    best_score = np.full(len(X), -np.inf)
    for vi, v in enumerate(SENSITIVE_VALUES):
        Xv = X.copy()
        Xv[:, s_idx] = v
        pv = proba(Xv)[:, 1]
        # likelihood that this candidate reproduces the observed output * prior
        score = -np.abs(pv - observed) + np.log(s_prior[vi] + 1e-9)
        upd = score > best_score
        best_v[upd] = v
        best_score[upd] = score[upd]
    return best_v


def run_attribute_inference(
    data: AttributeData, clf: object, X_test: np.ndarray
) -> dict[str, float]:
    """Run the attack on the test population and score it vs. the baseline."""
    s_true = X_test[:, data.s_idx].astype(int)
    # marginal prior over S estimated from the (public) test features
    prior = np.array([(s_true == v).mean() for v in SENSITIVE_VALUES])
    baseline_value = int(SENSITIVE_VALUES[int(prior.argmax())])

    s_pred = infer_sensitive(clf, X_test, data.s_idx, prior)

    attack_acc = float((s_pred == s_true).mean())
    baseline_acc = float((s_true == baseline_value).mean())
    lift = attack_acc - baseline_acc
    # precision/recall for S=1 (the minority-leakage view)
    tp = int(((s_pred == 1) & (s_true == 1)).sum())
    fp = int(((s_pred == 1) & (s_true == 0)).sum())
    fn = int(((s_pred == 0) & (s_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "attack_accuracy": attack_acc,
        "baseline_accuracy": baseline_acc,
        "lift_over_baseline": lift,
        "precision_s1": precision,
        "recall_s1": recall,
        "n_test": int(len(s_true)),
    }
