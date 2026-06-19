"""Differentiable surrogate model for gradient-based evasion of a tree IDS.

The deployed IDS is a RandomForest (the shared ``ids_pipeline`` default), which
is **non-differentiable** -- there is no input gradient to follow. Real-world
evasion handles this with a *substitute model* (Papernot et al., 2017): train a
smooth, differentiable model to mimic the target's decisions, attack the
substitute with a gradient method, and transfer the resulting examples to the
real target.

Here the surrogate is a plain logistic regression on the *raw numeric features*
(we keep categoricals fixed, so they are not part of the attack surface). It is
trained on the target's hard predictions over the training flows -- i.e. it
learns the target's decision boundary, not the ground truth. We expose its
weights so the attack module can compute closed-form input gradients without any
deep-learning dependency: for logistic regression,

    p(attack | x) = sigmoid(w . x_scaled + b)
    d loss / d x  = (p - y_target) * w / scale     (chain rule through the scaler)

which is the gradient a hand-rolled FGSM steps along.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GradientSurrogate:
    """Logistic-regression substitute exposing raw-space input gradients.

    ``w`` and ``b`` are the logistic coefficients in *scaled* space; ``mean`` and
    ``scale`` are the StandardScaler statistics so we can map gradients back to
    raw feature space (the space the constraints live in).
    """

    w: np.ndarray       # (n_features,)
    b: float
    mean: np.ndarray    # (n_features,) scaler mean (fit on TRAIN only)
    scale: np.ndarray   # (n_features,) scaler std

    def _scale(self, x_raw: np.ndarray) -> np.ndarray:
        return (x_raw - self.mean) / self.scale

    def proba(self, x_raw: np.ndarray) -> np.ndarray:
        """P(attack | x) under the surrogate, taking RAW features."""
        z = self._scale(x_raw) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def loss_gradient_raw(self, x_raw: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        """Gradient of BCE loss w.r.t. RAW input features.

        For an evasion attack we want to *increase* the model's loss on the true
        label (= push attack flows toward the benign side). With BCE and a
        logistic head the gradient w.r.t. scaled input is ``(p - y) * w``;
        dividing by ``scale`` lifts it into raw feature space (chain rule).
        """
        p = self.proba(x_raw)                       # (n,)
        grad_scaled = (p - y_target)[:, None] * self.w[None, :]  # (n, d)
        return grad_scaled / self.scale[None, :]


def fit_surrogate(
    target_pipeline,
    X_train_raw: np.ndarray,
    numeric_features: list[str],
    *,
    seed: int = 42,
) -> GradientSurrogate:
    """Train a logistic-regression substitute that mimics ``target_pipeline``.

    Labels are the TARGET's hard predictions on the training flows (a black-box
    substitute: we query the deployed model, we do not peek at ground truth).
    The substitute is fit on the same raw numeric features the attack perturbs.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Query the target for its labels on the training flows (black-box).
    target_labels = _target_predict(target_pipeline, X_train_raw, numeric_features)

    scaler = StandardScaler().fit(X_train_raw)
    Xs = scaler.transform(X_train_raw)

    # If the target labels happen to be single-class on this slice, fall back to
    # a tiny jitter so LogisticRegression still fits a (degenerate) boundary.
    if len(np.unique(target_labels)) < 2:
        target_labels = target_labels.copy()
        target_labels[0] = 1 - target_labels[0]

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(Xs, target_labels)

    return GradientSurrogate(
        w=clf.coef_.ravel().astype(np.float64),
        b=float(clf.intercept_[0]),
        mean=scaler.mean_.astype(np.float64),
        scale=scaler.scale_.astype(np.float64),
    )


def _target_predict(target_pipeline, X_raw: np.ndarray, numeric_features: list[str]) -> np.ndarray:
    """Query the deployed pipeline for hard labels on raw numeric flows.

    The pipeline expects a DataFrame with both numeric AND categorical columns.
    Because the attack only moves numeric features, we hold categoricals at a
    fixed reference value so the substitute learns the boundary in the subspace
    the attack actually explores.
    """
    df = _raw_to_frame(target_pipeline, X_raw, numeric_features)
    return np.asarray(target_pipeline.predict(df)).astype(int)


def _raw_to_frame(target_pipeline, X_raw: np.ndarray, numeric_features: list[str]):
    """Build a model-ready DataFrame from a raw numeric matrix.

    Categorical columns are filled with a constant reference category (the most
    common training value would be ideal; a stable default keeps it simple and
    deterministic). The pipeline's OneHotEncoder handles them identically for
    clean and adversarial rows, so they never affect the *delta* in the score.
    """
    import pandas as pd

    pre = target_pipeline.named_steps["preprocess"]
    cat_features = [c for name, _, cols in pre.transformers_ if name == "cat" for c in cols]

    data = {c: X_raw[:, i] for i, c in enumerate(numeric_features)}
    # Stable reference categories matching the synthetic schema.
    cat_defaults = {"protocol_type": "tcp", "service": "http", "flag": "SF"}
    for c in cat_features:
        data[c] = [cat_defaults.get(c, "other")] * len(X_raw)
    return pd.DataFrame(data)
