"""Defenses: re-train the IDS to resist the constrained evasion attack.

Two complementary hardening strategies, both built on the shared pipeline:

1. **Adversarial training.** Craft constrained-FGSM evasions of the *training*
   attack flows against the current model's surrogate, label them as attacks
   (they ARE attacks -- the label does not change, only the features), and add
   them to the training set. The retrained model learns the decision boundary
   the attacker was exploiting. (Goodfellow 2015; Madry 2018, adapted to a
   constrained tabular setting with a tree model + substitute gradients.)

2. **Ensemble / bagging diversity.** A larger, deeper RandomForest with more
   feature subsampling presents a noisier gradient to the substitute, so
   transferred examples land less reliably. This is the cheap, model-agnostic
   defense a SOC can ship without a full adversarial-training loop.

Both return a fitted ``sklearn.Pipeline`` with the exact same I/O contract as
the baseline, so the report card compares apples to apples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .attack import constrained_fgsm
from .constraints import ConstraintSpec
from .surrogate import GradientSurrogate, fit_surrogate


def _frame_from_raw(numeric_features: list[str], X_raw: np.ndarray, template: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a full model-input frame: perturbed numerics + original categoricals."""
    out = template.copy().reset_index(drop=True)
    for i, c in enumerate(numeric_features):
        out[c] = X_raw[:, i]
    return out


def adversarially_train(
    api,
    dataset,
    cspec: ConstraintSpec,
    surrogate: GradientSurrogate,
    *,
    epsilon: float = 0.3,
    steps: int = 10,
    seed: int = 42,
):
    """Augment TRAIN with constrained adversarial attack flows and refit.

    ``api`` is the shared ``ids_pipeline`` module. We attack only the *attack*
    rows in the training split (the ones an adversary would try to disguise),
    append them with label 1, and rebuild + refit a fresh pipeline.
    """
    num = dataset.numeric_features
    X_train_num = dataset.X_train[num].to_numpy(dtype=np.float64)
    y_train = dataset.y_train.to_numpy()

    atk_idx = np.where(y_train == 1)[0]
    X_atk = X_train_num[atk_idx]
    y_atk = y_train[atk_idx]

    X_adv = constrained_fgsm(
        surrogate, cspec, X_atk, y_atk, epsilon=epsilon, steps=steps
    )

    template = dataset.X_train.iloc[atk_idx].reset_index(drop=True)
    adv_frame = _frame_from_raw(num, X_adv, template)

    X_train_aug = pd.concat([dataset.X_train.reset_index(drop=True), adv_frame], ignore_index=True)
    y_train_aug = pd.concat(
        [dataset.y_train.reset_index(drop=True), pd.Series(np.ones(len(adv_frame), dtype=int))],
        ignore_index=True,
    )

    aug_ds = api.Dataset(
        X_train=X_train_aug,
        X_test=dataset.X_test.reset_index(drop=True),
        y_train=y_train_aug,
        y_test=dataset.y_test.reset_index(drop=True),
        numeric_features=list(dataset.numeric_features),
        categorical_features=list(dataset.categorical_features),
        source=dataset.source + "+adv",
    )
    pipe = api.build_pipeline(aug_ds, seed=seed)
    api.train(pipe, aug_ds)
    return pipe, aug_ds


def build_robust_ensemble(api, dataset, *, seed: int = 42):
    """A deeper, more-diverse RandomForest to blunt gradient transfer.

    Same leak-free preprocessing as the baseline (via the shared pipeline) but a
    larger forest with stronger feature subsampling -> a rougher surface for the
    substitute to imitate. Trained on the clean training split.
    """
    from sklearn.ensemble import RandomForestClassifier

    pipe = api.build_pipeline(dataset, seed=seed)
    pipe.set_params(
        classifier=RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            bootstrap=True,
            random_state=seed,
            n_jobs=-1,
        )
    )
    api.train(pipe, dataset)
    return pipe


def refit_surrogate_for(api, target_pipeline, dataset, *, seed: int = 42) -> GradientSurrogate:
    """Fit a fresh substitute against a (possibly hardened) target pipeline."""
    num = dataset.numeric_features
    X_train_num = dataset.X_train[num].to_numpy(dtype=np.float64)
    return fit_surrogate(target_pipeline, X_train_num, num, seed=seed)
