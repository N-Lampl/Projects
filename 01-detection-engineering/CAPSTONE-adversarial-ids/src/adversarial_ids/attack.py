"""Constrained adversarial evasion of a tabular IDS.

Two attack engines, one interface (:func:`craft_adversarial`):

1. **Hand-rolled constrained FGSM (DEFAULT, offline).** Uses the differentiable
   :class:`~adversarial_ids.surrogate.GradientSurrogate` to get input gradients,
   then steps along the gradient sign -- but only on MUTABLE features, only in
   allowed directions, projected back into the validity box after every step.
   This is multi-step FGSM (a.k.a. BIM/PGD-L-inf) on the *mutable subspace*.
   Pure numpy + scikit-learn; no attack library required.

2. **IBM ART (OPTIONAL).** If ``adversarial-robustness-toolbox==1.20.1`` is
   installed, :func:`craft_adversarial_art` wraps the surrogate in an ART
   ``ScikitlearnLogisticRegression`` estimator and runs ART's
   ``FastGradientMethod``, then re-applies the SAME feature-mutability +
   consistency projection so ART's free perturbation is forced onto the feasible
   set. Imported lazily inside a try/except so this module imports without ART.

Attack success is measured on the deployed RandomForest target (the substitute
is only a means to a gradient), so what we report is genuine *transfer* ASR.
"""

from __future__ import annotations

import numpy as np

from .constraints import ConstraintSpec
from .surrogate import GradientSurrogate, _raw_to_frame


def constrained_fgsm(
    surrogate: GradientSurrogate,
    cspec: ConstraintSpec,
    X_attack: np.ndarray,
    y_true: np.ndarray,
    *,
    epsilon: float = 0.3,
    steps: int = 10,
    feature_std: np.ndarray | None = None,
) -> np.ndarray:
    """Multi-step constrained FGSM on the mutable feature subspace.

    Parameters
    ----------
    surrogate:
        Differentiable substitute supplying raw-space loss gradients.
    cspec:
        Feature-mutability + consistency constraints (the heart of the attack).
    X_attack, y_true:
        Raw numeric flows to evade and their true labels (attacks = 1).
    epsilon:
        L-inf budget as a FRACTION of each feature's std (per-feature scaling).
    steps:
        Number of gradient steps (1 == single-step FGSM; >1 == BIM/PGD).
    feature_std:
        Per-feature std used to scale the budget; defaults to the surrogate's
        scaler ``scale`` (train-fit, leak-free).
    """
    if feature_std is None:
        feature_std = surrogate.scale
    x0 = X_attack.astype(np.float64).copy()
    x_adv = x0.copy()

    # Raw NIDS features span wildly different scales (bytes vs. rates), so a
    # single scalar epsilon is meaningless -- budget each feature an L-inf move
    # of ``epsilon * feature_std`` and project back into that box every step.
    eps_vec = epsilon * feature_std  # per-feature L-inf budget
    box_lo, box_hi = x0 - eps_vec, x0 + eps_vec
    step_size = eps_vec / max(steps, 1)

    for _ in range(max(steps, 1)):
        grad = surrogate.loss_gradient_raw(x_adv, y_true.astype(np.float64))
        # ascend the loss -> push attack flows toward "benign" prediction
        delta = step_size * np.sign(grad)
        delta = cspec.mask_perturbation(delta)  # mutable + direction constraints
        x_adv = x_adv + delta
        # project: epsilon box, then per-feature validity box
        x_adv = np.clip(x_adv, box_lo, box_hi)
        x_adv = cspec.project(x_adv)
        # hard-restore immutable features (numerical safety net)
        immutable = cspec.mutable_mask == 0.0
        x_adv[:, immutable] = x0[:, immutable]

    return x_adv


def craft_adversarial_art(
    surrogate: GradientSurrogate,
    cspec: ConstraintSpec,
    X_attack: np.ndarray,
    y_true: np.ndarray,
    *,
    epsilon: float = 0.3,
    feature_std: np.ndarray | None = None,
):
    """OPTIONAL IBM ART path: FastGradientMethod on the logistic surrogate.

    Requires ``adversarial-robustness-toolbox==1.20.1``. Imported lazily so the
    module still works offline without ART. After ART crafts its (unconstrained)
    perturbation we re-apply the feature-mutability + validity projection so the
    output respects the SAME constraints as the hand-rolled path.

    Returns ``(x_adv, used_art)`` -- ``used_art=False`` means ART was missing and
    we transparently fell back to :func:`constrained_fgsm`.
    """
    if feature_std is None:
        feature_std = surrogate.scale
    try:
        from art.attacks.evasion import FastGradientMethod
        from art.estimators.classification import ScikitlearnLogisticRegression
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return (
            constrained_fgsm(
                surrogate, cspec, X_attack, y_true,
                epsilon=epsilon, feature_std=feature_std,
            ),
            False,
        )

    # Rebuild an sklearn LogisticRegression in SCALED space from surrogate coeffs
    # so ART can wrap it. (ART needs an estimator object, not raw weights.)
    clf = LogisticRegression()
    clf.classes_ = np.array([0, 1])
    clf.coef_ = surrogate.w.reshape(1, -1)
    clf.intercept_ = np.array([surrogate.b])
    clf.n_features_in_ = surrogate.w.shape[0]

    est = ScikitlearnLogisticRegression(model=clf)
    fgm = FastGradientMethod(estimator=est, eps=float(epsilon), norm=np.inf)

    x_scaled = (X_attack - surrogate.mean) / surrogate.scale
    x_adv_scaled = fgm.generate(x=x_scaled.astype(np.float32))
    x_adv = x_adv_scaled * surrogate.scale + surrogate.mean  # back to raw space

    # Force ART's free perturbation onto our feasible set.
    delta = cspec.mask_perturbation(x_adv - X_attack)
    x_adv = X_attack + delta
    eps_vec = epsilon * feature_std
    x_adv = np.clip(x_adv, X_attack - eps_vec, X_attack + eps_vec)
    x_adv = cspec.project(x_adv)
    immutable = cspec.mutable_mask == 0.0
    x_adv[:, immutable] = X_attack[:, immutable]
    return x_adv, True


def craft_adversarial(
    surrogate: GradientSurrogate,
    cspec: ConstraintSpec,
    X_attack: np.ndarray,
    y_true: np.ndarray,
    *,
    epsilon: float = 0.3,
    steps: int = 10,
    use_art: bool = False,
    feature_std: np.ndarray | None = None,
):
    """Unified entry point. ``use_art=True`` tries ART, else hand-rolled FGSM.

    Returns ``(x_adv, engine)`` where ``engine`` is ``"art"`` or ``"fgsm"``.
    """
    if use_art:
        x_adv, used = craft_adversarial_art(
            surrogate, cspec, X_attack, y_true, epsilon=epsilon, feature_std=feature_std
        )
        return x_adv, ("art" if used else "fgsm")
    x_adv = constrained_fgsm(
        surrogate, cspec, X_attack, y_true,
        epsilon=epsilon, steps=steps, feature_std=feature_std,
    )
    return x_adv, "fgsm"


def attack_success_rate(
    target_pipeline,
    cspec: ConstraintSpec,
    numeric_features: list[str],
    X_attack_raw: np.ndarray,
    X_adv_raw: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Evaluate evasion on the DEPLOYED target (true transfer ASR).

    Success = an originally-detected attack flow that the target now labels
    benign, while STILL satisfying the feature-mutability + validity constraints
    (a flow that cheated the constraints would not survive a real network).
    """
    df_clean = _raw_to_frame(target_pipeline, X_attack_raw, numeric_features)
    df_adv = _raw_to_frame(target_pipeline, X_adv_raw, numeric_features)
    pred_clean = np.asarray(target_pipeline.predict(df_clean)).astype(int)
    pred_adv = np.asarray(target_pipeline.predict(df_adv)).astype(int)

    is_attack = y_true.astype(int) == 1
    detected = is_attack & (pred_clean == 1)            # caught before the attack
    n_detected = int(detected.sum())

    # constraint compliance of the adversarial flows
    consistent = cspec.is_consistent(X_adv_raw)
    immut_ok = cspec.immutable_preserved(X_attack_raw, X_adv_raw)
    feasible = consistent & immut_ok

    evaded = detected & (pred_adv == 0) & feasible      # now slips past, legally
    asr = float(evaded.sum() / n_detected) if n_detected else 0.0

    return {
        "n_attack_flows": int(is_attack.sum()),
        "n_detected_before": n_detected,
        "n_evaded": int(evaded.sum()),
        "attack_success_rate": asr,
        "fraction_feasible": float(feasible.mean()),
        "immutable_preserved_rate": float(immut_ok.mean()),
        "consistency_rate": float(consistent.mean()),
    }
