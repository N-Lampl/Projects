"""adversarial_ids: constrained adversarial evasion + hardening of a tabular IDS.

The CAPSTONE for the detection-engineering track. It reuses the shared
``ids_pipeline`` library (imported BY PATH from ``../shared/ids_pipeline``) to
train a RandomForest NIDS, then:

1. fits a differentiable logistic-regression SUBSTITUTE to the target's
   decisions (the tree has no input gradient);
2. crafts evasions with a hand-rolled, FEATURE-MUTABILITY-CONSTRAINED FGSM
   (numpy/sklearn only -- ART is an optional drop-in);
3. measures the transfer attack-success-rate on the deployed target;
4. HARDENS via adversarial training / a diverse ensemble and re-measures;
5. emits an "IDS Robustness Report Card" + ``metrics.json``.

Quick API::

    from adversarial_ids import (
        set_seed, get_pipeline_api, build_constraints,
        fit_surrogate, craft_adversarial, attack_success_rate,
        adversarially_train, render_report_card,
    )
"""

from .attack import (
    attack_success_rate,
    constrained_fgsm,
    craft_adversarial,
    craft_adversarial_art,
)
from .constraints import (
    IMMUTABLE_FEATURES,
    MUTABLE_FEATURES,
    ConstraintSpec,
    build_constraints,
)
from .harden import (
    adversarially_train,
    build_robust_ensemble,
    refit_surrogate_for,
)
from .report import render_report_card, write_report_card
from .surrogate import GradientSurrogate, fit_surrogate
from .utils import (
    ensure_ids_pipeline_on_path,
    get_device,
    set_seed,
    shared_pipeline_src,
)


def get_pipeline_api():
    """Import and return the shared ``ids_pipeline`` module (loads it by path)."""
    ensure_ids_pipeline_on_path()
    import ids_pipeline

    return ids_pipeline


__all__ = [
    # reproducibility + shared-lib plumbing
    "set_seed",
    "get_device",
    "shared_pipeline_src",
    "ensure_ids_pipeline_on_path",
    "get_pipeline_api",
    # constraints
    "ConstraintSpec",
    "build_constraints",
    "MUTABLE_FEATURES",
    "IMMUTABLE_FEATURES",
    # surrogate + attack
    "GradientSurrogate",
    "fit_surrogate",
    "constrained_fgsm",
    "craft_adversarial",
    "craft_adversarial_art",
    "attack_success_rate",
    # hardening
    "adversarially_train",
    "build_robust_ensemble",
    "refit_surrogate_for",
    # reporting
    "render_report_card",
    "write_report_card",
]
