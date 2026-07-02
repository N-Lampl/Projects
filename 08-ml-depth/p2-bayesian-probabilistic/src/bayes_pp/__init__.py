"""Bayesian hierarchical modeling with a from-scratch numpy Gibbs sampler (CPU)."""

from __future__ import annotations

from .data import HierDataset, make_hierarchical
from .experiment import (
    calibration_curve,
    convergence,
    fit_dataset,
    shrinkage_report,
    shrinkage_study,
)
from .inference import (
    complete_pooling,
    credible_interval,
    no_pooling,
    posterior_predictive_check,
    posterior_summary,
    rhat,
    rmse,
)
from .model import GibbsPosterior, gibbs_sampler
from .utils import get_device, set_seed

__all__ = [
    "GibbsPosterior",
    "HierDataset",
    "calibration_curve",
    "complete_pooling",
    "convergence",
    "credible_interval",
    "fit_dataset",
    "gibbs_sampler",
    "get_device",
    "make_hierarchical",
    "no_pooling",
    "posterior_predictive_check",
    "posterior_summary",
    "rhat",
    "rmse",
    "set_seed",
    "shrinkage_report",
    "shrinkage_study",
]
