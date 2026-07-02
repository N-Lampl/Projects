"""Causal inference: estimate a known treatment effect four ways (CPU, from scratch)."""

from __future__ import annotations

from .data import CausalDataset, load_ihdp
from .estimators import (
    ATEResult,
    aipw,
    all_estimators,
    estimate_propensity,
    ipw,
    naive_diff,
    regression_adjustment,
    standardized_mean_diff,
)
from .experiment import balance_table, coverage_study, point_estimates
from .scm import SCM, make_scm
from .utils import get_device, set_seed

__all__ = [
    "ATEResult",
    "CausalDataset",
    "SCM",
    "aipw",
    "all_estimators",
    "balance_table",
    "coverage_study",
    "estimate_propensity",
    "get_device",
    "ipw",
    "load_ihdp",
    "make_scm",
    "naive_diff",
    "point_estimates",
    "regression_adjustment",
    "set_seed",
    "standardized_mean_diff",
]
