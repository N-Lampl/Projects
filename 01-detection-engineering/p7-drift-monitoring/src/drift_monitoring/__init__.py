"""Concept-drift / model monitoring as a security control.

Simulate a tabular detector's input stream, inject distribution drift, and score
each monitoring window with PSI + KS drift metrics against alert thresholds.

Public API:
    set_seed, get_device          -- reproducibility helpers
    psi, ks_statistic, ks_pvalue  -- the two drift statistics
    psi_severity                  -- PSI -> {stable, moderate, major}
    StreamConfig, generate_stream -- the synthetic drifting stream
    AlertThresholds               -- PSI/KS alert thresholds
    score_feature, evaluate_window, run_monitor, first_alert_window, psi_matrix
"""

from .metrics import ks_pvalue, ks_statistic, psi, psi_severity
from .monitor import (
    AlertThresholds,
    evaluate_window,
    first_alert_window,
    psi_matrix,
    run_monitor,
    score_feature,
)
from .stream import FEATURE_NAMES, StreamConfig, generate_stream
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "psi",
    "ks_statistic",
    "ks_pvalue",
    "psi_severity",
    "StreamConfig",
    "generate_stream",
    "FEATURE_NAMES",
    "AlertThresholds",
    "score_feature",
    "evaluate_window",
    "run_monitor",
    "first_alert_window",
    "psi_matrix",
]
