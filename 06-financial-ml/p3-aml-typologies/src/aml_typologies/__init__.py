"""AML typology detection on a synthetic transaction graph.

Public API:
    set_seed, get_device          -- reproducibility helpers
    generate_aml_graph, AMLGraph  -- synthetic graph with planted typologies
    build_features                -- per-account graph feature matrix
    cycle_members, HAVE_NETWORKX  -- cycle detection (networkx or pure-python)
    score_isolation_forest        -- unsupervised anomaly detector
    score_rules_rf                -- rules + class-weighted RandomForest hybrid
    evaluate                      -- PR-AUC / ROC-AUC / precision@k / recall@FPR-budget
"""

from .detect import (
    evaluate,
    feature_importances,
    pr_curve,
    precision_at_k,
    recall_at_fpr_budget,
    score_isolation_forest,
    score_rules_rf,
)
from .features import FEATURE_NAMES, HAVE_NETWORKX, build_features, cycle_members
from .graph import REPORTING_THRESHOLD, AMLGraph, generate_aml_graph
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "generate_aml_graph",
    "AMLGraph",
    "REPORTING_THRESHOLD",
    "build_features",
    "cycle_members",
    "FEATURE_NAMES",
    "HAVE_NETWORKX",
    "score_isolation_forest",
    "score_rules_rf",
    "feature_importances",
    "evaluate",
    "precision_at_k",
    "recall_at_fpr_budget",
    "pr_curve",
]
