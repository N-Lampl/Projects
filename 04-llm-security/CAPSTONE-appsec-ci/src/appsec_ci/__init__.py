"""appsec_ci: a CI-gated LLM AppSec red-team pipeline (the track-04 capstone).

Glues the OFFLINE red-team harnesses from the sibling projects
(``../p2-garak-scan`` + ``../p3-promptfoo-redteam``) into a single CI gate against
``../p4-vulnerable-rag``, fails the build if attack-success rate (ASR) exceeds a
threshold, trends ASR per OWASP LLM category to a dashboard figure, and emits a
consulting-style threat-model + remediation report.

Public API:
    set_seed, get_device                 -- reproducibility helpers
    OWASP_CATEGORIES, ProbeResult        -- normalised result shape
    load_target, load_defended_target    -- p4 (vuln) / p7 (remediated) targets
    run_garak, run_promptfoo             -- reuse the sibling offline harnesses
    CategoryStat, GateResult, evaluate, aggregate -- the CI gate
    plot_before_after, plot_trend, seed_history, append_history -- the dashboard
    build_report                         -- the markdown threat-model report
"""

from .dashboard import append_history, plot_before_after, plot_trend, seed_history
from .gate import CategoryStat, GateResult, aggregate, evaluate
from .harness import (
    OWASP_CATEGORIES,
    ProbeResult,
    load_defended_target,
    load_target,
    run_garak,
    run_promptfoo,
)
from .report import build_report
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "OWASP_CATEGORIES",
    "ProbeResult",
    "load_target",
    "load_defended_target",
    "run_garak",
    "run_promptfoo",
    "CategoryStat",
    "GateResult",
    "aggregate",
    "evaluate",
    "plot_before_after",
    "plot_trend",
    "seed_history",
    "append_history",
    "build_report",
]
