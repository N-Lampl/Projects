"""garak-scan: an offline garak-style LLM red-team scan + report parser.

Default path is fully OFFLINE: a tiny built-in probe set is run against the p4
VulnerableRAG mock (or a self-contained fallback), producing a garak-compatible
``report.jsonl`` that the parser turns into bootstrap-CI attack-success rates and
a bar chart. The same parser also ingests a REAL garak ``report.jsonl`` (Ollama /
API targets) -- see the README.

Public API:
    set_seed, get_device          -- reproducibility helpers
    Probe, BUILTIN_PROBES, get_probes -- the allowlisted probe set + detectors
    get_target, RagTarget, StandaloneMockTarget -- attack targets
    run_scan                      -- run probes -> garak-style report.jsonl
    summarize, load_attempts, bootstrap_ci -- report.jsonl -> metrics
"""

from .probes import (
    BUILTIN_PROBES,
    Probe,
    detect_not_refused,
    detect_pii_leak,
    detect_pwned,
    detect_secret_leak,
    detect_system_prompt_leak,
    get_probes,
)
from .report import bootstrap_ci, load_attempts, summarize
from .scanner import run_scan
from .target import RagTarget, StandaloneMockTarget, get_target
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "Probe",
    "BUILTIN_PROBES",
    "get_probes",
    "detect_pwned",
    "detect_secret_leak",
    "detect_system_prompt_leak",
    "detect_pii_leak",
    "detect_not_refused",
    "get_target",
    "RagTarget",
    "StandaloneMockTarget",
    "run_scan",
    "summarize",
    "load_attempts",
    "bootstrap_ci",
]
