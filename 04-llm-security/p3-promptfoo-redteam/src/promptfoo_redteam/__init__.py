"""promptfoo-style LLM red-team harness (offline pure-python default).

Two ways to run the same OWASP-LLM red-team:
  * DEFAULT (offline): a pure-python harness here that runs a probe library
    against the local mock RAG target in ../p4-vulnerable-rag and emits
    metrics.json + a chart. No Node, no network, no keys.
  * OPTIONAL (eval-as-code): promptfoo's `redteam` (owasp:llm preset) driven by
    promptfoo/promptfooconfig.yaml, requiring node/npx -- see the README.

Public API:
    set_seed, get_device        -- reproducibility helpers
    Probe, build_probes         -- the OWASP-LLM probe library
    OWASP_CATEGORIES            -- category id -> label map
    grade, GradeResult          -- the success oracle (leak/jailbreak detector)
    load_target                 -- resolve the p4 RAG (or a built-in mock)
    run_probes, summarize       -- run the harness + aggregate metrics
    ProbeRun, EchoMockTarget    -- result record + offline fallback target
"""

from .grader import GradeResult, grade
from .harness import EchoMockTarget, ProbeRun, load_target, run_probes, summarize
from .probes import OWASP_CATEGORIES, Probe, build_probes
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "Probe",
    "build_probes",
    "OWASP_CATEGORIES",
    "grade",
    "GradeResult",
    "load_target",
    "run_probes",
    "summarize",
    "ProbeRun",
    "EchoMockTarget",
]
