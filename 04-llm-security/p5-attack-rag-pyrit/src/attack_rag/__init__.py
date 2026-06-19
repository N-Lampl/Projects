"""Attack the authorized lab RAG target (../p4-vulnerable-rag).

Offline by default (deterministic mock target); real garak/PyRIT + LLM optional.
Authorized use only -- see ../../ETHICS.md.

Public API:
    set_seed, get_device              -- reproducibility helpers
    load_target, build_target         -- load the p4 lab target
    detect_sensitive, MockOutboundChannel -- find + simulate-exfiltrate secrets
    AttackResult, ATTACKS, run_suite  -- the attack suite
    attack_success_rate               -- ASR over a result list
"""

from .attacks import (
    ATTACKS,
    INJECTION,
    AttackResult,
    attack_success_rate,
    benign_control,
    run_suite,
)
from .exfil import ExfilEvent, MockOutboundChannel, detect_sensitive
from .target import build_target, clean_corpus, load_target, make_document, planted_api_key
from .utils import get_device, get_int_env, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "get_int_env",
    "load_target",
    "build_target",
    "clean_corpus",
    "make_document",
    "planted_api_key",
    "detect_sensitive",
    "MockOutboundChannel",
    "ExfilEvent",
    "AttackResult",
    "ATTACKS",
    "INJECTION",
    "run_suite",
    "benign_control",
    "attack_success_rate",
]
