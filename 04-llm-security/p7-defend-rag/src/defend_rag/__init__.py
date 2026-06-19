"""Layered defenses for the p4 vulnerable RAG -- the FLAGSHIP defense project.

Public API:
    set_seed, get_device, load_target   -- reproducibility + target loader
    generate_dataset, InjectionDataset  -- synthetic injection corpus (offline)
    InjectionDetector, train_detector    -- from-scratch TF-IDF + LogReg detector
    DefendedRAG, GuardDecision           -- four-layer guardrail wrapper
    redact_secrets, harden_prompt        -- individual defense primitives
    build_attacks, run_battery, asr      -- the p5-style attack replay harness
"""

from .attacks import (
    Attack,
    AttackOutcome,
    asr,
    asr_by_family,
    attack_succeeded,
    build_attacks,
    build_undefended_target,
    run_battery,
)
from .dataset import InjectionDataset, generate_dataset
from .detector import EvalReport, InjectionDetector, train_detector
from .guardrails import (
    HARDENED_SYSTEM_PROMPT,
    DefendedRAG,
    DefendedResult,
    GuardDecision,
    harden_prompt,
    redact_secrets,
)
from .utils import get_device, load_target, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "load_target",
    "generate_dataset",
    "InjectionDataset",
    "InjectionDetector",
    "train_detector",
    "EvalReport",
    "DefendedRAG",
    "DefendedResult",
    "GuardDecision",
    "redact_secrets",
    "harden_prompt",
    "HARDENED_SYSTEM_PROMPT",
    "Attack",
    "AttackOutcome",
    "build_attacks",
    "build_undefended_target",
    "run_battery",
    "asr",
    "asr_by_family",
    "attack_succeeded",
]
