"""STRIDE threat model of an ML inference service.

Public API:
    set_seed, get_device          -- reproducibility helpers (repo convention)
    build_ml_inference_service    -- the reference data-flow model (dataclasses)
    System, DataFlow, STRIDE      -- model primitives
    analyze, summarize, Threat    -- the deterministic STRIDE rule engine
    mermaid_dfd, render_markdown  -- renderers (Mermaid + docs/threat-model.md)
    try_pytm_model                -- optional pytm path (None if pytm not installed)
"""

from .model import STRIDE, DataFlow, System, build_ml_inference_service
from .report import mermaid_dfd, render_markdown, try_pytm_model
from .threats import Threat, analyze, summarize
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "build_ml_inference_service",
    "System",
    "DataFlow",
    "STRIDE",
    "analyze",
    "summarize",
    "Threat",
    "mermaid_dfd",
    "render_markdown",
    "try_pytm_model",
]
