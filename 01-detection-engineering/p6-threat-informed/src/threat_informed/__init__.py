"""Threat-informed detection-as-code: validate Sigma rules + map to ATT&CK.

Public API:
    set_seed, get_device       -- reproducibility helpers
    load_rules, parse_rule     -- load + validate Sigma YAML rules
    SigmaRule                  -- parsed/validated rule model
    build_coverage             -- technique -> rule-count map
    coverage_grid, summarize   -- ATT&CK matrix + headline metrics
    TACTICS, TECHNIQUES        -- the embedded offline ATT&CK catalog
"""

from .attack import TACTICS, TECHNIQUES, tactics_for_technique, technique_name
from .coverage import build_coverage, coverage_grid, summarize
from .loader import SigmaRule, load_rules, load_yaml, parse_rule
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "load_rules",
    "parse_rule",
    "load_yaml",
    "SigmaRule",
    "build_coverage",
    "coverage_grid",
    "summarize",
    "TACTICS",
    "TECHNIQUES",
    "tactics_for_technique",
    "technique_name",
]
