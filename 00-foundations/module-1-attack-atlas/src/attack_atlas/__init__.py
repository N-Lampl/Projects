"""Attack Atlas: map ml-security-portfolio tracks/projects to MITRE ATLAS and
MITRE ATT&CK. Stdlib-only, fully offline.

Public API:
    set_seed, get_device          -- reproducibility helpers (portfolio convention)
    ATLAS_TACTICS, ATLAS_TECHNIQUES, PORTFOLIO_MAP  -- the curated catalog + map
    build_atlas_map               -- track/project -> ATLAS technique document
    build_metrics                 -- coverage counts derived from the map
    render_coverage_chart         -- ASCII bar chart (the committed "figure")
    build_navigator_layer         -- valid ATT&CK Navigator layer JSON
"""

from .atlas import ATLAS_TACTICS, ATLAS_TECHNIQUES, PORTFOLIO_MAP
from .builder import build_atlas_map, build_metrics, render_coverage_chart
from .navigator import build_navigator_layer
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "ATLAS_TACTICS",
    "ATLAS_TECHNIQUES",
    "PORTFOLIO_MAP",
    "build_atlas_map",
    "build_metrics",
    "render_coverage_chart",
    "build_navigator_layer",
]
