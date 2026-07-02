"""Bundled tactical scenarios and their loader.

Scenarios are defined in code (the single source of truth) as plain dicts and built into a fresh
:class:`GameState` on demand, so tests and CI need nothing on disk. The same definitions are
mirrored as human-readable JSON under ``data/scenarios/`` for inspection; :func:`export_json`
regenerates them.
"""

from __future__ import annotations

import json
from pathlib import Path

from .hexgrid import Hex
from .state import GameState
from .terrain import Terrain
from .units import BLUE, RED, UnitKind, make_unit

# Each scenario: a parallelogram hex map (width x height), terrain overrides, unit placements,
# an optional objective hex BLUE must seize and hold, and a turn cap.
SCENARIO_DEFS: dict[str, dict] = {
    "meeting_engagement": {
        "description": "Symmetric infantry+armor clash on mixed terrain. A fair fight lands "
        "near 50% and is how we sanity-check the engine.",
        # Mirror axis between q=3 and q=4; terrain and forces are symmetric under q -> 7-q.
        "width": 8,
        "height": 8,
        "terrain": {
            (3, 3): "forest", (4, 3): "forest", (3, 5): "hill", (4, 5): "hill",
        },
        "blue": [("infantry", 1, 2), ("infantry", 1, 6), ("armor", 1, 4)],
        "red": [("infantry", 6, 2), ("infantry", 6, 6), ("armor", 6, 4)],
        "objective": None,
        "objective_hold_turns": 1,
        "max_turns": 20,
    },
    "seize_the_ridge": {
        "description": "BLUE (attacker) must seize and hold a fortified hill for 3 turns against "
        "a dug-in RED defense with artillery support. Showcases terrain defense and a defensive "
        "policy.",
        "width": 9,
        "height": 8,
        "terrain": {
            (4, 3): "hill", (4, 4): "hill", (5, 3): "hill", (3, 3): "forest", (3, 4): "forest",
            (5, 4): "forest", (4, 2): "forest", (4, 5): "forest",
        },
        "blue": [("armor", 1, 3), ("armor", 1, 5), ("infantry", 1, 2), ("infantry", 1, 6)],
        "red": [
            ("infantry", 4, 3), ("infantry", 4, 4), ("infantry", 5, 3), ("artillery", 7, 4),
        ],
        "objective": (4, 3),
        "objective_hold_turns": 3,
        "max_turns": 18,
    },
    "combined_arms": {
        "description": "BLUE armor+artillery+air assaults entrenched RED across a river with a "
        "single crossing. Exercises every unit type, indirect fire, and a chokepoint.",
        "width": 11,
        "height": 9,
        "terrain": {
            # A river down column q=5, with a bridge (clear) at r=4.
            **{(5, r): "water" for r in range(9) if r != 4},
            (8, 3): "urban", (8, 4): "urban", (8, 5): "urban", (9, 4): "urban", (7, 4): "forest",
        },
        "blue": [("armor", 2, 4), ("artillery", 1, 4), ("air", 2, 6), ("infantry", 2, 2)],
        "red": [("infantry", 8, 3), ("infantry", 8, 5), ("artillery", 9, 4), ("armor", 8, 4)],
        "objective": None,
        "objective_hold_turns": 1,
        "max_turns": 24,
    },
}

_TERRAIN_BY_NAME = {t.value: t for t in Terrain}
_KIND_BY_NAME = {k.value: k for k in UnitKind}


def list_scenarios() -> list[tuple[str, str]]:
    """Return ``(name, description)`` for every bundled scenario."""
    return [(name, d["description"]) for name, d in SCENARIO_DEFS.items()]


def _build_units(specs: list, side: str, start: int) -> dict:
    units = {}
    for i, (kind, q, r) in enumerate(specs):
        uid = f"{side[0]}{start + i}"
        units[uid] = make_unit(uid, side, _KIND_BY_NAME[kind], Hex(q, r))
    return units


def load_scenario(name: str) -> GameState:
    """Build a fresh :class:`GameState` for the named scenario."""
    try:
        d = SCENARIO_DEFS[name]
    except KeyError as exc:
        raise ValueError(f"unknown scenario {name!r}; choices: {sorted(SCENARIO_DEFS)}") from exc

    tiles: dict[Hex, Terrain] = {}
    overrides = {(q, r): _TERRAIN_BY_NAME[t] for (q, r), t in d["terrain"].items()}
    for r in range(d["height"]):
        for q in range(d["width"]):
            tiles[Hex(q, r)] = overrides.get((q, r), Terrain.CLEAR)

    units = {}
    units.update(_build_units(d["blue"], BLUE, 1))
    units.update(_build_units(d["red"], RED, 1))

    obj = Hex(*d["objective"]) if d["objective"] is not None else None
    return GameState(
        tiles=tiles,
        units=units,
        max_turns=d["max_turns"],
        objective=obj,
        objective_hold_turns=d["objective_hold_turns"],
    )


def export_json(out_dir: str | Path) -> list[Path]:
    """Write each scenario definition to ``out_dir`` as JSON (for inspection); returns paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, d in SCENARIO_DEFS.items():
        payload = dict(d)
        payload["terrain"] = [{"q": q, "r": r, "type": t} for (q, r), t in d["terrain"].items()]
        payload["name"] = name
        path = out / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2))
        written.append(path)
    return written
