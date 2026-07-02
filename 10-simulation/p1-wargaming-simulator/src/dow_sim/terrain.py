"""Terrain types and their tactical modifiers.

Data-driven: a single table maps each terrain to its move cost, defensive multiplier, and
whether it blocks line-of-sight. ``defense_mod`` multiplies the defender's effective defense
in combat, so higher terrain (hills, urban, forest) makes a position harder to take.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Terrain(Enum):
    CLEAR = "clear"
    FOREST = "forest"
    HILL = "hill"
    URBAN = "urban"
    WATER = "water"  # impassable


@dataclass(frozen=True, slots=True)
class TerrainInfo:
    move_cost: int  # move points to enter (impassable if move_cost is None-like/large)
    defense_mod: float  # multiplier on defender's effective defense
    blocks_los: bool
    passable: bool


TERRAIN: dict[Terrain, TerrainInfo] = {
    Terrain.CLEAR: TerrainInfo(move_cost=1, defense_mod=1.0, blocks_los=False, passable=True),
    Terrain.FOREST: TerrainInfo(move_cost=2, defense_mod=1.5, blocks_los=True, passable=True),
    Terrain.HILL: TerrainInfo(move_cost=2, defense_mod=1.75, blocks_los=True, passable=True),
    Terrain.URBAN: TerrainInfo(move_cost=2, defense_mod=2.0, blocks_los=True, passable=True),
    Terrain.WATER: TerrainInfo(move_cost=99, defense_mod=1.0, blocks_los=False, passable=False),
}


def info(t: Terrain) -> TerrainInfo:
    return TERRAIN[t]
