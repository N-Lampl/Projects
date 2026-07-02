"""Units and their stat table.

``UnitStats`` is immutable reference data keyed by ``UnitKind``. A live ``Unit`` carries its
mutable battle state (position, hp, whether it has acted). HP is a float so the continuous
Lanchester combat model can erode it smoothly; the CRT model uses whole-point steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .hexgrid import Hex

BLUE = "BLUE"
RED = "RED"


def enemy_of(side: str) -> str:
    return RED if side == BLUE else BLUE


class UnitKind(Enum):
    INFANTRY = "infantry"
    ARMOR = "armor"
    ARTILLERY = "artillery"
    AIR = "air"


@dataclass(frozen=True, slots=True)
class UnitStats:
    attack: float
    defense: float
    move: int  # move points per turn
    rng: int  # weapon range in hexes
    max_hp: float


STATS: dict[UnitKind, UnitStats] = {
    UnitKind.INFANTRY: UnitStats(attack=4.0, defense=6.0, move=2, rng=1, max_hp=10.0),
    UnitKind.ARMOR: UnitStats(attack=8.0, defense=4.0, move=5, rng=2, max_hp=12.0),
    UnitKind.ARTILLERY: UnitStats(attack=10.0, defense=2.0, move=1, rng=5, max_hp=6.0),
    UnitKind.AIR: UnitStats(attack=7.0, defense=3.0, move=8, rng=3, max_hp=8.0),
}


@dataclass(slots=True)
class Unit:
    uid: str
    side: str
    kind: UnitKind
    pos: Hex
    hp: float
    has_moved: bool = False
    has_fired: bool = False

    @property
    def stats(self) -> UnitStats:
        return STATS[self.kind]

    @property
    def alive(self) -> bool:
        return self.hp > 0

    def reset_turn(self) -> None:
        self.has_moved = False
        self.has_fired = False


def make_unit(uid: str, side: str, kind: UnitKind, pos: Hex) -> Unit:
    return Unit(uid=uid, side=side, kind=kind, pos=pos, hp=STATS[kind].max_hp)
