"""Axial hex-coordinate geometry.

Uses axial coordinates ``(q, r)`` with an implied cube coordinate ``s = -q - r``. Hexes are
frozen dataclasses so they are hashable and usable as dict keys for the terrain map. Reference:
Red Blob Games, "Hexagonal Grids".
"""

from __future__ import annotations

from dataclasses import dataclass

# Axial directions for the six neighbours of a hex (pointy-top layout).
_DIRECTIONS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


@dataclass(frozen=True, slots=True)
class Hex:
    q: int
    r: int

    @property
    def s(self) -> int:
        return -self.q - self.r

    def __add__(self, other: Hex) -> Hex:
        return Hex(self.q + other.q, self.r + other.r)


def neighbors(h: Hex) -> list[Hex]:
    """The six adjacent hexes."""
    return [Hex(h.q + dq, h.r + dr) for dq, dr in _DIRECTIONS]


def distance(a: Hex, b: Hex) -> int:
    """Hex (cube) distance: the number of steps between two hexes."""
    return (abs(a.q - b.q) + abs(a.r - b.r) + abs(a.s - b.s)) // 2


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _cube_round(q: float, r: float, s: float) -> Hex:
    rq, rr, rs = round(q), round(r), round(s)
    dq, dr, ds = abs(rq - q), abs(rr - r), abs(rs - s)
    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    return Hex(int(rq), int(rr))


def line(a: Hex, b: Hex) -> list[Hex]:
    """The hexes crossed by a straight line from ``a`` to ``b`` (inclusive), for line-of-sight."""
    n = distance(a, b)
    if n == 0:
        return [a]
    out: list[Hex] = []
    for i in range(n + 1):
        t = i / n
        out.append(
            _cube_round(
                _lerp(a.q, b.q, t),
                _lerp(a.r, b.r, t),
                _lerp(a.s, b.s, t),
            )
        )
    return out


def within_range(center: Hex, radius: int) -> list[Hex]:
    """All hexes within ``radius`` steps of ``center`` (including the center)."""
    out: list[Hex] = []
    for dq in range(-radius, radius + 1):
        for dr in range(max(-radius, -dq - radius), min(radius, -dq + radius) + 1):
            out.append(Hex(center.q + dq, center.r + dr))
    return out
