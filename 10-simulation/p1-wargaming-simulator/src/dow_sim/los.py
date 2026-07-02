"""Line-of-sight between hexes.

Sight is blocked when an *intermediate* hex on the straight line holds LOS-blocking terrain.
The endpoints themselves never block (you can see into and out of a forest you border), so a
unit standing in cover can still be targeted from within range.
"""

from __future__ import annotations

from .hexgrid import Hex, distance, line
from .state import GameState
from .terrain import info


def has_los(state: GameState, a: Hex, b: Hex) -> bool:
    """True if nothing on the line between ``a`` and ``b`` blocks sight."""
    path = line(a, b)
    for h in path[1:-1]:
        if not state.on_board(h):
            continue
        if info(state.terrain_at(h)).blocks_los:
            return False
    return True


def can_target(state: GameState, attacker_pos: Hex, target_pos: Hex, weapon_range: int) -> bool:
    """A target is engageable if it is within weapon range and in line-of-sight."""
    return distance(attacker_pos, target_pos) <= weapon_range and has_los(
        state, attacker_pos, target_pos
    )
