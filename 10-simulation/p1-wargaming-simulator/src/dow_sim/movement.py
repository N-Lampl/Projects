"""Movement: reachable hexes within a unit's move-point budget.

A terrain-weighted Dijkstra expansion over move points. A unit may pass through empty passable
hexes only; enemy- or friendly-occupied hexes block movement and cannot be entered (except the
unit's own starting hex). The result is the set of legal destinations and the cost to reach each.
"""

from __future__ import annotations

import heapq

from .hexgrid import Hex, neighbors
from .state import GameState
from .terrain import info
from .units import Unit


def reachable(state: GameState, unit: Unit) -> dict[Hex, int]:
    """Map each reachable destination hex to the minimum move-point cost to reach it."""
    budget = unit.stats.move
    start = unit.pos
    best: dict[Hex, int] = {start: 0}
    pq: list[tuple[int, int, int]] = [(0, start.q, start.r)]
    while pq:
        cost, q, r = heapq.heappop(pq)
        here = Hex(q, r)
        if cost > best.get(here, budget + 1):
            continue
        for nb in neighbors(here):
            if not state.passable(nb):
                continue
            if nb != start and state.occupied(nb):
                continue
            step = info(state.terrain_at(nb)).move_cost
            new_cost = cost + step
            if new_cost <= budget and new_cost < best.get(nb, budget + 1):
                best[nb] = new_cost
                heapq.heappush(pq, (new_cost, nb.q, nb.r))
    best.pop(start, None)
    return best


def legal_destinations(state: GameState, unit: Unit) -> list[Hex]:
    return list(reachable(state, unit).keys())
