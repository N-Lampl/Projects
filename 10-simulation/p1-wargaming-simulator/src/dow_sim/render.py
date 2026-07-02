"""Hex-board rendering shared by the static figures and the Streamlit visualizer.

Draws pointy-top hexes colored by terrain, the objective marker, and unit tokens colored by side
with an HP label. A single ``render_board`` takes primitives (not live objects) so the dashboard
can call it with a recorded snapshot just as easily as with a live state.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

from .hexgrid import Hex
from .state import GameState
from .terrain import Terrain

_TERRAIN_COLOR = {
    Terrain.CLEAR: "#e8e4d0",
    Terrain.FOREST: "#6b8e5a",
    Terrain.HILL: "#c9a26b",
    Terrain.URBAN: "#9a9a9a",
    Terrain.WATER: "#5b8bb0",
}
_SIDE_COLOR = {"BLUE": "#2a5db0", "RED": "#b03a3a"}
_KIND_MARKER = {"infantry": "I", "armor": "A", "artillery": "R", "air": "V"}


def _pixel(q: int, r: int, size: float = 1.0) -> tuple[float, float]:
    x = size * math.sqrt(3) * (q + r / 2)
    y = -size * 1.5 * r  # invert so r increases downward
    return x, y


def render_board(
    tiles: dict[Hex, Terrain],
    units: list[tuple[str, str, str, float, int, int]],
    objective: Hex | None = None,
    title: str = "",
    size: float = 1.0,
):
    """Render a board. ``units`` are ``(uid, side, kind, hp, q, r)`` tuples. Returns a Figure."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for h, terr in tiles.items():
        x, y = _pixel(h.q, h.r, size)
        ax.add_patch(
            RegularPolygon(
                (x, y), numVertices=6, radius=size, orientation=0,
                facecolor=_TERRAIN_COLOR[terr], edgecolor="#3d3d3d", linewidth=0.5,
            )
        )
    if objective is not None:
        ox, oy = _pixel(objective.q, objective.r, size)
        ax.add_patch(
            RegularPolygon(
                (ox, oy), numVertices=6, radius=size, orientation=0,
                facecolor="none", edgecolor="#e6c200", linewidth=3.0,
            )
        )
    for _uid, side, kind, hp, q, r in units:
        x, y = _pixel(q, r, size)
        ax.add_patch(plt.Circle((x, y), size * 0.55, color=_SIDE_COLOR[side], zorder=3))
        ax.text(x, y, _KIND_MARKER.get(kind, "?"), ha="center", va="center",
                color="white", fontweight="bold", fontsize=10, zorder=4)
        ax.text(x, y - size * 0.75, f"{hp:.0f}", ha="center", va="center",
                color="black", fontsize=7, zorder=4)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def render_state(state: GameState, title: str = ""):
    """Convenience wrapper: render a live :class:`GameState`."""
    units = [
        (u.uid, u.side, u.kind.value, u.hp, u.pos.q, u.pos.r)
        for u in state.units.values()
        if u.alive
    ]
    return render_board(state.tiles, units, state.objective, title)
