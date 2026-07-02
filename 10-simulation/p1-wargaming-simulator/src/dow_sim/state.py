"""Game state: the map, both sides' units, and victory conditions."""

from __future__ import annotations

from dataclasses import dataclass, field

from .hexgrid import Hex
from .terrain import Terrain, info
from .units import BLUE, RED, Unit


@dataclass(slots=True)
class GameState:
    tiles: dict[Hex, Terrain]  # sparse; hexes absent from the map are off-board
    units: dict[str, Unit]
    turn: int = 1
    max_turns: int = 20
    objective: Hex | None = None
    # BLUE holds the objective after this many consecutive end-of-turns -> BLUE wins.
    objective_hold_turns: int = 1
    _blue_held: int = field(default=0)

    # --- map queries -------------------------------------------------------
    def terrain_at(self, h: Hex) -> Terrain:
        return self.tiles.get(h, Terrain.CLEAR)

    def on_board(self, h: Hex) -> bool:
        return h in self.tiles

    def passable(self, h: Hex) -> bool:
        return self.on_board(h) and info(self.terrain_at(h)).passable

    def unit_at(self, h: Hex) -> Unit | None:
        for u in self.units.values():
            if u.alive and u.pos == h:
                return u
        return None

    def occupied(self, h: Hex) -> bool:
        return self.unit_at(h) is not None

    # --- roster queries ----------------------------------------------------
    def units_of(self, side: str) -> list[Unit]:
        return [u for u in self.units.values() if u.alive and u.side == side]

    def alive_count(self, side: str) -> int:
        return len(self.units_of(side))

    # --- victory -----------------------------------------------------------
    def winner(self) -> str | None:
        """Return the winning side, "DRAW", or None if the battle is still live."""
        blue = self.alive_count(BLUE)
        red = self.alive_count(RED)
        if red == 0 and blue == 0:
            return "DRAW"
        if red == 0:
            return BLUE
        if blue == 0:
            return RED
        if self.objective is not None and self._blue_held >= self.objective_hold_turns:
            return BLUE
        if self.turn > self.max_turns:
            # Objective scenarios favour the defender on a timeout; otherwise most-units wins.
            if self.objective is not None:
                return RED
            if blue != red:
                return BLUE if blue > red else RED
            return "DRAW"
        return None

    def is_over(self) -> bool:
        return self.winner() is not None

    def record_objective_progress(self) -> None:
        """Call at end of turn: track how long BLUE has held the objective hex."""
        if self.objective is None:
            return
        holder = self.unit_at(self.objective)
        if holder is not None and holder.side == BLUE:
            self._blue_held += 1
        else:
            self._blue_held = 0
