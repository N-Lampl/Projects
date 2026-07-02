"""A self-contained gridworld MDP in numpy — no gym, no external RL libraries.

An ``N x N`` grid. The agent starts at ``start`` and must reach ``goal``. Each
step costs ``step_penalty`` (so shorter paths are better), reaching the goal pays
``goal_reward`` and ends the episode. Actions are the four cardinal moves; moves
into a wall keep the agent in place (and still cost a step). The state handed to a
policy is a small feature vector, not a bare index, so a torch MLP can generalise.
"""

from __future__ import annotations

import numpy as np

# up, down, left, right
_MOVES = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])


class GridWorld:
    """Deterministic NxN gridworld with a per-step penalty and a terminal reward."""

    n_actions = 4

    def __init__(
        self,
        size: int = 5,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
        step_penalty: float = 0.05,
        goal_reward: float = 1.0,
        max_steps: int = 50,
    ) -> None:
        self.size = size
        self.start = start
        self.goal = goal if goal is not None else (size - 1, size - 1)
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.state_dim = 4  # (row, col, drow_to_goal, dcol_to_goal) normalised
        self._pos = np.array(self.start, dtype=np.int64)
        self._t = 0

    def _features(self) -> np.ndarray:
        r, c = self._pos
        gr, gc = self.goal
        s = self.size - 1 if self.size > 1 else 1
        return np.array(
            [r / s, c / s, (gr - r) / s, (gc - c) / s],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        """Return the start-state feature vector."""
        self._pos = np.array(self.start, dtype=np.int64)
        self._t = 0
        return self._features()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Apply an action; return (next_state_features, reward, done)."""
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action}")
        self._t += 1
        nxt = self._pos + _MOVES[action]
        # Walls: stay put on an out-of-bounds move.
        if 0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size:
            self._pos = nxt

        at_goal = bool(self._pos[0] == self.goal[0] and self._pos[1] == self.goal[1])
        reward = -self.step_penalty
        done = False
        if at_goal:
            reward += self.goal_reward
            done = True
        elif self._t >= self.max_steps:
            done = True
        return self._features(), reward, done
