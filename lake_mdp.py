from __future__ import annotations
from typing import Iterable, List, Tuple, Dict

from mdp import MDP, State, Action

UP, RIGHT, DOWN, LEFT, ABSORB = "UP", "RIGHT", "DOWN", "LEFT", "⊥"


class LakeMDP(MDP):
    """
    Grid map (matrix of single-character strings), e.g.:
      [
        ['S','F','F','F'],
        ['F','H','F','F'],
        ['F','F','F','F'],
        ['H','F','F','G'],
      ]

    Rewards are *state-entry* rewards. After entering H or G, the next state is
    the absorbing state ⊥ with only legal action ⊥ and 0 reward forever.
    """

    def __init__(self, grid: Iterable[Iterable[str]]):
        self.grid: List[List[str]] = [list(row) for row in grid]
        self.m, self.n = len(self.grid), len(self.grid[0])

        starts = [
            ((i, j), "S")
            for i in range(self.m)
            for j in range(self.n)
            if self.grid[i][j] == "S"
        ]
        assert len(starts) == 1, "There must be exactly one start state 'S'."
        self._start = starts[0]

        # rewards on entry to cells
        self._reward: Dict[str, float] = {
            "S": 0.0,
            "F": -0.1,
            "H": -1.0,
            "G": 1.0,
            "⊥": 0.0,
        }

        # probabilities of movement
        self._p_movement = {
            "f": 0.8,
            "l": 0.1,
            "r": 0.1,
        }

    # --- MDP interface -----------------------------------------------------
    def start_state(self) -> State:
        return self._start

    def actions(self, s: State) -> Iterable[Action]:
        if self.is_terminal(s) or s[0] == ABSORB:
            return (ABSORB,)
        return (UP, RIGHT, DOWN, LEFT)

    def reward(self, s: State) -> float:
        _, symbol = s
        return self._reward[symbol]

    def is_terminal(self, s: State) -> bool:
        return s[1] in ("H", "G", ABSORB)

    # --- helpers -----------------------------------------------------------
    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.m and 0 <= j < self.n

    def _move(self, s: State, a: Action) -> tuple[int, int]:
        pos, symbol = s
        if symbol == ABSORB:
            return (ABSORB, ABSORB)
        di, dj = 0, 0
        if a == UP:
            di, dj = -1, 0
        elif a == RIGHT:
            di, dj = 0, +1
        elif a == DOWN:
            di, dj = +1, 0
        elif a == LEFT:
            di, dj = 0, -1
        elif a == ABSORB:
            return s
        ni, nj = pos[0] + di, pos[1] + dj
        if not self.in_bounds(ni, nj):
            return s  # bump -> stay
        return ((ni, nj), self.grid[ni][nj])

    def _laterals(self, a: Action) -> tuple[Action, Action]:
        if a in (UP, DOWN):
            return LEFT, RIGHT
        if a in (LEFT, RIGHT):
            return UP, DOWN
        return (ABSORB, ABSORB)

    def transition(self, s: State, a: Action) -> List[Tuple[State, float]]:
        # Absorbing behavior
        if self.is_terminal(s) or a == ABSORB:
            return [((ABSORB, ABSORB), 1.0)]  # no-op from non-absorbing

        ns_main = self._move(s, a)
        left_a, right_a = self._laterals(a)
        ns_left = self._move(s, left_a)
        ns_right = self._move(s, right_a)

        out: Dict[State, float] = {}

        def add(s: State, p: float):
            if self.is_terminal(s):
                out[s] = out.get(s, 0.0) + p
            else:
                out[s] = out.get(s, 0.0) + p

        add(ns_main, self._p_movement["f"])
        add(ns_left, self._p_movement["l"])
        add(ns_right, self._p_movement["r"])
        return list(out.items())
