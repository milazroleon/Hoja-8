from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple, Union

State = Union[Tuple[int, int], str]  # (row, col) or '⊥'
Action = str  # 'UP','RIGHT','DOWN','LEFT','⊥'


class MDP(ABC):
    """Abstract base class for a finite MDP with state-entry rewards."""

    @abstractmethod
    def start_state(self) -> State: ...

    @abstractmethod
    def actions(self, s: State) -> Iterable[Action]: ...

    @abstractmethod
    def is_terminal(self, s: State) -> bool: ...

    @abstractmethod
    def reward(self, s: State) -> float: ...

    @abstractmethod
    def transition(self, s: State, a: Action) -> List[Tuple[State, float]]: ...

    def step(self, s: State, a: Action, rng) -> Tuple[State, float]:
        if self.is_terminal(s):
            return s, 0.0

        dist = self.transition(s, a)
        total = sum(p for _, p in dist)
        if total <= 0:
            raise ValueError("Transition distribution has zero mass.")
        if abs(total - 1.0) > 1e-8:
            dist = [(ns, p / total) for ns, p in dist]
        probs = [p for _, p in dist]
        idx = rng.choice(len(dist), p=probs)
        ns = dist[idx][0]
        return ns, self.reward(ns)
