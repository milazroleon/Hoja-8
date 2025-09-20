# policies.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from mdp import MDP, State, Action


class Policy(ABC):
    """Callable policy base: policy(state, rng) -> action."""

    def __init__(self, mdp: MDP, rng: np.random.Generator):
        self.mdp = mdp
        self.rng = rng

    def __call__(self, s: State) -> Action:
        return self._decision(s)

    @abstractmethod
    def _decision(self, s: State) -> Action:
        """Children implement their decision rule here."""
        raise NotImplementedError
