from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from lake_mdp import ABSORB
from policy import Policy


def enumerate_states(mdp) -> List[object]:
    """
    Deterministically enumerate all states reachable from the start state.

    States are treated as opaque hashables and can be shaped like:
      - ((i, j), ch) for grid cells, where ch ∈ {'S','F','H','G'}
      - (ch, ch) for the absorbing state

    We rely only on the environment's own transitions.
    """
    start = mdp.start_state()
    seen = {start}
    order: List[object] = [start]
    stack = [start]
    while stack:
        s = stack.pop()
        for a in mdp.actions(s):
            for ns, p in mdp.transition(s, a):
                if ns not in seen:
                    seen.add(ns)
                    order.append(ns)
                    stack.append(ns)
    return order


def build_policy_Pr(
    mdp, policy: Policy, states: List[object]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the policy-induced transition matrix P and reward vector r for a fixed policy.

    P[i, j] = Pr(S_{t+1} = states[j] | S_t = states[i], A_t ~ π(S_t))
    r[i]    = reward(states[i]) using the environment's reward convention (on entry).

    Terminal/absorbing states are detected via actions(s) == [ABSORB] and given a self-loop.
    """
    S = len(states)
    index: Dict[object, int] = {s: i for i, s in enumerate(states)}
    P = np.zeros((S, S), dtype=float)

    for s in states:
        i = index[s]

        acts = list(mdp.actions(s))
        if acts == [ABSORB]:
            absorb_idx = index[(ABSORB, ABSORB)]
            P[i, absorb_idx] = 1.0
            continue

        # Optional stochastic policies via action_probs(s)
        probs = None
        ap = getattr(policy, "action_probs", None)
        if callable(ap):
            pa = ap(s)
            if pa is not None:
                # validate once and reuse
                probs = {a: float(pa[a]) for a in pa if pa[a] > 0.0}

        if probs is None:
            a = policy(s)
            for ns, p in mdp.transition(s, a):
                P[i, index[ns]] += float(p)
        else:
            for a, pa in probs.items():
                if pa <= 0.0:
                    continue
                for ns, p in mdp.transition(s, a):
                    P[i, index[ns]] += float(pa) * float(p)

        # Defensive normalization (handles tiny drift)
        row_sum = P[i].sum()
        if row_sum > 0 and abs(row_sum - 1.0) > 1e-8:
            P[i] /= row_sum

    rewards = np.array([mdp.reward(sj) for sj in states], dtype=float)
    r = P @ rewards
    return P, r
