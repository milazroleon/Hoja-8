# template.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import numpy as np

from mdp import MDP, State, Action
from policy import Policy
from lake_mdp import UP, RIGHT, DOWN, LEFT, ABSORB
from mdp_utils import enumerate_states, build_policy_Pr

_ACTION_ORDER = (UP, RIGHT, DOWN, LEFT)


class TabularPolicy(Policy):
    """
    Deterministic or uniform-random tabular policy.

    A simple policy that, for each state, either:
      • returns the action stored in an internal table (deterministic), or
      • chooses uniformly at random among admissible actions if no table is provided.

    This class MUST:
      1) Be callable on a state s: `a = policy(s)` (implemented in the Policy base).
      2) Expose a probability distribution over actions via `probs(s)`:
         - If deterministic: 1.0 on the chosen action, 0.0 on others.
         - If random: uniform over admissible actions (no global RNG; use `self.rng`).
      3) Provide an alias `action_probs(s)` returning the same dict as `probs(s)`.
         The autograder will use this to build P^π and r^π without sampling.

    Parameters
    ----------
    mdp : MDP
        The environment implementing transition and reward interfaces.
    rng : np.random.Generator
        Random generator to use for all sampling in this policy.
    table : Optional[Dict[State, Action]]
        If provided, a deterministic mapping from state to action.
        If None, the policy acts as a uniform random policy.

    Notes
    -----
    • Terminal/absorbing states may have no admissible actions. In such cases,
      return ABSORB and a degenerate distribution {ABSORB: 1.0}.
    • Do not use global random state. Only use `self.rng`.
    """

    def __init__(
        self,
        mdp: MDP,
        rng: np.random.Generator,
        table: Optional[Dict[State, Action]] = None,
    ):
        super().__init__(mdp, rng)
        self.table = table

    def _decision(self, s: State) -> Action:
        if self.table is not None and s in self.table:
            return self.table[s]

        acts = list(self.mdp.actions(s))
        if not acts or acts == [ABSORB]:
            return ABSORB

        idx = self.rng.integers(len(acts))
        return acts[idx]
    
    def probs(self, s: State) -> Dict[Action, float]:
        acts = list(self.mdp.actions(s))
        if not acts or acts == [ABSORB]:
            return {ABSORB: 1.0}

        if self.table is not None and s in self.table:
            chosen_action = self.table[s]
            return {a: 1.0 if a == chosen_action else 0.0 for a in acts}

        uniform_prob = 1.0 / len(acts)
        return {a: uniform_prob for a in acts}

    def action_probs(self, s: State) -> Dict[Action, float]:
        return self.probs(s)

def q_from_v(
    mdp: MDP, v: Dict[State, float], gamma: float
) -> Dict[Tuple[State, Action], float]:
    """
    Compute state–action values q^π(s, a) given state values v^π(s) for the same policy π.

    Definition
    ----------
    q^π(s,a) = Σ_{s'} P(s' | s, a) [ r(s') + γ v^π(s') ]

    Requirements
    ------------
    • Interpret rewards as “on-entry”: `mdp.reward(s_next)` is the reward for entering s'.
    • Use the MDP’s transition iterator: `for (s_next, p) in mdp.transition(s, a)`.
    • If a state has no admissible actions (terminal), define q(s, ABSORB) = 0.0.
    • Use `enumerate_states(mdp)` to iterate states in a consistent order.

    Parameters
    ----------
    mdp : MDP
        The environment with transitions and rewards.
    v : Dict[State, float]
        State-value function under the same policy π.
    gamma : float
        Discount factor γ ∈ (0, 1].

    Returns
    -------
    Dict[Tuple[State, Action], float]
        Mapping (s, a) ↦ q^π(s, a).

    Notes
    -----
    • The provided `v` is assumed to correspond to the same policy π implicitly
      used when `enumerate_states` and `transition` are evaluated.
    """
    q = {}
    for s in enumerate_states(mdp):
        if mdp.is_terminal(s):
            continue
        for a in mdp.actions(s):
            total = 0.0
            for ns, p in mdp.transition(s, a):
                total += p * (mdp.reward(ns) + gamma * v.get(ns, 0.0))
            q[(s, a)] = total
    return q


def v_from_q(
    q: Dict[Tuple[State, Action], float], policy: TabularPolicy
) -> Dict[State, float]:
    """
    Compute state values v^π(s) given q^π(s, a) and a policy π.

    Deterministic π
    ---------------
    v^π(s) = q^π(s, π(s)).

    Stochastic (uniform) π
    ----------------------
    v^π(s) = Σ_a π(a|s) q^π(s, a).
    In this assignment, a stochastic TabularPolicy is uniform over admissible actions.

    Parameters
    ----------
    q : Dict[Tuple[State, Action], float]
        State–action values for a fixed policy π.
    policy : TabularPolicy
        A policy object providing action probabilities per state.

    Returns
    -------
    Dict[State, float]
        Mapping s ↦ v^π(s).

    Edge Cases
    ----------
    • Terminal states with no actions should return v(s) = 0.0.
    """
    mdp = policy.mdp
    states = enumerate_states(mdp)
    v: Dict[State, float] = {}

    for s in states:
        acts = list(mdp.actions(s))
        if acts == [ABSORB]:
            v[s] = 0.0
            continue

        probs = policy.action_probs(s)

        total = 0.0
        for a, pa in probs.items():
            total += float(pa) * float(q.get((s, a), 0.0))
        v[s] = float(total)

    return v


def policy_evaluation(
    P: np.ndarray,
    r: np.ndarray,
    gamma: float,
    states: List[State],
    eps: float = 1e-6,
    max_iters: int = 100_000,
) -> Dict[State, float]:
    """
    Iterative policy evaluation (matrix form) returning v^π as a dict keyed by `states`.

    Fixed-point iteration
    ---------------------
    v_{k+1} = r + γ P v_k,
    stopping when ||v_{k+1} - v_k||_∞ < eps * (1-γ)/γ (for γ < 1).

    Exact solve (γ ≈ 1)
    -------------------
    When γ is numerically 1 (or extremely close), prefer the direct linear solve:
        (I - γ P) v = r

    Parameters
    ----------
    P : np.ndarray, shape (S, S)
        Row-stochastic transition matrix under the current policy π (built via `build_policy_Pr`).
    r : np.ndarray, shape (S,)
        Reward vector aligned with `states`. Reward convention: on entry to state.
    gamma : float
        Discount factor γ ∈ (0, 1].
    states : List[State]
        State ordering corresponding to rows/cols of P and entries of r.
    eps : float, default=1e-6
        Tolerance for the contraction-based stopping rule.
    max_iters : int, default=100000
        Hard cap on iterations (should not be hit for γ < 1 with a reasonable eps).

    Returns
    -------
    Dict[State, float]
        Mapping s ↦ v^π(s) using the same ordering as `states`.

    Notes
    -----
    • Use only NumPy (no SciPy). Use `np.linalg.solve` for the exact solve path.
    • For γ very close to 1, prefer the exact solve path to avoid slow convergence.
    """
    S = P.shape[0]
    if abs(1.0 - gamma) < 1e-12:
        # Solve (I - gamma P) v = r
        I = np.eye(S, dtype=float)
        A = I - float(gamma) * P
        v_vec = np.linalg.solve(A, r)
        return {states[i]: float(v_vec[i]) for i in range(S)}

    v = np.zeros(S, dtype=float)
    tol = float(eps) * (1.0 - float(gamma)) / float(max(gamma, 1e-12))
    for _ in range(max_iters):
        v_next = r + float(gamma) * (P @ v)
        diff = np.max(np.abs(v_next - v))
        v = v_next
        if diff < tol:
            break
    return {states[i]: float(v[i]) for i in range(S)}


def _action_order_key(a: Action) -> Tuple[int, str]:
    """
    Stable tie-breaking key for actions.

    Returns a pair (rank, name) where:
      • rank is the index of `a` in `_ACTION_ORDER` (UP, RIGHT, DOWN, LEFT),
        or len(_ACTION_ORDER) if `a` is not present (e.g., ABSORB).
      • name is `str(a)` for lexicographic stability among unknown actions.

    This provides deterministic argmax behavior when q-values tie.

    Parameters
    ----------
    a : Action
        The action symbol to rank.

    Returns
    -------
    Tuple[int, str]
        Comparable key for sorted().
    """
    try:
        rank = _ACTION_ORDER.index(a)
    except ValueError:
        rank = len(_ACTION_ORDER)
    return (rank, str(a))


def policy_improvement(
    mdp: MDP, v: Dict[State, float], gamma: float
) -> Tuple[TabularPolicy, Dict[Tuple[State, Action], float]]:
    """
    Advantage-based policy improvement.

    Steps
    -----
    1) Compute q^π(s, a) from the provided v^π using `q_from_v`.
    2) Compute the advantage for every state–action pair:
           A^π(s, a) = q^π(s, a) - v^π(s).
    3) Improve the policy greedily:
           π'(s) = argmax_a A^π(s, a)
       with stable tie-breaking using `_ACTION_ORDER`.
    4) Return the improved deterministic TabularPolicy and the advantage dictionary.

    Parameters
    ----------
    mdp : MDP
        Environment used to enumerate states and actions.
    v : Dict[State, float]
        State-value function under the current policy π.
    gamma : float
        Discount factor γ ∈ (0, 1].

    Returns
    -------
    Tuple[TabularPolicy, Dict[Tuple[State, Action], float]]
        (π', advantage), where
         • π' is a deterministic `TabularPolicy` (has a `.table`),
         • advantage[(s, a)] = A^π(s, a) for all s and admissible a.

    Edge Cases
    ----------
    • If a state has no admissible actions, set π'(s) = ABSORB and skip arrows.
    """

    q = q_from_v(mdp, v, gamma)
    advantage = {}
    table = {}

    for s in enumerate_states(mdp):
        if mdp.is_terminal(s):
            continue

        best_a, best_val = None, float("-inf")
        for a in mdp.actions(s):
            adv = q[(s, a)] - v[s]
            advantage[(s, a)] = adv
            if adv > best_val or (abs(adv - best_val) < 1e-10 and _action_order_key(a) < _action_order_key(best_a)):
                best_a, best_val = a, adv

        table[s] = best_a
        advantage[(s, table[s])] = 0.0

    return TabularPolicy(mdp, mdp.rng, table=table), advantage
    


def policy_iteration(
    mdp: MDP,
    policy: Policy,
    gamma: float,
) -> Tuple[TabularPolicy, Dict[State, float]]:
    """
    Classic policy iteration (Howard's) until convergence.

    Loop
    ----
    repeat:
        1) Build P^π and r^π from current π via `build_policy_Pr(mdp, π, states)`.
        2) Evaluate π: v^π = policy_evaluation(P^π, r^π, γ, states).
        3) Improve: (π', A) = policy_improvement(mdp, v^π, γ).
    until π' == π  (stability check on the deterministic action table)

    Parameters
    ----------
    mdp : MDP
        The environment.
    policy : Policy
        Initial policy. May be a TabularPolicy with or without a table.
        If stochastic (no table), the first improvement will produce a deterministic table.
    gamma : float
        Discount factor γ ∈ (0, 1].

    Returns
    -------
    Tuple[TabularPolicy, Dict[State, float]]
        (π*, v^{π*}) where π* is stable (optimal for finite MDPs with γ ∈ (0,1)).

    Notes
    -----
    • Use `enumerate_states(mdp)` once per iteration to fix an ordering.
    • The stability check must be deterministic: compare action tables state by state.
    • Do not plot or print inside this function (keep it pure for autograding).
    """
    current_policy = policy

    while True:
        states = enumerate_states(mdp)
        P, r = build_policy_Pr(mdp, current_policy, states)
        v = policy_evaluation(P, r, gamma, states)

        # improve
        new_policy, _ = policy_improvement(mdp, v, gamma)

        stable = False
        if isinstance(current_policy, TabularPolicy) and current_policy.table is not None:
            stable = True
            for s in states:
                old_a = current_policy.table.get(s)
                new_a = new_policy.table.get(s)
                if old_a != new_a:
                    stable = False
                    break

        if stable:
            return current_policy, v
        else:
            current_policy = new_policy


def get_optimal_policy(
    mdp: MDP,
    gamma: float,
    rng: np.random.Generator,
) -> TabularPolicy:
    """
    Convenience runner:
      1) Construct a uniform-random TabularPolicy (no table).
      2) Run policy iteration to convergence.
      3) Return the optimal deterministic policy π*.

    Parameters
    ----------
    mdp : MDP
        The environment instance (e.g., LakeMDP).
    gamma : float
        Discount factor γ ∈ (0, 1].
    rng : np.random.Generator
        RNG to initialize the starting random policy.

    Returns
    -------
    TabularPolicy
        Optimal deterministic policy (has a `.table` covering all states).

    Notes
    -----
    • Do not change function name or signature (autograded).
    • Do not print or plot here.
    """
    init_policy = TabularPolicy(mdp=mdp, rng=rng, table=None)
    pi_star, _ = policy_iteration(mdp, init_policy, gamma)
    return pi_star
