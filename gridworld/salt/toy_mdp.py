from __future__ import annotations

"""Exact toy MDP for quantifying the separation-bound gap.

This module implements a *small* cooperative multi-agent MDP on a grid in
which **both** sides of the separation bound of Theorem 3.1 can be computed
exactly by brute-force dynamic programming:

* the **true** population cost-to-go ``J_0^N`` (Problem 1 / Problem 2), obtained
  by backward DP over the *joint* state space ``S^N`` with an optimal-transport
  terminal cost, and
* the **OT surrogate** upper bound ``K[j_0](mu_0, nu)`` (eq. (5)), obtained by
  first computing the single-agent target-conditioned cost-to-go ``j_t(s, z)``
  (eq. (3)) via single-agent DP and then solving the assignment problem with
  transportation cost ``j_0``.

The quantity of interest is the **bound gap**

    gap(p) = K[j_0](mu_0, nu)  -  J_0^N(s_0, nu),

which is guaranteed non-negative by Theorem 3.1 and, by Corollary 3.2, vanishes
when the dynamics are deterministic. Here ``p`` is the action-noise level: with
probability ``p`` the intended action is replaced by a uniformly random *other*
action (the ``individual`` noise model of Section 2, i.e. noise drawn
independently across agents -- the canonical finite-population case in which the
realized empirical noise kernel is genuinely random and the ``E inf <= inf E``
interchange of Lemma B.3 is strict, cf. Example B.7).

The module additionally computes a **noise-variance proxy** -- the variance of
the realized one-step cost-to-go ``hat_j`` under the noise kernel, accumulated
along the optimal single-agent trajectories. The discussion after Theorem 3.1
attributes the gap to exactly this variance (optimal-transport couplings, being
minimizations, do not commute with the noise expectation); the proxy makes that
qualitative statement measurable and, like the gap, vanishes with ``p``.

Nothing here modifies the existing environment or algorithms: we only *reuse*
the action set and clamping semantics from :mod:`salt.env` and reproduce the
transition probabilities of the ``individual`` model of :mod:`salt.noise`.
"""

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .env import ACTIONS, clamp

Pos = Tuple[int, int]

# Ordered action deltas, matching salt.env.ACTIONS (up, down, left, right, stay).
_A = len(ACTIONS)
_DELTAS = [ACTIONS[a] for a in range(_A)]


# --------------------------------------------------------------------------- #
# Toy-MDP configuration                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class ToyConfig:
    """Configuration of the exact toy MDP.

    The defaults describe a small grid on which the joint DP over ``S^N`` is
    cheap: ``|S| = h * w`` and the joint value tensor has ``|S|^N`` entries.
    """

    h: int = 3
    w: int = 3
    horizon: int = 6
    n_agents: int = 3
    # Transportation / terminal cost exponent: ``g_T(s, z) = ||s - z||_1^power``.
    # power = 1 recovers the 1-Wasserstein (Manhattan) terminal cost.
    terminal_power: int = 1
    # Per-active-step stage cost. Kept at 0 by default so that the gap isolates
    # the terminal optimal-transport interchange (the sole source of slack in
    # Theorem 3.1); a deterministic stage cost does not affect the gap.
    step_cost: float = 0.0
    noise_kind: str = "individual"

    @property
    def n_states(self) -> int:
        return self.h * self.w


# --------------------------------------------------------------------------- #
# Grid geometry helpers                                                        #
# --------------------------------------------------------------------------- #
def state_coords(cfg: ToyConfig) -> np.ndarray:
    """Return an ``(|S|, 2)`` int array of ``(row, col)`` for each flat state."""
    coords = np.array([(s // cfg.w, s % cfg.w) for s in range(cfg.n_states)],
                      dtype=np.int64)
    return coords


def flat_index(cfg: ToyConfig, pos: Pos) -> int:
    return int(pos[0]) * cfg.w + int(pos[1])


def manhattan_matrix(cfg: ToyConfig) -> np.ndarray:
    """``(|S|, |S|)`` matrix of Manhattan distances between flat states."""
    c = state_coords(cfg)
    return (np.abs(c[:, None, 0] - c[None, :, 0])
            + np.abs(c[:, None, 1] - c[None, :, 1])).astype(np.float64)


def terminal_cost_matrix(cfg: ToyConfig) -> np.ndarray:
    """``g_T(s, z) = ||s - z||_1^power`` as an ``(|S|, |S|)`` matrix."""
    d = manhattan_matrix(cfg)
    return d if cfg.terminal_power == 1 else d ** cfg.terminal_power


# --------------------------------------------------------------------------- #
# Single-agent transition kernel                                              #
# --------------------------------------------------------------------------- #
def build_single_agent_kernel(cfg: ToyConfig, p: float) -> np.ndarray:
    """Exact single-agent transition kernel ``P1[s, a, s']`` under individual noise.

    With probability ``1 - p`` the intended action ``a`` is executed; with
    probability ``p`` a uniformly random *different* action is executed. This is
    the per-agent law ``xi(. | s, a)`` of the ``individual`` model in
    :mod:`salt.noise`. Positions are clamped to the grid, so several actions may
    map to the same next state (their probabilities accumulate).
    """
    if cfg.noise_kind != "individual":
        raise NotImplementedError(
            "The exact joint DP factorizes over agents only for 'individual' "
            "noise; 'local'/'global' induce a correlated joint kernel."
        )
    S = cfg.n_states
    P1 = np.zeros((S, _A, S), dtype=np.float64)
    for r in range(cfg.h):
        for col in range(cfg.w):
            s = r * cfg.w + col
            # Precompute the next state for every action from this cell.
            nxt = np.empty(_A, dtype=np.int64)
            for a in range(_A):
                dr, dc = _DELTAS[a]
                nr = clamp(r + dr, 0, cfg.h - 1)
                nc = clamp(col + dc, 0, cfg.w - 1)
                nxt[a] = nr * cfg.w + nc
            for a in range(_A):
                P1[s, a, nxt[a]] += (1.0 - p)
                if p > 0.0:
                    share = p / (_A - 1)
                    for a2 in range(_A):
                        if a2 == a:
                            continue
                        P1[s, a, nxt[a2]] += share
    # Numerical safety: rows are probability distributions.
    row_sums = P1.sum(axis=2, keepdims=True)
    P1 /= row_sums
    return P1


# --------------------------------------------------------------------------- #
# Single-agent target-conditioned cost-to-go  j_t(s, z)   (eq. (2)-(3))        #
# --------------------------------------------------------------------------- #
def single_agent_costtogo(
    cfg: ToyConfig, P1: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward DP for ``j_t(s, z)`` and the greedy action ``a*_t(s, z)``.

    Returns
    -------
    j : ``(T + 1, |S|, |S|)`` array with ``j[t, s, z]``.
    a_opt : ``(T, |S|, |S|)`` int array with the cost-minimizing action.

    Implements ``j_T(s, z) = g_T(s, z)`` and the Bellman recursion (3)
    ``j_t(s, z) = min_a E_w[ g_t + j_{t+1}(f(s, a, w), z) ]`` with the noise
    expectation taken under ``P1``.
    """
    S = cfg.n_states
    T = cfg.horizon
    gT = terminal_cost_matrix(cfg)  # (S, S) -> (s, z)

    j = np.zeros((T + 1, S, S), dtype=np.float64)
    a_opt = np.zeros((T, S, S), dtype=np.int64)
    j[T] = gT
    for t in range(T - 1, -1, -1):
        # EV[s, a, z] = sum_{s'} P1[s, a, s'] * j[t+1, s', z]
        # -> einsum over s': P1 (S,A,S') x j (S',S_z)
        EV = np.einsum("sau,uz->saz", P1, j[t + 1], optimize=True)
        Q = cfg.step_cost + EV  # (S, A, Z)
        j[t] = Q.min(axis=1)
        a_opt[t] = Q.argmin(axis=1)
    return j, a_opt


# --------------------------------------------------------------------------- #
# Optimal-transport / assignment helpers                                       #
# --------------------------------------------------------------------------- #
def assignment_cost(cost_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    """Minimum 1-to-1 assignment cost and the assigned column per row.

    For equal-mass empirical measures the optimal transport plan is a
    permutation (Birkhoff-von Neumann), so ``K`` reduces to this linear
    assignment. The returned cost is the *sum*; callers normalize by ``N``.
    """
    row, col = linear_sum_assignment(cost_matrix)
    total = float(cost_matrix[row, col].sum())
    order = np.empty(cost_matrix.shape[0], dtype=np.int64)
    order[row] = col
    return total, order


def ot_surrogate(
    cfg: ToyConfig,
    j: np.ndarray,
    starts_idx: List[int],
    targets_idx: List[int],
) -> Tuple[float, np.ndarray]:
    """OT surrogate bound ``K[j_0](mu_0, nu)`` and the induced assignment.

    Cost matrix ``C[i, k] = j_0(s^i, z^k)``; the value is the normalized
    minimum-cost assignment. Returns ``(value, sigma)`` where ``sigma[i]`` is the
    *target index* assigned to agent ``i``.
    """
    C = j[0][np.ix_(starts_idx, targets_idx)]  # (N, N): j_0(s^i, z^k)
    total, order = assignment_cost(C)
    return total / len(starts_idx), order


def terminal_transport(
    cfg: ToyConfig, positions_idx: Tuple[int, ...], targets_idx: List[int]
) -> float:
    """Normalized terminal OT cost ``K[g_T](mu_T, nu)`` for a joint state."""
    gT = terminal_cost_matrix(cfg)
    C = gT[np.ix_(list(positions_idx), targets_idx)]
    total, _ = assignment_cost(C)
    return total / len(positions_idx)


# --------------------------------------------------------------------------- #
# Exact joint (true) population cost-to-go  J_0^N   (Problem 1, eq. (6))        #
# --------------------------------------------------------------------------- #
def joint_true_costtogo(
    cfg: ToyConfig,
    P1: np.ndarray,
    starts_idx: List[int],
    targets_idx: List[int],
) -> float:
    """Exact brute-force DP for the true population cost-to-go ``J_0^N``.

    The joint value ``V_t`` is a tensor over the ordered joint state space
    ``S^N``.  Terminal condition ``V_T(s) = K[g_T](mu_s, nu)`` (the
    optimal-transport terminal cost of the realized population), and backward
    recursion
        ``V_t(s) = min_a  E_w[ stage + V_{t+1}(s') ]``.
    Because ``individual`` noise is independent across agents, the joint
    transition factorizes, ``P(s' | s, a) = prod_i P1(s'^i | s^i, a^i)``, so the
    expectation is a per-mode contraction of ``V_{t+1}`` with ``P1``.

    Returns the normalized (per-agent) value ``V_0`` at the given ordered start
    configuration -- this is ``J_0^N(s_0, nu^N)`` of Problem 1.
    """
    N = cfg.n_agents
    S = cfg.n_states
    if len(starts_idx) != N or len(targets_idx) != N:
        raise ValueError("starts_idx and targets_idx must both have length N.")

    shape = (S,) * N

    # Terminal tensor: V_T(joint state) = normalized OT cost to targets.
    gT = terminal_cost_matrix(cfg)
    tgt = list(targets_idx)
    V = np.empty(shape, dtype=np.float64)
    for joint in np.ndindex(*shape):
        C = gT[np.ix_(list(joint), tgt)]
        row, col = linear_sum_assignment(C)
        V[joint] = C[row, col].sum() / N

    joint_actions = list(product(range(_A), repeat=N))
    stage = cfg.step_cost  # total normalized stage cost per step contributed below

    for _t in range(cfg.horizon - 1, -1, -1):
        Vt = np.full(shape, np.inf, dtype=np.float64)
        for a_tuple in joint_actions:
            EV = V
            for i, a in enumerate(a_tuple):
                M = P1[:, a, :]                       # M[s, s'] = P1(s' | s, a)
                EV = np.tensordot(M, EV, axes=([1], [i]))
                EV = np.moveaxis(EV, 0, i)
            # Stage cost: cfg.step_cost per active agent per step, normalized by N.
            np.minimum(Vt, stage + EV, out=Vt)
        V = Vt

    return float(V[tuple(starts_idx)])


# --------------------------------------------------------------------------- #
# Noise-variance proxy (Lemma B.3 / discussion after Theorem 3.1)             #
# --------------------------------------------------------------------------- #
def noise_variance_proxy(
    cfg: ToyConfig,
    P1: np.ndarray,
    j: np.ndarray,
    a_opt: np.ndarray,
    starts_idx: List[int],
    sigma_targets_idx: List[int],
) -> float:
    """Variance of the realized cost-to-go ``hat_j`` under the noise kernel.

    The discussion after Theorem 3.1 identifies the gap's origin as the
    non-commutation of the optimal-transport minimization with the noise
    expectation (Lemma B.3, ``E inf <= inf E``). The controlling quantity is the
    variance of the *realized* one-step cost-to-go
        ``hat_j_t(s, z) = g_t + j_{t+1}(f(s, a*, w), z)``
    under the noise kernel ``w ~ xi(. | s, a*)``. This routine accumulates that
    variance along the optimal single-agent trajectory of each agent toward its
    surrogate-assigned target and averages over agents.

    ``sigma_targets_idx[i]`` is the flat *state* index of the target assigned to
    agent ``i`` by the OT surrogate. Returns a normalized (per-agent) scalar
    that vanishes as ``p -> 0`` (the kernel becomes deterministic).
    """
    S = cfg.n_states
    T = cfg.horizon
    total = 0.0
    for i, s0 in enumerate(starts_idx):
        z = sigma_targets_idx[i]
        d = np.zeros(S, dtype=np.float64)
        d[s0] = 1.0
        acc = 0.0
        for t in range(T):
            a_col = a_opt[t][:, z]                    # optimal action per state
            kernel_pol = P1[np.arange(S), a_col, :]   # (S, S') under greedy policy
            jz = j[t + 1][:, z]                       # cost-to-go of next states
            mean_s = kernel_pol @ jz                  # E[j_{t+1} | s, a*]
            ex2_s = kernel_pol @ (jz ** 2)
            var_s = np.maximum(ex2_s - mean_s ** 2, 0.0)
            acc += float(d @ var_s)                   # weight by visitation prob
            d = d @ kernel_pol                        # propagate state distribution
        total += acc
    return total / cfg.n_agents


# --------------------------------------------------------------------------- #
# One full evaluation for a fixed configuration and noise level                #
# --------------------------------------------------------------------------- #
@dataclass
class GapResult:
    p: float
    true_cost: float          # J_0^N
    surrogate: float          # K[j_0](mu_0, nu)
    gap: float                # surrogate - true (>= 0 by Theorem 3.1)
    rel_gap: float            # gap / max(true, eps)
    variance_proxy: float     # Var of hat_j under the noise kernel


def evaluate_gap(
    cfg: ToyConfig,
    p: float,
    starts_idx: List[int],
    targets_idx: List[int],
    check: bool = True,
) -> GapResult:
    """Compute the exact true cost, OT surrogate, gap and variance proxy."""
    P1 = build_single_agent_kernel(cfg, p)
    j, a_opt = single_agent_costtogo(cfg, P1)

    surrogate, sigma = ot_surrogate(cfg, j, starts_idx, targets_idx)
    true_cost = joint_true_costtogo(cfg, P1, starts_idx, targets_idx)
    gap = surrogate - true_cost

    sigma_targets = [targets_idx[k] for k in sigma]
    var_proxy = noise_variance_proxy(cfg, P1, j, a_opt, starts_idx, sigma_targets)

    if check and gap < -1e-8:
        raise AssertionError(
            f"Separation bound violated (Theorem 3.1): surrogate={surrogate:.6f}"
            f" < true={true_cost:.6f} at p={p:.3f}."
        )

    eps = 1e-9
    return GapResult(
        p=float(p),
        true_cost=float(true_cost),
        surrogate=float(surrogate),
        gap=float(max(gap, 0.0)),
        rel_gap=float(max(gap, 0.0) / max(abs(true_cost), eps)),
        variance_proxy=float(var_proxy),
    )


# --------------------------------------------------------------------------- #
# Canonical and random configurations                                          #
# --------------------------------------------------------------------------- #
def canonical_config(cfg: ToyConfig) -> Tuple[List[int], List[int]]:
    """Symmetric, paper-aligned configuration.

    Agents start in the first column, targets sit in the last column, at the
    same set of rows. This mirrors the grid-world setup (start col 0, targets in
    the final columns) and is symmetric ex-ante, so any ex-post reassignment
    gain is purely noise-driven -- the cleanest illustration of the interchange.
    """
    rows = list(range(cfg.n_agents))
    if cfg.n_agents > cfg.h:
        raise ValueError("n_agents must be <= grid height for the canonical config.")
    starts = [r * cfg.w + 0 for r in rows]
    targets = [r * cfg.w + (cfg.w - 1) for r in rows]
    return starts, targets


def random_config(
    cfg: ToyConfig, rng: np.random.Generator
) -> Tuple[List[int], List[int]]:
    """Random start/target configuration (distinct cells within each group)."""
    S = cfg.n_states
    starts = list(rng.choice(S, size=cfg.n_agents, replace=False))
    targets = list(rng.choice(S, size=cfg.n_agents, replace=False))
    return [int(s) for s in starts], [int(t) for t in targets]
