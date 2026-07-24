"""
Mean-Field Q-learning, *faithful* variant (MFQ-local).

This is a second Mean-Field Q-learning baseline that stays closer to the
original Yang et al., "Mean Field Multi-Agent Reinforcement Learning"
(ICML 2018) design along the two axes where the primary `mfq.py` baseline
deliberately departs from it:

1. **N-independent observation.** The primary MFQ observes the relative
   offset + active flag of *every* target (`obs_dim = 5 + 3*M`), so the
   per-agent representation — and hence the replay footprint — grows with the
   population. That contradicts MFQ's signature property (Yang et al., Sec. 4:
   "the parameters of the Q-function is independent of the number of agents").
   Here the target set is encoded as a fixed-resolution **egocentric density
   map**: a `K x K` histogram of the active targets' positions *relative to the
   agent*, with each bin holding (#active targets in bin) / N. The map sums to
   the fraction of targets still active, so it carries both the spatial
   distribution and the remaining quantity in `obs_dim = 5 + K*K` features,
   constant in N. This mirrors the mean-field philosophy: the population of
   targets is summarised as a distribution, exactly as the population of agents
   is summarised by the mean action.

2. **Local mean field.** The primary MFQ averages the Boltzmann policy over
   *all* active agents (the Gaussian-Squeeze / global-coupling regime). Here the
   mean action `a_bar_i` is averaged only over active agents within a spatial
   radius `R` of agent `i` (Chebyshev distance, i.e. a (2R+1)x(2R+1) box),
   matching the *nearest-neighbour* mean field of the Ising and Battle-game
   experiments. Agents with no active neighbour in range fall back to a uniform
   mean action.

Everything else — the network, replay buffer, Boltzmann fixed point, soft
mean-field Bellman target, double-Q option, noise model, per-agent reward
decomposition (own step cost / goal bonus + Hungarian-split terminal OT cost),
early stopping, and evaluation metrics — is identical to `mfq.py`, so the two
MFQ variants and the VDN/QMIX/QPLEX baselines remain apples-to-apples.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import GridConfig, ACTIONS, Pos, target_col_lo
from .noise import NoiseConfig
from .matching import terminal_ot_cost, target_coverage_rate, assign_goals, manhattan
# Reuse the generic, obs-shape-agnostic pieces from the primary MFQ baseline.
from .mfq import MFQQNet, MFQReplayBuffer, _vec_slip, _geom_decay, _ACTIONS_ARR

MFQ_LOCAL_OBS_MODES = {"density_map", "fov"}

torch.set_num_threads(min(torch.get_num_threads(), 8))


@dataclass
class MFQLocalConfig:
    gamma: float = 0.98
    hidden_size: int = 64
    lr: float = 3e-4
    batch_episodes: int = 64
    n_batches: int = 500
    target_update_every: int = 50
    # Target encoding: "density_map" (coarse egocentric K x K histogram spanning
    # the whole relative range) or "fov" (a (2R+1)x(2R+1) egocentric window with a
    # target-presence channel). Both are independent of the number of agents.
    obs_mode: str = "density_map"
    # "density_map" resolution (K x K bins) -> obs_dim = 5 + K*K.
    density_bins: int = 5
    # "fov" window half-width R -> obs_dim = 5 + (2R+1)*(2R+1).
    fov_radius: int = 2
    # Local mean-field neighbourhood radius (Chebyshev distance, grid cells).
    # Independent of the target-observation window; see fov_radius for that.
    mf_radius: int = 3
    # Boltzmann temperature schedule (tau): high -> exploratory, low -> greedy.
    tau_start: float = 5.0
    tau_end: float = 0.1
    tau_decay_episodes: int = 20_000
    # Small uniform-random exploration floor, decayed like the temperature.
    eps_start: float = 0.3
    eps_end: float = 0.02
    eps_decay_episodes: int = 20_000
    # Mean-field fixed-point iterations per timestep.
    meanfield_iters: int = 2
    replay_capacity: int = 100_000
    min_replay_size: int = 2_048
    batch_size: int = 512
    updates_per_batch: int = 8
    double_q: bool = True
    # Early stopping (0 / disabled by default), mirroring MFQ/MAPPO/Sep-PPO.
    early_stop_patience_batches: int = 0
    early_stop_plateau_window_batches: int = 0
    early_stop_max_delta_reach_rate: float = 0.0
    early_stop_max_delta_mean_reward: float = 0.0
    early_stop_reach_target: float = 0.0
    # Plateau stop only fires once the reach-rate MA is at least this high; guards
    # against a false stop during early, flat-but-low exploration. 0 disables.
    early_stop_min_reach: float = 0.0


def _validate_obs_mode(obs_mode: str) -> str:
    if obs_mode not in MFQ_LOCAL_OBS_MODES:
        raise ValueError(
            f"Invalid MFQ-local obs mode '{obs_mode}'. "
            f"Expected one of {sorted(MFQ_LOCAL_OBS_MODES)}."
        )
    return obs_mode


def _fill_density_obs(
    obs_l: np.ndarray, pos_r: np.ndarray, pos_c: np.ndarray, aid: np.ndarray,
    reached: np.ndarray, tgt_r: np.ndarray, tgt_c: np.ndarray, tgt_active: np.ndarray,
    t: int, grid: GridConfig, H: int, K: int, N: int,
) -> None:
    """Populate an (E, N, 5 + K*K) observation buffer in place.

    Layout: [row, col, agent_id, time, not_reached, <K*K egocentric density>].
    The density bin for target j seen by agent i is determined by the relative
    offset (tgt - pos) mapped onto a K x K grid spanning the full relative range
    [-(h-1), h-1] x [-(w-1), w-1]; each bin holds (#active targets)/N.
    """
    E = pos_r.shape[0]
    M = tgt_r.shape[1]
    obs_l[:, :, 0] = pos_r / max(1, grid.h - 1)
    obs_l[:, :, 1] = pos_c / max(1, grid.w - 1)
    obs_l[:, :, 2] = aid
    obs_l[:, :, 3] = t / max(1, H - 1)
    obs_l[:, :, 4] = (~reached).astype(np.float32)
    obs_l[:, :, 5:] = 0.0

    # Relative target offsets per (env, agent, target): (E, N, M).
    rel_r = tgt_r[:, None, :] - pos_r[:, :, None]
    rel_c = tgt_c[:, None, :] - pos_c[:, :, None]
    denom_r = max(1, 2 * (grid.h - 1))
    denom_c = max(1, 2 * (grid.w - 1))
    frac_r = (rel_r + (grid.h - 1)) / denom_r
    frac_c = (rel_c + (grid.w - 1)) / denom_c
    bin_r = np.clip((frac_r * K).astype(np.int64), 0, K - 1)
    bin_c = np.clip((frac_c * K).astype(np.int64), 0, K - 1)
    flat = bin_r * K + bin_c  # (E, N, M) in [0, K*K)

    dens = obs_l[:, :, 5:]  # view into the buffer, (E, N, K*K)
    active_mask = np.broadcast_to(tgt_active[:, None, :], (E, N, M))
    ee, ii, jj = np.nonzero(active_mask)
    if ee.size:
        np.add.at(dens, (ee, ii, flat[ee, ii, jj]), 1.0)
        dens /= max(1, N)


def _fill_fov_obs(
    obs_l: np.ndarray, pos_r: np.ndarray, pos_c: np.ndarray, aid: np.ndarray,
    reached: np.ndarray, tgt_r: np.ndarray, tgt_c: np.ndarray, tgt_active: np.ndarray,
    t: int, grid: GridConfig, H: int, R: int, N: int,
) -> None:
    """Populate an (E, N, 5 + (2R+1)^2) observation buffer in place.

    Layout: [row, col, agent_id, time, not_reached, <(2R+1)^2 FOV cells>]. Each FOV
    cell is a binary target-presence flag for an active target at that relative
    offset within the (2R+1)x(2R+1) window centred on the agent (Battle-game style).
    Targets outside the window are not observed, so unlike the density map the FOV
    can be all-zero when no target is nearby.
    """
    E = pos_r.shape[0]
    M = tgt_r.shape[1]
    W = 2 * R + 1
    obs_l[:, :, 0] = pos_r / max(1, grid.h - 1)
    obs_l[:, :, 1] = pos_c / max(1, grid.w - 1)
    obs_l[:, :, 2] = aid
    obs_l[:, :, 3] = t / max(1, H - 1)
    obs_l[:, :, 4] = (~reached).astype(np.float32)
    obs_l[:, :, 5:] = 0.0

    rel_r = tgt_r[:, None, :] - pos_r[:, :, None]  # (E, N, M)
    rel_c = tgt_c[:, None, :] - pos_c[:, :, None]
    in_win = (np.abs(rel_r) <= R) & (np.abs(rel_c) <= R) & tgt_active[:, None, :]
    cell = (rel_r + R) * W + (rel_c + R)  # (E, N, M), valid only where in_win
    fov = obs_l[:, :, 5:]  # view, (E, N, W*W)
    ee, ii, jj = np.nonzero(in_win)
    if ee.size:
        # Binary presence: multiple co-located targets still read as "present".
        fov[ee, ii, cell[ee, ii, jj]] = 1.0


def _target_obs_dim(obs_mode: str, density_bins: int, fov_radius: int) -> int:
    if obs_mode == "density_map":
        return 5 + int(density_bins) * int(density_bins)
    if obs_mode == "fov":
        return 5 + (2 * int(fov_radius) + 1) ** 2
    raise ValueError(f"Invalid MFQ-local obs mode '{obs_mode}'.")


def _fill_target_obs(
    obs_mode: str, obs_l: np.ndarray, pos_r: np.ndarray, pos_c: np.ndarray,
    aid: np.ndarray, reached: np.ndarray, tgt_r: np.ndarray, tgt_c: np.ndarray,
    tgt_active: np.ndarray, t: int, grid: GridConfig, H: int, N: int,
    density_bins: int, fov_radius: int,
) -> None:
    """Dispatch to the configured N-independent target encoder."""
    if obs_mode == "density_map":
        _fill_density_obs(obs_l, pos_r, pos_c, aid, reached, tgt_r, tgt_c,
                          tgt_active, t, grid, H, int(density_bins), N)
    else:
        _fill_fov_obs(obs_l, pos_r, pos_c, aid, reached, tgt_r, tgt_c,
                      tgt_active, t, grid, H, int(fov_radius), N)


def _local_meanfield_policy(
    q_net: MFQQNet, obs_flat: torch.Tensor, pos_r: np.ndarray, pos_c: np.ndarray,
    active_f: np.ndarray, E: int, N: int, n_act: int, tau: float, iters: int,
    radius: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterate the *local* mean-action / Boltzmann-policy fixed point.

    Returns (policy, mean_action), both (E, N, n_act). For each agent i the mean
    action is averaged over active agents within Chebyshev distance `radius`
    (excluding self); agents with no active neighbour use a uniform mean action.
    The neighbour mask depends only on positions/activity, so it is built once
    and reused across fixed-point iterations.
    """
    pr = pos_r.astype(np.int64)
    pc = pos_c.astype(np.int64)
    dr = np.abs(pr[:, :, None] - pr[:, None, :])
    dc = np.abs(pc[:, :, None] - pc[:, None, :])
    cheb = np.maximum(dr, dc)  # (E, N, N)
    neigh = (cheb <= radius)
    eye = np.eye(N, dtype=bool)[None, :, :]
    neigh = neigh & (~eye)                          # exclude self
    neigh = neigh & (active_f[:, None, :] > 0.5)    # only active neighbours
    neigh_t = torch.from_numpy(neigh.astype(np.float32))  # (E, N, N)
    counts = neigh_t.sum(dim=2, keepdim=True)             # (E, N, 1)
    has_neigh = (counts > 0).float()

    mean_a = torch.full((E, N, n_act), 1.0 / n_act, dtype=torch.float32)
    policy = None
    with torch.no_grad():
        for _ in range(max(1, iters)):
            q = q_net(obs_flat, mean_a.reshape(E * N, n_act)).reshape(E, N, n_act)
            policy = F.softmax(q / max(tau, 1e-6), dim=-1)
            local_mean = torch.bmm(neigh_t, policy) / counts.clamp_min(1.0)
            mean_a = has_neigh * local_mean + (1.0 - has_neigh) * (1.0 / n_act)
    pol_np = policy.numpy().astype(np.float32)
    mean_np = mean_a.numpy().astype(np.float32)
    return pol_np, mean_np


def train_mfq_local(
    grid: GridConfig,
    noise_cfg: NoiseConfig,
    n_agents: int,
    cfg: MFQLocalConfig | None = None,
    seed: int = 0,
    log_every: int = 10,
    print_every: int = 1000,
    n_targets: int | None = None,
    wb_run=None,
    wb_prefix: str = "train_mfq_local",
) -> Tuple[MFQQNet, Dict[str, Any]]:
    if cfg is None:
        cfg = MFQLocalConfig()
    obs_mode = _validate_obs_mode(cfg.obs_mode)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n_act = len(ACTIONS)
    H, N, E = grid.horizon, n_agents, cfg.batch_episodes
    M = n_targets if n_targets is not None else N
    K = int(cfg.density_bins)
    Rfov = int(cfg.fov_radius)
    R = int(cfg.mf_radius)
    obs_dim = _target_obs_dim(obs_mode, K, Rfov)

    q_net = MFQQNet(obs_dim, n_act, cfg.hidden_size)
    target_net = copy.deepcopy(q_net)
    opt = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay = MFQReplayBuffer(cfg.replay_capacity, N, obs_dim, n_act)
    total_updates = 0

    aid = np.arange(N, dtype=np.float32) / max(1, N - 1)
    kind = noise_cfg.kind
    p_noise = float(noise_cfg.p)
    batch_rewards_log: List[float] = []
    reach_log: List[float] = []
    train_curve: List[Dict[str, float]] = []
    consecutive_small_updates = 0
    consecutive_target_ok = 0
    early_stopped = False
    stop_reason = ""
    if wb_run is not None:
        p = wb_prefix
        wb_run.define_metric(f"{p}/iter")
        wb_run.define_metric(f"{p}/*", step_metric=f"{p}/iter")

    for batch_idx in range(cfg.n_batches):
        pos_r = rng.integers(0, grid.h, size=(E, N)).astype(np.int32)
        pos_c = np.zeros((E, N), dtype=np.int32)
        tgt_r = rng.integers(0, grid.h, size=(E, M)).astype(np.int32)
        col_lo = target_col_lo(grid)
        tgt_c = rng.integers(col_lo, grid.w, size=(E, M)).astype(np.int32)
        reached = np.zeros((E, N), dtype=bool)
        tgt_active = np.ones((E, M), dtype=bool)

        s_obs = np.empty((H, E, N, obs_dim), dtype=np.float32)
        s_mean_a = np.empty((H, E, N, n_act), dtype=np.float32)
        s_acts = np.empty((H, E, N), dtype=np.int64)
        s_rew = np.empty((H, E, N), dtype=np.float32)
        s_mask = np.empty((H, E, N), dtype=np.float32)
        s_next_obs = np.empty((H, E, N, obs_dim), dtype=np.float32)
        s_next_mean_a = np.empty((H, E, N, n_act), dtype=np.float32)
        s_next_mask = np.empty((H, E, N), dtype=np.float32)
        s_not_done = np.empty((H, E, N), dtype=np.float32)

        obs_l = np.empty((E, N, obs_dim), dtype=np.float32)
        next_obs_l = np.empty((E, N, obs_dim), dtype=np.float32)

        tau = _geom_decay(batch_idx * E, cfg.tau_start, cfg.tau_end, cfg.tau_decay_episodes)
        eps = _geom_decay(batch_idx * E, cfg.eps_start, cfg.eps_end, cfg.eps_decay_episodes)

        # As in mfq.py: the frozen-network next-state mean-field solution at step
        # t equals the current-state solution at t+1 (same obs, positions, and
        # active set carry over). Cache and reuse it. Reset each batch.
        cached_policy = None
        cached_mean_a = None
        for t in range(H):
            _fill_target_obs(obs_mode, obs_l, pos_r, pos_c, aid, reached, tgt_r,
                             tgt_c, tgt_active, t, grid, H, N, K, Rfov)
            active_f = (~reached).astype(np.float32)
            mask = active_f.copy()

            if cached_policy is not None:
                policy, mean_a = cached_policy, cached_mean_a
            else:
                obs_flat = torch.from_numpy(obs_l.reshape(E * N, obs_dim))
                policy, mean_a = _local_meanfield_policy(
                    q_net, obs_flat, pos_r, pos_c, active_f, E, N, n_act, tau,
                    cfg.meanfield_iters, R,
                )
            cum = np.cumsum(policy, axis=-1)
            draws = rng.random((E, N, 1))
            sampled = (draws < cum).argmax(axis=-1).astype(np.int64)
            rand_a = rng.integers(0, n_act, size=(E, N), dtype=np.int64)
            use_rand = rng.random((E, N)) < eps
            actions = np.where(use_rand, rand_a, sampled).astype(np.int64)
            actions[reached] = 4

            exec_a = actions.copy().astype(np.int32)
            active = ~reached
            if kind != "none" and p_noise > 0:
                if kind == "individual":
                    flat = exec_a.ravel()
                    idx = np.where(active.ravel())[0]
                    if len(idx):
                        slip = rng.random(len(idx)) < p_noise
                        si = idx[slip]
                        if len(si):
                            flat[si] = _vec_slip(rng, flat[si], n_act)
                elif kind == "global":
                    ep_slip = rng.random(E) < p_noise
                    for e in np.where(ep_slip)[0]:
                        ie = np.where(active[e])[0]
                        if len(ie):
                            exec_a[e, ie] = _vec_slip(rng, exec_a[e, ie], n_act)
                elif kind == "local":
                    cache: Dict[tuple, int] = {}
                    for e in range(E):
                        for i in np.where(active[e])[0]:
                            key = (e, int(pos_r[e, i]), int(pos_c[e, i]), int(actions[e, i]))
                            if key not in cache:
                                if rng.random() < p_noise:
                                    o = int(rng.integers(0, n_act - 1))
                                    a = int(actions[e, i])
                                    cache[key] = o + (1 if o >= a else 0)
                                else:
                                    cache[key] = int(actions[e, i])
                            exec_a[e, i] = cache[key]

            new_r = pos_r.copy()
            new_c = pos_c.copy()
            dr = _ACTIONS_ARR[exec_a, 0]
            dc = _ACTIONS_ARR[exec_a, 1]
            new_r[active] = np.clip(pos_r[active] + dr[active], 0, grid.h - 1).astype(np.int32)
            new_c[active] = np.clip(pos_c[active] + dc[active], 0, grid.w - 1).astype(np.int32)

            # Per-agent reward: own step cost / own goal bonus.
            rew = np.zeros((E, N), dtype=np.float32)
            for e in range(E):
                for i in np.where(active[e])[0]:
                    ri, ci = int(new_r[e, i]), int(new_c[e, i])
                    candidates = np.where(
                        tgt_active[e] & (tgt_r[e] == ri) & (tgt_c[e] == ci)
                    )[0]
                    if len(candidates):
                        tgt_active[e, int(candidates[0])] = False
                        reached[e, i] = True
                        rew[e, i] = float(grid.goal_bonus)
                    else:
                        rew[e, i] = -float(grid.step_cost)
            if t == H - 1:
                # Split the terminal OT cost across agents via Hungarian
                # assignment: sum_i manhattan(final_i, assigned_i) == OT cost.
                for e in range(E):
                    fp = [(int(new_r[e, i]), int(new_c[e, i])) for i in range(N)]
                    tl = [(int(tgt_r[e, j]), int(tgt_c[e, j])) for j in range(M)]
                    assigned = assign_goals(fp, tl)
                    for i in range(N):
                        rew[e, i] -= float(manhattan(fp[i], assigned[i]))

            next_t = min(t + 1, H - 1)
            _fill_target_obs(obs_mode, next_obs_l, new_r, new_c, aid, reached,
                             tgt_r, tgt_c, tgt_active, next_t, grid, H, N, K, Rfov)
            next_active_f = (~reached).astype(np.float32)
            next_obs_flat = torch.from_numpy(next_obs_l.reshape(E * N, obs_dim))
            next_policy, next_mean_a = _local_meanfield_policy(
                q_net, next_obs_flat, new_r, new_c, next_active_f, E, N, n_act,
                tau, cfg.meanfield_iters, R,
            )
            cached_policy, cached_mean_a = next_policy, next_mean_a

            s_obs[t] = obs_l
            s_mean_a[t] = mean_a
            s_acts[t] = actions
            s_rew[t] = rew
            s_mask[t] = mask
            s_next_obs[t] = next_obs_l
            s_next_mean_a[t] = next_mean_a
            s_next_mask[t] = next_active_f
            all_reached = reached.all(axis=1)
            nd = ((t < (H - 1)) & (~all_reached)).astype(np.float32)
            s_not_done[t] = np.repeat(nd[:, None], N, axis=1)
            pos_r, pos_c = new_r, new_c

        B = H * E
        replay.add_batch(
            obs=s_obs.reshape(B, N, obs_dim),
            mean_a=s_mean_a.reshape(B, N, n_act),
            acts=s_acts.reshape(B, N),
            rew=s_rew.reshape(B, N),
            mask=s_mask.reshape(B, N),
            next_obs=s_next_obs.reshape(B, N, obs_dim),
            next_mean_a=s_next_mean_a.reshape(B, N, n_act),
            next_mask=s_next_mask.reshape(B, N),
            not_done=s_not_done.reshape(B, N),
        )

        loss_v = float("nan")
        if replay.size >= max(1, cfg.min_replay_size, cfg.batch_size):
            for _ in range(max(1, cfg.updates_per_batch)):
                batch = replay.sample(cfg.batch_size, rng)
                bs = int(batch["obs"].shape[0])
                b_obs = torch.from_numpy(batch["obs"].reshape(bs * N, obs_dim))
                b_mean = torch.from_numpy(batch["mean_a"].reshape(bs * N, n_act))
                b_next_obs = torch.from_numpy(batch["next_obs"].reshape(bs * N, obs_dim))
                b_next_mean = torch.from_numpy(batch["next_mean_a"].reshape(bs * N, n_act))
                b_acts = torch.from_numpy(batch["acts"])
                b_rew = torch.from_numpy(batch["rew"])
                b_mask = torch.from_numpy(batch["mask"])
                b_next_mask = torch.from_numpy(batch["next_mask"])
                b_not_done = torch.from_numpy(batch["not_done"])

                q_all = q_net(b_obs, b_mean).reshape(bs, N, n_act)
                q_taken = torch.gather(q_all, dim=2, index=b_acts.unsqueeze(-1)).squeeze(-1)

                with torch.no_grad():
                    q_next = target_net(b_next_obs, b_next_mean).reshape(bs, N, n_act)
                    if cfg.double_q:
                        q_next_online = q_net(b_next_obs, b_next_mean).reshape(bs, N, n_act)
                        pi_next = F.softmax(q_next_online / max(cfg.tau_end, 1e-6), dim=-1)
                    else:
                        pi_next = F.softmax(q_next / max(cfg.tau_end, 1e-6), dim=-1)
                    v_next = (pi_next * q_next).sum(dim=2) * b_next_mask
                    td_target = b_rew + cfg.gamma * b_not_done * v_next

                se = (q_taken - td_target) ** 2 * b_mask
                denom = b_mask.sum().clamp_min(1.0)
                loss = se.sum() / denom
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 0.5)
                opt.step()
                loss_v = float(loss.item())
                total_updates += 1
                if total_updates % max(1, cfg.target_update_every) == 0:
                    target_net.load_state_dict(q_net.state_dict())

        mean_rew = float(s_rew.sum()) / E
        reach_rate = float(reached.mean())
        batch_rewards_log.append(mean_rew)
        reach_log.append(reach_rate)
        episodes_done = (batch_idx + 1) * E
        train_curve.append({
            "batch": float(batch_idx + 1),
            "episodes": float(episodes_done),
            "env_interactions": float(episodes_done * N * H),
            "reach_rate": float(reach_rate),
        })
        if episodes_done % print_every == 0:
            rec = float(np.mean(batch_rewards_log[-log_every:]))
            print(
                f"[MFQlocal-{noise_cfg.kind}|{obs_mode}] batch {batch_idx+1:4d}/{cfg.n_batches}"
                f"  episodes={episodes_done:6d}"
                f"  mean_reward={rec:.2f}"
                f"  reach_rate={reach_rate:.2%}"
                f"  tau={tau:.2f}"
            )
        if (batch_idx + 1) % log_every == 0 and wb_run is not None:
            rec = float(np.mean(batch_rewards_log[-log_every:]))
            p = wb_prefix
            wb_run.log({
                f"{p}/iter": episodes_done,
                f"{p}/batch": batch_idx + 1,
                f"{p}/episodes": episodes_done,
                f"{p}/tau": float(tau),
                f"{p}/epsilon": float(eps),
                f"{p}/mean_reward": rec,
                f"{p}/last_batch_reward": float(mean_rew),
                f"{p}/reach_rate": reach_rate,
                f"{p}/td_loss": loss_v,
                f"{p}/replay_size": int(replay.size),
            })

        # ---- Early stopping (identical policy to mfq.py) ----
        stop_now = False
        if cfg.early_stop_reach_target > 0.0:
            if reach_rate >= cfg.early_stop_reach_target:
                consecutive_target_ok += 1
            else:
                consecutive_target_ok = 0
            tw = (cfg.early_stop_plateau_window_batches
                  or cfg.early_stop_patience_batches or 1)
            if consecutive_target_ok >= max(1, tw):
                stop_now = True
                stop_reason = (f"reach rate >= {cfg.early_stop_reach_target:.0%} "
                               f"for {consecutive_target_ok} batches")
        if not stop_now and cfg.early_stop_patience_batches > 0:
            w = cfg.early_stop_plateau_window_batches
            plateau_ok = False
            if w > 0 and len(reach_log) >= 2 * w:
                prev_reach = float(np.mean(reach_log[-2 * w:-w]))
                curr_reach = float(np.mean(reach_log[-w:]))
                reach_flat = (abs(curr_reach - prev_reach)
                              <= cfg.early_stop_max_delta_reach_rate)
                if cfg.early_stop_max_delta_mean_reward > 0.0:
                    prev_rew = float(np.mean(batch_rewards_log[-2 * w:-w]))
                    curr_rew = float(np.mean(batch_rewards_log[-w:]))
                    reward_flat = (abs(curr_rew - prev_rew)
                                   <= cfg.early_stop_max_delta_mean_reward)
                else:
                    reward_flat = True
                reach_high_enough = curr_reach >= cfg.early_stop_min_reach
                plateau_ok = reach_flat and reward_flat and reach_high_enough
            if plateau_ok:
                consecutive_small_updates += 1
            else:
                consecutive_small_updates = 0
            if consecutive_small_updates >= cfg.early_stop_patience_batches:
                stop_now = True
                stop_reason = (f"reach/reward plateau for "
                               f"{cfg.early_stop_patience_batches} batches (window={w})")
        if stop_now:
            early_stopped = True
            print(f"[MFQlocal-{noise_cfg.kind}|{obs_mode}] early stop at batch "
                  f"{batch_idx + 1}: {stop_reason}")
            break

    q_net.obs_mode = obs_mode
    q_net.n_targets = M
    q_net.density_bins = K
    q_net.fov_radius = Rfov
    q_net.mf_radius = R
    q_net.meanfield_iters = cfg.meanfield_iters
    q_net.tau_eval = cfg.tau_end
    batches_run = len(reach_log)
    return q_net, {
        "algorithm": "MFQ-local",
        "episodes": batches_run * E,
        "n_agents": N,
        "n_targets": M,
        "noise_kind": noise_cfg.kind,
        "p": noise_cfg.p,
        "mfq_local_config": asdict(cfg),
        "obs_mode": obs_mode,
        "density_bins": K,
        "fov_radius": Rfov,
        "mf_radius": R,
        "env_interactions": batches_run * E * N * H,
        "batches_run": batches_run,
        "early_stopped": bool(early_stopped),
        "early_stop_reason": stop_reason,
        "mean_reward_last10": float(np.mean(batch_rewards_log[-10:])),
        "train_curve": train_curve,
    }


def rollout_mfq_local(
    q_net: MFQQNet,
    grid: GridConfig,
    noise_cfg: NoiseConfig,
    n_agents: int,
    targets: List[Pos] | None = None,
    seed: int = 0,
    obs_mode: str | None = None,
    start_rows: List[int] | None = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    N, H = n_agents, grid.horizon
    n_act = len(ACTIONS)
    if targets is None:
        n_targets = int(getattr(q_net, "n_targets", grid.h))
        rows = rng.integers(0, grid.h, size=n_targets)
        col_lo = max(0, grid.w - 10)
        cols = rng.integers(col_lo, grid.w, size=n_targets)
        targets = [(int(r), int(c)) for r, c in zip(rows, cols)]

    if start_rows is None:
        pos_r = rng.integers(0, grid.h, size=N).astype(np.int32)
    else:
        if len(start_rows) != N:
            raise ValueError(
                f"start_rows length ({len(start_rows)}) must match n_agents ({N})"
            )
        pos_r = np.clip(np.asarray(start_rows, dtype=np.int32), 0, grid.h - 1)
    pos_c = np.zeros(N, dtype=np.int32)
    reached = np.zeros(N, dtype=bool)
    tgt_r = np.array([int(r) for r, _ in targets], dtype=np.int32)
    tgt_c = np.array([int(c) for _, c in targets], dtype=np.int32)
    M = len(targets)
    tgt_active = np.ones(M, dtype=bool)
    arrival = [None] * N
    trajectories: List[List[Pos]] = [
        [(int(pos_r[i]), int(pos_c[i]))] for i in range(N)
    ]
    total_cost = 0.0
    aid = np.arange(N, dtype=np.float32) / max(1, N - 1)

    if obs_mode is None:
        obs_mode = getattr(q_net, "obs_mode", "density_map")
    obs_mode = _validate_obs_mode(obs_mode)
    K = int(getattr(q_net, "density_bins", 5))
    Rfov = int(getattr(q_net, "fov_radius", 2))
    R = int(getattr(q_net, "mf_radius", 3))
    obs_dim = _target_obs_dim(obs_mode, K, Rfov)
    p_noise = float(noise_cfg.p)
    kind = noise_cfg.kind
    mf_iters = int(getattr(q_net, "meanfield_iters", 2))
    tau_eval = float(getattr(q_net, "tau_eval", 0.5))

    q_net.eval()
    for t in range(H):
        obs_l = np.empty((1, N, obs_dim), dtype=np.float32)
        _fill_target_obs(obs_mode, obs_l, pos_r[None, :], pos_c[None, :], aid,
                         reached[None, :], tgt_r[None, :], tgt_c[None, :],
                         tgt_active[None, :], t, grid, H, N, K, Rfov)
        active_f = (~reached).astype(np.float32)[None, :]
        obs_flat = torch.from_numpy(obs_l.reshape(N, obs_dim))
        policy, _ = _local_meanfield_policy(
            q_net, obs_flat, pos_r[None, :], pos_c[None, :], active_f, 1, N, n_act,
            tau_eval, mf_iters, R,
        )
        actions_np = policy[0].argmax(-1).astype(np.int64)
        actions_np[reached] = 4
        active = ~reached
        idx_a = np.where(active)[0]
        exec_a = actions_np.copy().astype(np.int32)

        if kind != "none" and p_noise > 0 and len(idx_a):
            if kind == "individual":
                slip = rng.random(len(idx_a)) < p_noise
                si = idx_a[slip]
                if len(si):
                    exec_a[si] = _vec_slip(rng, exec_a[si], n_act)
            elif kind == "global":
                if rng.random() < p_noise:
                    exec_a[idx_a] = _vec_slip(rng, exec_a[idx_a], n_act)
            elif kind == "local":
                cache: Dict[tuple, int] = {}
                for i in idx_a:
                    key = (int(pos_r[i]), int(pos_c[i]), int(actions_np[i]))
                    if key not in cache:
                        if rng.random() < p_noise:
                            o = int(rng.integers(0, n_act - 1))
                            a = int(actions_np[i])
                            cache[key] = o + (1 if o >= a else 0)
                        else:
                            cache[key] = int(actions_np[i])
                    exec_a[i] = cache[key]

        new_r, new_c = pos_r.copy(), pos_c.copy()
        if len(idx_a):
            sr, sc = pos_r[idx_a], pos_c[idx_a]
            new_r[idx_a] = np.clip(sr + _ACTIONS_ARR[exec_a[idx_a], 0], 0, grid.h - 1).astype(np.int32)
            new_c[idx_a] = np.clip(sc + _ACTIONS_ARR[exec_a[idx_a], 1], 0, grid.w - 1).astype(np.int32)

        nj = ns = 0
        for i in idx_a:
            ri, ci = int(new_r[i]), int(new_c[i])
            candidates = np.where(tgt_active & (tgt_r == ri) & (tgt_c == ci))[0]
            if len(candidates):
                tgt_active[int(candidates[0])] = False
                reached[i] = True
                arrival[i] = t + 1
                nj += 1
            else:
                ns += 1
        total_cost += float(grid.step_cost) * ns - float(grid.goal_bonus) * nj
        pos_r, pos_c = new_r, new_c
        for i in range(N):
            trajectories[i].append((int(pos_r[i]), int(pos_c[i])))

    fp = [(int(pos_r[i]), int(pos_c[i])) for i in range(N)]
    tc = terminal_ot_cost(fp, targets)
    cov = target_coverage_rate(fp, targets)
    nr = sum(1 for a in arrival if a is not None)
    rt = [a for a in arrival if a is not None]
    rt_with_horizon = [(a if a is not None else H) for a in arrival]
    return {
        "algorithm": "MFQ-local",
        "noise_kind": noise_cfg.kind,
        "p": noise_cfg.p,
        "obs_mode": obs_mode,
        "density_bins": K,
        "fov_radius": Rfov,
        "mf_radius": R,
        "agents": N,
        "horizon": H,
        "target_coverage": cov,
        "reach_rate": nr / N,
        "mean_time_to_reach": float(np.mean(rt)) if rt else float("nan"),
        "mean_time_to_reach_including_unreached": float(np.mean(rt_with_horizon)),
        "total_cost": float(total_cost),
        "terminal_ot_cost": float(tc),
        "final_positions": [list(p) for p in fp],
        "arrival_times": arrival,
        "targets": [list(t) for t in targets],
        "trajectories": [
            [list(p) for p in traj] for traj in trajectories
        ],
    }
