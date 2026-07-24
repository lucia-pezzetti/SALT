"""
Mean-Field Q-learning (MFQ) cooperative MARL baseline.

Faithful to Yang et al., "Mean Field Multi-Agent Reinforcement Learning"
(ICML 2018). Each agent shares a per-agent Q-network whose input is augmented
with the population **mean action** (the average one-hot / Boltzmann action of
its neighbours):

    Q_i(o_i, a_i, a_bar_i),   a_bar_i = (1 / |N(i)|) * sum_{k in N(i)} a_k .

The mean action and the Boltzmann policy form a fixed point that is iterated a
few times per timestep. The Bellman target uses the mean-field *soft* value:

    y_i = r_i + gamma * sum_{a} pi(a | o_i', a_bar_i') * Q_target(o_i', a, a_bar_i') ,
    pi(a | o, a_bar) = softmax_a( Q(o, a, a_bar) / tau ) .

Design choices for the SALT population-control setting
------------------------------------------------------
* We use a **global** mean field: the neighbourhood N(i) is the set of all
  currently-active agents. This is the natural regime for the homogeneous,
  large-population fleet SALT targets, and it is exactly the regime mean-field
  methods are designed for (as reviewer R1 requested, "despite the potential
  violation of certain assumptions"). Mean actions are averaged over agents that
  have not yet reached a target.
* The team objective is decomposed into per-agent rewards (each agent pays its
  own step cost / earns its own goal bonus, and the terminal optimal-transport
  cost is split via the same Hungarian assignment used everywhere else). The sum
  of per-agent rewards equals the team reward used by VDN/QMIX/QPLEX, so MFQ is
  trained on the same environment and evaluated with the same metrics.

The environment dynamics, noise model, observation layout, and evaluation
metrics mirror `vdn.py` exactly so the comparison stays apples-to-apples.
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

_ACTIONS_ARR = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)], dtype=np.int32)
MFQ_OBS_MODES = {"relative_targets"}

# The small tensor workloads here are faster with bounded intra-op parallelism.
torch.set_num_threads(min(torch.get_num_threads(), 8))


@dataclass
class MFQConfig:
    gamma: float = 0.98
    hidden_size: int = 64
    lr: float = 3e-4
    batch_episodes: int = 64
    n_batches: int = 500
    target_update_every: int = 50
    obs_mode: str = "relative_targets"
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
    # Early stopping (0 / disabled by default), mirroring MAPPO/Sep-PPO.
    # Plateau stop: reach-rate (and, if max_delta_mean_reward>0, reward) moving
    # averages flat for `patience` consecutive batches. Target stop: training
    # reach rate sustained >= reach_target for `plateau_window` consecutive
    # batches (a "you've hit the milestone, stop" success criterion).
    early_stop_patience_batches: int = 0
    early_stop_plateau_window_batches: int = 0
    early_stop_max_delta_reach_rate: float = 0.0
    early_stop_max_delta_mean_reward: float = 0.0
    early_stop_reach_target: float = 0.0
    # Plateau stop only fires once the reach-rate MA is at least this high; guards
    # against a false stop during early, flat-but-low exploration. 0 disables.
    early_stop_min_reach: float = 0.0


class MFQQNet(nn.Module):
    """Shared per-agent Q-net taking [obs, mean_action] and returning Q(., a)."""

    def __init__(self, obs_dim: int, n_actions: int = 5, hidden: int = 64):
        super().__init__()
        self.n_actions = n_actions
        # Input = local observation concatenated with the mean-action vector.
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor, mean_action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, mean_action], dim=-1))


class MFQReplayBuffer:
    """Ring-buffer replay. Mean actions are stored alongside obs (baked at
    collection time), following the standard MFQ implementation."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, n_act: int):
        self.capacity = int(max(1, capacity))
        self.size = 0
        self.ptr = 0
        self.obs = np.empty((self.capacity, n_agents, obs_dim), dtype=np.float32)
        self.mean_a = np.empty((self.capacity, n_agents, n_act), dtype=np.float32)
        self.acts = np.empty((self.capacity, n_agents), dtype=np.int64)
        self.rew = np.empty((self.capacity, n_agents), dtype=np.float32)
        self.mask = np.empty((self.capacity, n_agents), dtype=np.float32)
        self.next_obs = np.empty((self.capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_mean_a = np.empty((self.capacity, n_agents, n_act), dtype=np.float32)
        self.next_mask = np.empty((self.capacity, n_agents), dtype=np.float32)
        self.not_done = np.empty((self.capacity, n_agents), dtype=np.float32)

    def add_batch(self, **kw: np.ndarray) -> None:
        n = int(kw["obs"].shape[0])
        idx = (np.arange(n, dtype=np.int64) + self.ptr) % self.capacity
        self.obs[idx] = kw["obs"]
        self.mean_a[idx] = kw["mean_a"]
        self.acts[idx] = kw["acts"]
        self.rew[idx] = kw["rew"]
        self.mask[idx] = kw["mask"]
        self.next_obs[idx] = kw["next_obs"]
        self.next_mean_a[idx] = kw["next_mean_a"]
        self.next_mask[idx] = kw["next_mask"]
        self.not_done[idx] = kw["not_done"]
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = int(min(self.capacity, self.size + n))

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        bs = int(max(1, batch_size))
        idx = rng.integers(0, self.size, size=bs, dtype=np.int64)
        return {
            "obs": self.obs[idx],
            "mean_a": self.mean_a[idx],
            "acts": self.acts[idx],
            "rew": self.rew[idx],
            "mask": self.mask[idx],
            "next_obs": self.next_obs[idx],
            "next_mean_a": self.next_mean_a[idx],
            "next_mask": self.next_mask[idx],
            "not_done": self.not_done[idx],
        }


def _validate_obs_mode(obs_mode: str) -> str:
    if obs_mode not in MFQ_OBS_MODES:
        raise ValueError(
            f"Invalid MFQ obs mode '{obs_mode}'. "
            f"Expected one of {sorted(MFQ_OBS_MODES)}."
        )
    return obs_mode


def _vec_slip(rng: np.random.Generator, acts: np.ndarray, n_act: int = 5) -> np.ndarray:
    off = rng.integers(0, n_act - 1, size=len(acts)).astype(np.int32)
    return off + (off >= acts).astype(np.int32)


def _geom_decay(ep: int, v_start: float, v_end: float, decay_eps: int) -> float:
    if ep >= decay_eps:
        return float(v_end)
    rate = (v_end / max(v_start, 1e-8)) ** (1.0 / max(1, decay_eps))
    return float(max(v_start * rate ** ep, v_end))


def _fill_obs(
    obs_l: np.ndarray, pos_r: np.ndarray, pos_c: np.ndarray, aid: np.ndarray,
    reached: np.ndarray, tgt_r: np.ndarray, tgt_c: np.ndarray, tgt_active: np.ndarray,
    t: int, grid: GridConfig, H: int, M: int,
) -> None:
    """Populate an (E, N, obs_dim) observation buffer in place (relative targets)."""
    obs_l[:, :, 0] = pos_r / max(1, grid.h - 1)
    obs_l[:, :, 1] = pos_c / max(1, grid.w - 1)
    obs_l[:, :, 2] = aid
    obs_l[:, :, 3] = t / max(1, H - 1)
    obs_l[:, :, 4] = (~reached).astype(np.float32)
    obs_l[:, :, 5:5 + 3 * M:3] = (tgt_r[:, None, :] - pos_r[:, :, None]) / max(1, grid.h - 1)
    obs_l[:, :, 6:5 + 3 * M:3] = (tgt_c[:, None, :] - pos_c[:, :, None]) / max(1, grid.w - 1)
    obs_l[:, :, 7:5 + 3 * M:3] = tgt_active[:, None, :].astype(np.float32)


def _meanfield_policy(
    q_net: MFQQNet, obs_flat: torch.Tensor, active_f: np.ndarray,
    E: int, N: int, n_act: int, tau: float, iters: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterate the global mean-action / Boltzmann-policy fixed point.

    Returns (policy, mean_action) where policy is (E, N, n_act) and mean_action
    is the (E, n_act) global mean broadcast back to (E, N, n_act). The mean is
    taken over active agents only.
    """
    active_t = torch.from_numpy(active_f.reshape(E, N, 1))
    n_active = active_t.sum(dim=1).clamp_min(1.0)  # (E, 1)
    # Warm start: uniform mean action.
    mean_a = torch.full((E, n_act), 1.0 / n_act, dtype=torch.float32)
    policy = None
    with torch.no_grad():
        for _ in range(max(1, iters)):
            ma_broadcast = mean_a[:, None, :].expand(E, N, n_act).reshape(E * N, n_act)
            q = q_net(obs_flat, ma_broadcast).reshape(E, N, n_act)
            policy = F.softmax(q / max(tau, 1e-6), dim=-1)
            # Global mean over active agents.
            mean_a = (policy * active_t).sum(dim=1) / n_active
    pol_np = policy.numpy().astype(np.float32)
    mean_np = mean_a[:, None, :].expand(E, N, n_act).numpy().astype(np.float32)
    return pol_np, mean_np


def train_mfq(
    grid: GridConfig,
    noise_cfg: NoiseConfig,
    n_agents: int,
    cfg: MFQConfig | None = None,
    seed: int = 0,
    log_every: int = 10,
    print_every: int = 1000,
    n_targets: int | None = None,
    wb_run=None,
    wb_prefix: str = "train_mfq",
) -> Tuple[MFQQNet, Dict[str, Any]]:
    if cfg is None:
        cfg = MFQConfig()
    obs_mode = _validate_obs_mode(cfg.obs_mode)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n_act = len(ACTIONS)
    H, N, E = grid.horizon, n_agents, cfg.batch_episodes
    M = n_targets if n_targets is not None else N
    obs_dim = 5 + 3 * M

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

        # The network is frozen during collection, so the mean-field solution for
        # the next state (computed at step t) equals the current-state solution at
        # step t+1 (obs == previous next_obs). Cache and reuse it to halve the
        # number of _meanfield_policy forward passes. Reset each batch (tau changes
        # only between batches). This is numerically exact.
        cached_policy = None
        cached_mean_a = None
        for t in range(H):
            _fill_obs(obs_l, pos_r, pos_c, aid, reached, tgt_r, tgt_c, tgt_active, t, grid, H, M)
            active_f = (~reached).astype(np.float32)
            mask = active_f.copy()

            if cached_policy is not None:
                policy, mean_a = cached_policy, cached_mean_a
            else:
                obs_flat = torch.from_numpy(obs_l.reshape(E * N, obs_dim))
                policy, mean_a = _meanfield_policy(
                    q_net, obs_flat, active_f, E, N, n_act, tau, cfg.meanfield_iters
                )
            # Behaviour: sample from the Boltzmann mean-field policy, with a
            # small uniform-random exploration floor.
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
            _fill_obs(next_obs_l, new_r, new_c, aid, reached, tgt_r, tgt_c, tgt_active,
                      next_t, grid, H, M)
            next_active_f = (~reached).astype(np.float32)
            next_obs_flat = torch.from_numpy(next_obs_l.reshape(E * N, obs_dim))
            next_policy, next_mean_a = _meanfield_policy(
                q_net, next_obs_flat, next_active_f, E, N, n_act, tau, cfg.meanfield_iters
            )
            # This next-state solution is the current-state solution at t+1.
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
                        # Mean-field soft value under the online Boltzmann policy,
                        # bootstrapped with the target network's Q-values.
                        pi_next = F.softmax(q_next_online / max(cfg.tau_end, 1e-6), dim=-1)
                    else:
                        pi_next = F.softmax(q_next / max(cfg.tau_end, 1e-6), dim=-1)
                    # Mask the bootstrap for agents that have already reached
                    # (absorbing) in the next state: their future value is 0.
                    v_next = (pi_next * q_next).sum(dim=2) * b_next_mask
                    td_target = b_rew + cfg.gamma * b_not_done * v_next

                # Only update Q for agents that acted (were active) at this step.
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
                f"[MFQ-{noise_cfg.kind}|{obs_mode}] batch {batch_idx+1:4d}/{cfg.n_batches}"
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

        # ---- Early stopping ----
        stop_now = False
        # (1) Target reached: training reach rate sustained at/above the target.
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
        # (2) Plateau: reach-rate (and optional reward) moving averages flat.
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
                    reward_flat = True  # reward gate disabled -> reach-only plateau
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
            print(f"[MFQ-{noise_cfg.kind}|{obs_mode}] early stop at batch "
                  f"{batch_idx + 1}: {stop_reason}")
            break

    q_net.obs_mode = obs_mode
    q_net.n_targets = M
    q_net.meanfield_iters = cfg.meanfield_iters
    q_net.tau_eval = cfg.tau_end
    batches_run = len(reach_log)
    return q_net, {
        "algorithm": "MFQ",
        "episodes": batches_run * E,
        "n_agents": N,
        "n_targets": M,
        "noise_kind": noise_cfg.kind,
        "p": noise_cfg.p,
        "mfq_config": asdict(cfg),
        "obs_mode": obs_mode,
        "env_interactions": batches_run * E * N * H,
        "batches_run": batches_run,
        "early_stopped": bool(early_stopped),
        "early_stop_reason": stop_reason,
        "mean_reward_last10": float(np.mean(batch_rewards_log[-10:])),
        "train_curve": train_curve,
    }


def rollout_mfq(
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
        obs_mode = getattr(q_net, "obs_mode", "relative_targets")
    obs_mode = _validate_obs_mode(obs_mode)
    obs_dim = 5 + 3 * M
    p_noise = float(noise_cfg.p)
    kind = noise_cfg.kind
    mf_iters = int(getattr(q_net, "meanfield_iters", 2))
    tau_eval = float(getattr(q_net, "tau_eval", 0.5))

    q_net.eval()
    for t in range(H):
        obs_l = np.empty((1, N, obs_dim), dtype=np.float32)
        _fill_obs(obs_l, pos_r[None, :], pos_c[None, :], aid, reached[None, :],
                  tgt_r[None, :], tgt_c[None, :], tgt_active[None, :], t, grid, H, M)
        active_f = (~reached).astype(np.float32)[None, :]
        obs_flat = torch.from_numpy(obs_l.reshape(N, obs_dim))
        policy, _ = _meanfield_policy(
            q_net, obs_flat, active_f, 1, N, n_act, tau_eval, mf_iters
        )
        # Greedy w.r.t. the mean-field-conditioned Q at deployment.
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
        "algorithm": "MFQ",
        "noise_kind": noise_cfg.kind,
        "p": noise_cfg.p,
        "obs_mode": obs_mode,
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
