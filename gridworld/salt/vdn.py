"""
Value Decomposition Networks (VDN) cooperative MARL baseline.

Shared per-agent Q-network + additive team value decomposition:
Q_tot = sum_i Q_i(o_i, a_i).
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
from .matching import terminal_ot_cost, target_coverage_rate

_ACTIONS_ARR = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)], dtype=np.int32)
VDN_OBS_MODES = {"relative_targets"}

# The small tensor workloads here are faster with bounded intra-op parallelism.
torch.set_num_threads(min(torch.get_num_threads(), 8))


@dataclass
class VDNConfig:
    gamma: float = 0.98
    hidden_size: int = 64
    lr: float = 3e-4
    batch_episodes: int = 64
    n_batches: int = 500
    target_update_every: int = 50
    obs_mode: str = "relative_targets"
    eps_start: float = 0.5
    eps_end: float = 0.05
    eps_decay_episodes: int = 20_000
    replay_capacity: int = 100_000
    min_replay_size: int = 2_048
    batch_size: int = 512
    updates_per_batch: int = 8
    double_q: bool = True


class VDNQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 5, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class VDNReplayBuffer:
    """Simple ring-buffer replay for transition-level VDN updates."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int):
        self.capacity = int(max(1, capacity))
        self.size = 0
        self.ptr = 0
        self.obs = np.empty((self.capacity, n_agents, obs_dim), dtype=np.float32)
        self.acts = np.empty((self.capacity, n_agents), dtype=np.int64)
        self.rew = np.empty((self.capacity,), dtype=np.float32)
        self.mask = np.empty((self.capacity, n_agents), dtype=np.float32)
        self.next_obs = np.empty((self.capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_mask = np.empty((self.capacity, n_agents), dtype=np.float32)
        self.not_done = np.empty((self.capacity,), dtype=np.float32)

    def add_batch(
        self,
        obs: np.ndarray,
        acts: np.ndarray,
        rew: np.ndarray,
        mask: np.ndarray,
        next_obs: np.ndarray,
        next_mask: np.ndarray,
        not_done: np.ndarray,
    ) -> None:
        n = int(obs.shape[0])
        idx = (np.arange(n, dtype=np.int64) + self.ptr) % self.capacity
        self.obs[idx] = obs
        self.acts[idx] = acts
        self.rew[idx] = rew
        self.mask[idx] = mask
        self.next_obs[idx] = next_obs
        self.next_mask[idx] = next_mask
        self.not_done[idx] = not_done
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = int(min(self.capacity, self.size + n))

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        bs = int(max(1, batch_size))
        idx = rng.integers(0, self.size, size=bs, dtype=np.int64)
        return {
            "obs": self.obs[idx],
            "acts": self.acts[idx],
            "rew": self.rew[idx],
            "mask": self.mask[idx],
            "next_obs": self.next_obs[idx],
            "next_mask": self.next_mask[idx],
            "not_done": self.not_done[idx],
        }


def _validate_obs_mode(obs_mode: str) -> str:
    if obs_mode not in VDN_OBS_MODES:
        raise ValueError(
            f"Invalid VDN obs mode '{obs_mode}'. "
            f"Expected one of {sorted(VDN_OBS_MODES)}."
        )
    return obs_mode


def _vec_slip(rng: np.random.Generator, acts: np.ndarray, n_act: int = 5) -> np.ndarray:
    off = rng.integers(0, n_act - 1, size=len(acts)).astype(np.int32)
    return off + (off >= acts).astype(np.int32)


def _epsilon_by_episode(ep: int, cfg: VDNConfig) -> float:
    if ep >= cfg.eps_decay_episodes:
        return float(cfg.eps_end)
    rate = (cfg.eps_end / max(cfg.eps_start, 1e-8)) ** (1.0 / max(1, cfg.eps_decay_episodes))
    return float(max(cfg.eps_start * rate ** ep, cfg.eps_end))


def train_vdn(
    grid: GridConfig,
    noise_cfg: NoiseConfig,
    n_agents: int,
    cfg: VDNConfig | None = None,
    seed: int = 0,
    log_every: int = 10,
    print_every: int = 1000,
    n_targets: int | None = None,
    wb_run=None,
    wb_prefix: str = "train_vdn",
) -> Tuple[VDNQNet, Dict[str, Any]]:
    if cfg is None:
        cfg = VDNConfig()
    obs_mode = _validate_obs_mode(cfg.obs_mode)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n_act = len(ACTIONS)
    H, N, E = grid.horizon, n_agents, cfg.batch_episodes
    M = n_targets if n_targets is not None else N
    obs_dim = 5 + 3 * M

    q_net = VDNQNet(obs_dim, n_act, cfg.hidden_size)
    target_net = copy.deepcopy(q_net)
    opt = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay = VDNReplayBuffer(cfg.replay_capacity, N, obs_dim)
    total_updates = 0

    aid = np.arange(N, dtype=np.float32) / max(1, N - 1)
    kind = noise_cfg.kind
    p_noise = float(noise_cfg.p)
    batch_rewards_log: List[float] = []
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
        s_acts = np.empty((H, E, N), dtype=np.int64)
        s_rew = np.empty((H, E), dtype=np.float32)
        s_mask = np.empty((H, E, N), dtype=np.float32)
        s_next_obs = np.empty((H, E, N, obs_dim), dtype=np.float32)
        s_next_mask = np.empty((H, E, N), dtype=np.float32)
        s_not_done = np.empty((H, E), dtype=np.float32)

        obs_l = np.empty((E, N, obs_dim), dtype=np.float32)
        next_obs_l = np.empty((E, N, obs_dim), dtype=np.float32)
        for t in range(H):
            obs_l[:, :, 0] = pos_r / max(1, grid.h - 1)
            obs_l[:, :, 1] = pos_c / max(1, grid.w - 1)
            obs_l[:, :, 2] = aid
            obs_l[:, :, 3] = t / max(1, H - 1)
            active_f = (~reached).astype(np.float32)
            obs_l[:, :, 4] = active_f

            obs_l[:, :, 5:5 + 3 * M:3] = (
                (tgt_r[:, None, :] - pos_r[:, :, None]) / max(1, grid.h - 1)
            )
            obs_l[:, :, 6:5 + 3 * M:3] = (
                (tgt_c[:, None, :] - pos_c[:, :, None]) / max(1, grid.w - 1)
            )
            obs_l[:, :, 7:5 + 3 * M:3] = tgt_active[:, None, :].astype(np.float32)

            mask = (~reached).astype(np.float32)
            obs_flat = obs_l.reshape(E * N, obs_dim)
            with torch.no_grad():
                qvals = q_net(torch.from_numpy(obs_flat)).numpy().reshape(E, N, n_act)
            greedy = qvals.argmax(-1).astype(np.int64)
            eps = _epsilon_by_episode(batch_idx * E, cfg)
            rand_a = rng.integers(0, n_act, size=(E, N), dtype=np.int64)
            use_rand = rng.random((E, N)) < eps
            actions = np.where(use_rand, rand_a, greedy).astype(np.int64)
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

            n_just = np.zeros(E, dtype=np.int32)
            n_still = np.zeros(E, dtype=np.int32)
            for e in range(E):
                for i in np.where(active[e])[0]:
                    ri, ci = int(new_r[e, i]), int(new_c[e, i])
                    candidates = np.where(
                        tgt_active[e] & (tgt_r[e] == ri) & (tgt_c[e] == ci)
                    )[0]
                    if len(candidates):
                        tgt_active[e, int(candidates[0])] = False
                        reached[e, i] = True
                        n_just[e] += 1
                    else:
                        n_still[e] += 1

            rew = (-float(grid.step_cost) * n_still + float(grid.goal_bonus) * n_just).astype(np.float32)
            if t == H - 1:
                for e in range(E):
                    fp = [(int(new_r[e, i]), int(new_c[e, i])) for i in range(N)]
                    tl = [(int(tgt_r[e, j]), int(tgt_c[e, j])) for j in range(M)]
                    rew[e] -= terminal_ot_cost(fp, tl)

            next_t = min(t + 1, H - 1)
            next_obs_l[:, :, 0] = new_r / max(1, grid.h - 1)
            next_obs_l[:, :, 1] = new_c / max(1, grid.w - 1)
            next_obs_l[:, :, 2] = aid
            next_obs_l[:, :, 3] = next_t / max(1, H - 1)
            next_active_f = (~reached).astype(np.float32)
            next_obs_l[:, :, 4] = next_active_f
            next_obs_l[:, :, 5:5 + 3 * M:3] = (
                (tgt_r[:, None, :] - new_r[:, :, None]) / max(1, grid.h - 1)
            )
            next_obs_l[:, :, 6:5 + 3 * M:3] = (
                (tgt_c[:, None, :] - new_c[:, :, None]) / max(1, grid.w - 1)
            )
            next_obs_l[:, :, 7:5 + 3 * M:3] = tgt_active[:, None, :].astype(np.float32)

            s_obs[t] = obs_l
            s_acts[t] = actions
            s_rew[t] = rew
            s_mask[t] = mask
            s_next_obs[t] = next_obs_l
            s_next_mask[t] = (~reached).astype(np.float32)
            all_reached = reached.all(axis=1)
            s_not_done[t] = (((t < (H - 1)) & (~all_reached)).astype(np.float32))
            pos_r, pos_c = new_r, new_c

        B = H * E
        replay.add_batch(
            obs=s_obs.reshape(B, N, obs_dim),
            acts=s_acts.reshape(B, N),
            rew=s_rew.reshape(B),
            mask=s_mask.reshape(B, N),
            next_obs=s_next_obs.reshape(B, N, obs_dim),
            next_mask=s_next_mask.reshape(B, N),
            not_done=s_not_done.reshape(B),
        )

        loss_v = float("nan")
        if replay.size >= max(1, cfg.min_replay_size, cfg.batch_size):
            for _ in range(max(1, cfg.updates_per_batch)):
                batch = replay.sample(cfg.batch_size, rng)
                bs = int(batch["obs"].shape[0])
                b_obs = torch.from_numpy(batch["obs"].reshape(bs * N, obs_dim))
                b_next_obs = torch.from_numpy(batch["next_obs"].reshape(bs * N, obs_dim))
                b_acts = torch.from_numpy(batch["acts"])
                b_rew = torch.from_numpy(batch["rew"])
                b_mask = torch.from_numpy(batch["mask"])
                b_next_mask = torch.from_numpy(batch["next_mask"])
                b_not_done = torch.from_numpy(batch["not_done"])

                q_all = q_net(b_obs).reshape(bs, N, n_act)
                q_taken = torch.gather(q_all, dim=2, index=b_acts.unsqueeze(-1)).squeeze(-1)
                q_tot = (q_taken * b_mask).sum(dim=1)

                with torch.no_grad():
                    if cfg.double_q:
                        q_next_online = q_net(b_next_obs).reshape(bs, N, n_act)
                        next_argmax = q_next_online.argmax(dim=2, keepdim=True)
                        q_next_target = target_net(b_next_obs).reshape(bs, N, n_act)
                        q_next_sel = torch.gather(
                            q_next_target, dim=2, index=next_argmax
                        ).squeeze(-1)
                    else:
                        q_next_target = target_net(b_next_obs).reshape(bs, N, n_act)
                        q_next_sel = q_next_target.max(dim=2).values
                    q_tot_next = (q_next_sel * b_next_mask).sum(dim=1)
                    td_target = b_rew + cfg.gamma * b_not_done * q_tot_next

                loss = F.mse_loss(q_tot, td_target)
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
        episodes_done = (batch_idx + 1) * E
        if episodes_done % print_every == 0:
            rec = float(np.mean(batch_rewards_log[-log_every:]))
            print(
                f"[VDN-{noise_cfg.kind}|{obs_mode}] batch {batch_idx+1:4d}/{cfg.n_batches}"
                f"  episodes={episodes_done:6d}"
                f"  mean_reward={rec:.2f}"
                f"  reach_rate={reach_rate:.2%}"
            )
        if (batch_idx + 1) % log_every == 0 and wb_run is not None:
            rec = float(np.mean(batch_rewards_log[-log_every:]))
            p = wb_prefix
            wb_run.log({
                f"{p}/iter": episodes_done,
                f"{p}/batch": batch_idx + 1,
                f"{p}/episodes": episodes_done,
                f"{p}/epsilon": float(_epsilon_by_episode(episodes_done - 1, cfg)),
                f"{p}/mean_reward": rec,
                f"{p}/last_batch_reward": float(mean_rew),
                f"{p}/reach_rate": reach_rate,
                f"{p}/td_loss": loss_v,
                f"{p}/replay_size": int(replay.size),
            })

    q_net.obs_mode = obs_mode
    q_net.n_targets = M
    return q_net, {
        "algorithm": "VDN",
        "episodes": cfg.n_batches * E,
        "n_agents": N,
        "n_targets": M,
        "noise_kind": noise_cfg.kind,
        "p": noise_cfg.p,
        "vdn_config": asdict(cfg),
        "obs_mode": obs_mode,
        "env_interactions": cfg.n_batches * E * N * H,
        "mean_reward_last10": float(np.mean(batch_rewards_log[-10:])),
    }


def rollout_vdn(
    q_net: VDNQNet,
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
    tgt_active = np.ones(len(targets), dtype=bool)
    arrival = [None] * N
    trajectories: List[List[Pos]] = [
        [(int(pos_r[i]), int(pos_c[i]))] for i in range(N)
    ]
    total_cost = 0.0
    aid = np.arange(N, dtype=np.float32) / max(1, N - 1)

    if obs_mode is None:
        obs_mode = getattr(q_net, "obs_mode", "relative_targets")
    obs_mode = _validate_obs_mode(obs_mode)
    obs_dim = 5 + 3 * len(targets)
    p_noise = float(noise_cfg.p)
    kind = noise_cfg.kind

    q_net.eval()
    for t in range(H):
        obs_l = np.empty((N, obs_dim), dtype=np.float32)
        obs_l[:, 0] = pos_r / max(1, grid.h - 1)
        obs_l[:, 1] = pos_c / max(1, grid.w - 1)
        obs_l[:, 2] = aid
        obs_l[:, 3] = t / max(1, H - 1)
        active_f = (~reached).astype(np.float32)
        obs_l[:, 4] = active_f

        obs_l[:, 5:5 + 3 * len(targets):3] = (tgt_r[None, :] - pos_r[:, None]) / max(1, grid.h - 1)
        obs_l[:, 6:5 + 3 * len(targets):3] = (tgt_c[None, :] - pos_c[:, None]) / max(1, grid.w - 1)
        obs_l[:, 7:5 + 3 * len(targets):3] = tgt_active[None, :].astype(np.float32)

        with torch.no_grad():
            actions_np = q_net(torch.from_numpy(obs_l)).argmax(-1).numpy()
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
        "algorithm": "VDN",
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
