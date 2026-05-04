"""
Independent Proximal Policy Optimization (IPPO) cooperative baseline.

Always uses relative-target observations and shared actor/critic parameters.
Unlike MAPPO, the critic is local (per-agent observation only).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import GridConfig, ACTIONS, Pos
from .noise import NoiseConfig
from .matching import terminal_ot_cost, target_coverage_rate

_ACTIONS_ARR = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)], dtype=np.int32)
IPPO_OBS_MODES = {"relative_targets"}

# The small tensor workloads here are faster with bounded intra-op parallelism.
torch.set_num_threads(min(torch.get_num_threads(), 8))


@dataclass
class IPPOConfig:
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    entropy_coef: float = 0.01
    entropy_coef_end: float = 0.001
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    batch_episodes: int = 64
    n_batches: int = 500
    obs_mode: str = "relative_targets"
    minibatch_size_actor: int = 4096
    minibatch_size_critic: int = 4096


class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 5, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(obs))

    def get_action(self, obs: torch.Tensor):
        dist = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.forward(obs)
        return dist.log_prob(actions), dist.entropy()


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def _validate_obs_mode(obs_mode: str) -> str:
    if obs_mode not in IPPO_OBS_MODES:
        raise ValueError(
            f"Invalid IPPO obs mode '{obs_mode}'. "
            f"Expected one of {sorted(IPPO_OBS_MODES)}."
        )
    return obs_mode


def _gae_batched(
    rewards: np.ndarray,
    values: np.ndarray,
    not_done: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    H, B = rewards.shape
    adv = np.empty((H, B), dtype=np.float32)
    gae = np.zeros(B, dtype=np.float32)
    for t in range(H - 1, -1, -1):
        nv = values[t + 1] if t < H - 1 else np.zeros(B, dtype=np.float32)
        nd = not_done[t]
        delta = rewards[t] + gamma * nd * nv - values[t]
        gae = delta + gamma * lam * nd * gae
        adv[t] = gae
    return adv, adv + values


def _vec_slip(rng: np.random.Generator, acts: np.ndarray, n_act: int = 5) -> np.ndarray:
    off = rng.integers(0, n_act - 1, size=len(acts)).astype(np.int32)
    return off + (off >= acts).astype(np.int32)


def _linear_decay(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(end)
    frac = min(max(step / float(total_steps - 1), 0.0), 1.0)
    return float(start + frac * (end - start))


def train_ippo(
    grid: GridConfig,
    noise_cfg: NoiseConfig,
    n_agents: int,
    cfg: IPPOConfig | None = None,
    seed: int = 0,
    log_every: int = 10,
    print_every: int = 1000,
    n_targets: int | None = None,
    wb_run=None,
    wb_prefix: str = "train_ippo",
) -> Tuple[Actor, Dict[str, Any]]:
    if cfg is None:
        cfg = IPPOConfig()
    obs_mode = _validate_obs_mode(cfg.obs_mode)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n_act = len(ACTIONS)
    H, N, E = grid.horizon, n_agents, cfg.batch_episodes
    M = n_targets if n_targets is not None else N
    obs_dim = 5 + 3 * M

    actor = Actor(obs_dim, n_act, cfg.hidden_size)
    critic = Critic(obs_dim, cfg.hidden_size)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    aid = np.arange(N, dtype=np.float32) / max(1, N - 1)
    kind = noise_cfg.kind
    p_noise = float(noise_cfg.p)
    batch_rewards_log: List[float] = []
    if wb_run is not None:
        p = wb_prefix
        wb_run.define_metric(f"{p}/iter")
        wb_run.define_metric(f"{p}/*", step_metric=f"{p}/iter")

    for batch_idx in range(cfg.n_batches):
        entropy_coef_t = _linear_decay(
            cfg.entropy_coef, cfg.entropy_coef_end, batch_idx, cfg.n_batches
        )
        pos_r = rng.integers(0, grid.h, size=(E, N)).astype(np.int32)
        pos_c = np.zeros((E, N), dtype=np.int32)
        tgt_r = rng.integers(0, grid.h, size=(E, M)).astype(np.int32)
        col_lo = max(0, grid.w // 2)
        tgt_c = rng.integers(col_lo, grid.w, size=(E, M)).astype(np.int32)
        reached = np.zeros((E, N), dtype=bool)
        tgt_active = np.ones((E, M), dtype=bool)

        s_obs = np.empty((H, E, N, obs_dim), dtype=np.float32)
        s_acts = np.empty((H, E, N), dtype=np.int64)
        s_logp = np.empty((H, E, N), dtype=np.float32)
        s_mask = np.empty((H, E, N), dtype=np.float32)
        s_rew = np.empty((H, E, N), dtype=np.float32)
        s_val = np.empty((H, E, N), dtype=np.float32)
        s_not_done = np.empty((H, E, N), dtype=np.float32)

        obs_l = np.empty((E, N, obs_dim), dtype=np.float32)
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
                act_t, lp_t = actor.get_action(torch.from_numpy(obs_flat))
                val_t = critic(torch.from_numpy(obs_flat))

            actions = act_t.numpy().reshape(E, N)
            logps = lp_t.numpy().reshape(E, N)
            values = val_t.numpy().reshape(E, N)
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

            just_reached = np.zeros((E, N), dtype=bool)
            rew_agents = np.zeros((E, N), dtype=np.float32)
            for e in range(E):
                for i in np.where(active[e])[0]:
                    ri, ci = int(new_r[e, i]), int(new_c[e, i])
                    candidates = np.where(
                        tgt_active[e] & (tgt_r[e] == ri) & (tgt_c[e] == ci)
                    )[0]
                    if len(candidates):
                        tgt_active[e, int(candidates[0])] = False
                        reached[e, i] = True
                        just_reached[e, i] = True

            still_active = active & (~just_reached)
            rew_agents[just_reached] = float(grid.goal_bonus)
            rew_agents[still_active] = -float(grid.step_cost)
            if t == H - 1:
                for e in range(E):
                    fp = [(int(new_r[e, i]), int(new_c[e, i])) for i in range(N)]
                    tl = [(int(tgt_r[e, j]), int(tgt_c[e, j])) for j in range(M)]
                    rew_agents[e, :] -= float(terminal_ot_cost(fp, tl)) / max(1, N)
            s_not_done[t] = (((t < (H - 1)) & (~reached))).astype(np.float32)

            s_obs[t] = obs_l
            s_acts[t] = actions
            s_logp[t] = logps
            s_mask[t] = mask
            s_rew[t] = rew_agents
            s_val[t] = values

            pos_r, pos_c = new_r, new_c

        adv, ret = _gae_batched(
            s_rew.reshape(H, E * N),
            s_val.reshape(H, E * N),
            s_not_done.reshape(H, E * N),
            cfg.gamma,
            cfg.gae_lambda,
        )

        b_obs = torch.from_numpy(s_obs.reshape(H * E * N, obs_dim))
        b_acts = torch.from_numpy(s_acts.reshape(H * E * N))
        b_logp = torch.from_numpy(s_logp.reshape(H * E * N))
        b_adv = torch.from_numpy(adv.reshape(H * E * N))
        b_ret = torch.from_numpy(ret.reshape(H * E * N))
        b_mask = torch.from_numpy(s_mask.reshape(H * E * N))

        am = b_mask > 0.5
        if am.sum() > 1:
            b_adv = (b_adv - b_adv[am].mean()) / (b_adv[am].std() + 1e-8)

        actor_losses: List[float] = []
        critic_losses: List[float] = []
        entropies: List[float] = []
        clip_fracs: List[float] = []
        n_samples = H * E * N
        mb_a = max(1, min(cfg.minibatch_size_actor, n_samples))
        mb_c = max(1, min(cfg.minibatch_size_critic, n_samples))
        for _ in range(cfg.ppo_epochs):
            perm_a = rng.permutation(n_samples)
            for start in range(0, n_samples, mb_a):
                idx_np = perm_a[start:start + mb_a]
                idx = torch.from_numpy(idx_np)
                mb_mask = b_mask[idx]
                ms = mb_mask.sum()
                if ms.item() <= 0:
                    continue
                nlp, ent = actor.evaluate(b_obs[idx], b_acts[idx])
                ratio = torch.exp(nlp - b_logp[idx])
                s1 = ratio * b_adv[idx]
                s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * b_adv[idx]
                clipped = ((ratio < (1 - cfg.clip_eps)) | (ratio > (1 + cfg.clip_eps))).float()
                clip_frac = (clipped * mb_mask).sum() / ms
                a_loss = (
                    -(torch.min(s1, s2) * mb_mask).sum() / ms
                    - entropy_coef_t * (ent * mb_mask).sum() / ms
                )
                opt_a.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                opt_a.step()
                actor_losses.append(float(a_loss.item()))
                entropies.append(float(((ent * mb_mask).sum() / ms).item()))
                clip_fracs.append(float(clip_frac.item()))

            perm_c = rng.permutation(n_samples)
            for start in range(0, n_samples, mb_c):
                idx_np = perm_c[start:start + mb_c]
                idx = torch.from_numpy(idx_np)
                mb_mask = b_mask[idx]
                ms = mb_mask.sum()
                if ms.item() <= 0:
                    continue
                v = critic(b_obs[idx])
                c_loss = ((v - b_ret[idx]).pow(2) * mb_mask).sum() / ms
                opt_c.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                opt_c.step()
                critic_losses.append(float(c_loss.item()))
        actor_loss_v = float(np.mean(actor_losses)) if actor_losses else float("nan")
        critic_loss_v = float(np.mean(critic_losses)) if critic_losses else float("nan")
        entropy_v = float(np.mean(entropies)) if entropies else float("nan")
        clip_frac_v = float(np.mean(clip_fracs)) if clip_fracs else float("nan")

        mean_rew = float(s_rew.sum()) / E
        reach_rate = float(reached.mean())
        batch_rewards_log.append(mean_rew)
        episodes_done = (batch_idx + 1) * E
        if episodes_done % print_every == 0:
            rec = np.mean(batch_rewards_log[-log_every:])
            print(
                f"[IPPO-{noise_cfg.kind}|{obs_mode}] batch {batch_idx+1:4d}/{cfg.n_batches}"
                f"  episodes={episodes_done:6d}"
                f"  mean_reward={rec:.2f}"
                f"  reach_rate={reach_rate:.2%}"
            )
        if (batch_idx + 1) % log_every == 0 and wb_run is not None:
            rec = np.mean(batch_rewards_log[-log_every:])
            p = wb_prefix
            wb_run.log({
                f"{p}/iter": episodes_done,
                f"{p}/batch": batch_idx + 1,
                f"{p}/episodes": episodes_done,
                f"{p}/mean_reward": float(rec),
                f"{p}/last_batch_reward": float(mean_rew),
                f"{p}/reach_rate": reach_rate,
                f"{p}/actor_loss": actor_loss_v,
                f"{p}/critic_loss": critic_loss_v,
                f"{p}/entropy": entropy_v,
                f"{p}/clip_fraction": clip_frac_v,
            })

    actor.obs_mode = obs_mode
    actor.n_targets = M
    return actor, {
        "algorithm": "IPPO",
        "episodes": cfg.n_batches * E,
        "n_agents": N,
        "n_targets": M,
        "noise_kind": noise_cfg.kind,
        "p": noise_cfg.p,
        "ippo_config": asdict(cfg),
        "obs_mode": obs_mode,
        "env_interactions": cfg.n_batches * E * N * H,
        "mean_reward_last10": float(np.mean(batch_rewards_log[-10:])),
        "entropy_coef_start": float(cfg.entropy_coef),
        "entropy_coef_end": float(cfg.entropy_coef_end),
    }


def rollout_ippo(
    actor: Actor,
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
        n_targets = int(getattr(actor, "n_targets", grid.h))
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
    arrival = [None] * N
    trajectories: List[List[Pos]] = [
        [(int(pos_r[i]), int(pos_c[i]))] for i in range(N)
    ]
    total_cost = 0.0
    aid = np.arange(N, dtype=np.float32) / max(1, N - 1)
    if obs_mode is None:
        obs_mode = getattr(actor, "obs_mode", "relative_targets")
    obs_mode = _validate_obs_mode(obs_mode)
    obs_dim = 5 + 3 * len(targets)
    p_noise = float(noise_cfg.p)
    kind = noise_cfg.kind
    tgt_r = np.array([int(r) for r, _ in targets], dtype=np.int32)
    tgt_c = np.array([int(c) for _, c in targets], dtype=np.int32)
    tgt_active = np.ones(len(targets), dtype=bool)

    actor.eval()
    for t in range(H):
        obs_l = np.empty((N, obs_dim), dtype=np.float32)
        obs_l[:, 0] = pos_r / max(1, grid.h - 1)
        obs_l[:, 1] = pos_c / max(1, grid.w - 1)
        obs_l[:, 2] = aid
        obs_l[:, 3] = t / max(1, H - 1)
        active_f = (~reached).astype(np.float32)
        obs_l[:, 4] = active_f
        obs_l[:, 5:5 + 3 * len(targets):3] = (
            (tgt_r[None, :] - pos_r[:, None]) / max(1, grid.h - 1)
        )
        obs_l[:, 6:5 + 3 * len(targets):3] = (
            (tgt_c[None, :] - pos_c[:, None]) / max(1, grid.w - 1)
        )
        obs_l[:, 7:5 + 3 * len(targets):3] = tgt_active[None, :].astype(np.float32)

        with torch.no_grad():
            actions_np = actor.net(torch.from_numpy(obs_l)).argmax(-1).numpy()

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
        "algorithm": "IPPO",
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
