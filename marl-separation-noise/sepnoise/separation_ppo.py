from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import GridConfig, SingleAgentGoalGrid, Pos
from .noise import NoiseConfig

# Cap intra-op threads for small tensor workloads.
torch.set_num_threads(min(torch.get_num_threads(), 8))
_ACTIONS_ARR = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)], dtype=np.int32)


@dataclass
class SepPPOConfig:
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    batch_episodes: int = 64
    n_batches: int = 500
    eps_start: float = 0.5
    eps_end: float = 0.05
    eps_decay_episodes: int = 20_000


class _Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 5, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(obs))

    def sample(self, obs: torch.Tensor):
        dist = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.forward(obs)
        return dist.log_prob(actions), dist.entropy()


class _Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def _obs_sep(s: Pos, z: Pos, t: int, grid: GridConfig) -> np.ndarray:
    sr, sc = s
    zr, zc = z
    manh = abs(sr - zr) + abs(sc - zc)
    max_manh = max(1, (grid.h - 1) + (grid.w - 1))
    norm_dist = float(manh) / float(max_manh)
    return np.array([
        sr / max(1, grid.h - 1),
        sc / max(1, grid.w - 1),
        zr / max(1, grid.h - 1),
        zc / max(1, grid.w - 1),
        t / max(1, grid.horizon - 1),
        norm_dist,
    ], dtype=np.float32)


def _gae_batched(
    rewards: np.ndarray,  # (H, E)
    values: np.ndarray,   # (H, E)
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    H, E = rewards.shape
    adv = np.empty((H, E), dtype=np.float32)
    gae = np.zeros(E, dtype=np.float32)
    for t in range(H - 1, -1, -1):
        nv = values[t + 1] if t < H - 1 else np.zeros(E, dtype=np.float32)
        delta = rewards[t] + gamma * nv - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + values


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _assign_goals_by_value(
    pos: List[Pos], targets: List[Pos], t: int, grid: GridConfig, critic: _Critic,
) -> List[Pos]:
    """OT assignment using current critic value estimates as costs (-V)."""
    n = len(pos)
    m = len(targets)
    size = max(n, m)
    big = 1e6
    C = np.full((size, size), big, dtype=np.float32)

    obs_pairs: List[np.ndarray] = []
    pair_ij: List[tuple[int, int]] = []
    for i, s in enumerate(pos):
        for j, z in enumerate(targets):
            obs_pairs.append(_obs_sep(s, z, t, grid))
            pair_ij.append((i, j))

    if obs_pairs:
        with torch.no_grad():
            vals = critic(torch.from_numpy(np.asarray(obs_pairs, dtype=np.float32))).numpy()
        for (i, j), v in zip(pair_ij, vals):
            C[i, j] = -float(v)

    row_ind, col_ind = linear_sum_assignment(C)
    row_to_col = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    assigned: List[Pos] = []
    for i in range(n):
        j = row_to_col.get(i, -1)
        if 0 <= j < m:
            assigned.append(targets[j])
        else:
            dists = [_manhattan(pos[i], z) for z in targets]
            assigned.append(targets[int(np.argmin(dists))])
    return assigned


def _epsilon_by_episode(ep: int, cfg: SepPPOConfig) -> float:
    """Exponential epsilon schedule (matches separation-q style)."""
    if ep >= cfg.eps_decay_episodes:
        return float(cfg.eps_end)
    rate = (cfg.eps_end / max(cfg.eps_start, 1e-8)) ** (
        1.0 / max(1, cfg.eps_decay_episodes)
    )
    return float(max(cfg.eps_start * rate ** ep, cfg.eps_end))


class GoalConditionedPPOPolicy:
    def __init__(self, actor: _Actor, h: int, w: int):
        self.actor = actor
        self.h = int(h)
        self.w = int(w)
        self.obs_dim = 6

    def act_greedy(self, s: Pos, z: Pos, t: int, grid: GridConfig) -> int:
        obs = _obs_sep(s, z, t, grid)[None, :]
        with torch.no_grad():
            logits = self.actor.net(torch.from_numpy(obs))
            return int(logits.argmax(-1).item())

    def save(self, path: str):
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "h": self.h,
            "w": self.w,
            "obs_dim": self.obs_dim,
        }, path)

    @classmethod
    def load(cls, path: str, hidden_size: int = 64) -> "GoalConditionedPPOPolicy":
        payload = torch.load(path, map_location="cpu")
        actor = _Actor(obs_dim=int(payload["obs_dim"]), hidden=hidden_size)
        actor.load_state_dict(payload["actor_state_dict"])
        actor.eval()
        return cls(actor=actor, h=int(payload["h"]), w=int(payload["w"]))


def train_goal_ppo(
    grid: GridConfig,
    noise_cfg: NoiseConfig,
    n_agents: int,
    cfg: SepPPOConfig | None = None,
    seed: int = 0,
    log_every: int = 10,
    print_every: int = 1000,
    wb_run=None,
    wb_prefix: str = "train_sep_ppo",
) -> tuple[GoalConditionedPPOPolicy, Dict[str, Any]]:
    if cfg is None:
        cfg = SepPPOConfig()

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    obs_dim = 6
    H = grid.horizon
    E = cfg.batch_episodes
    N = n_agents
    actor = _Actor(obs_dim=obs_dim, hidden=cfg.hidden_size)
    critic = _Critic(obs_dim=obs_dim, hidden=cfg.hidden_size)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    rewards_log: List[float] = []
    reach_log: List[float] = []
    if wb_run is not None:
        p = wb_prefix
        wb_run.define_metric(f"{p}/iter")
        wb_run.define_metric(f"{p}/*", step_metric=f"{p}/iter")

    for batch_idx in range(cfg.n_batches):
        s_obs = np.empty((H, E, N, obs_dim), dtype=np.float32)
        s_acts = np.empty((H, E, N), dtype=np.int64)
        s_logp = np.empty((H, E, N), dtype=np.float32)
        s_rew = np.empty((H, E, N), dtype=np.float32)
        s_val = np.empty((H, E, N), dtype=np.float32)
        s_mask = np.empty((H, E, N), dtype=np.float32)

        ep_rewards = np.zeros((E, N), dtype=np.float32)
        ep_reached = np.zeros((E, N), dtype=np.float32)

        starts_r = rng.integers(0, grid.h, size=(E, N)).astype(np.int32)
        starts_c = np.zeros((E, N), dtype=np.int32)
        targets_r = rng.integers(0, grid.h, size=(E, N)).astype(np.int32)
        col_lo = max(0, grid.w - 10)
        targets_c = rng.integers(col_lo, grid.w, size=(E, N)).astype(np.int32)
        goals_r = np.empty((E, N), dtype=np.int32)
        goals_c = np.empty((E, N), dtype=np.int32)

        for e in range(E):
            episode_idx = batch_idx * E + e
            eps = _epsilon_by_episode(episode_idx, cfg)
            starts = [(int(starts_r[e, i]), 0) for i in range(N)]
            targets = [(int(targets_r[e, j]), int(targets_c[e, j])) for j in range(N)]
            if rng.random() < eps:
                perm = rng.permutation(len(targets))
                goals = [targets[int(j)] for j in perm[:N]]
            else:
                goals = _assign_goals_by_value(starts, targets, t=0, grid=grid, critic=critic)
            goals_r[e] = np.asarray([g[0] for g in goals], dtype=np.int32)
            goals_c[e] = np.asarray([g[1] for g in goals], dtype=np.int32)

        pos_r = starts_r.copy()
        pos_c = starts_c.copy()
        reached = np.zeros((E, N), dtype=bool)
        p_noise = float(noise_cfg.p)
        n_act = 5

        for t in range(H):
            mask = (~reached).astype(np.float32)

            obs = np.empty((E, N, obs_dim), dtype=np.float32)
            obs[:, :, 0] = pos_r / max(1, grid.h - 1)
            obs[:, :, 1] = pos_c / max(1, grid.w - 1)
            obs[:, :, 2] = goals_r / max(1, grid.h - 1)
            obs[:, :, 3] = goals_c / max(1, grid.w - 1)
            obs[:, :, 4] = t / max(1, grid.horizon - 1)
            manh = np.abs(pos_r - goals_r) + np.abs(pos_c - goals_c)
            obs[:, :, 5] = manh / max(1, (grid.h - 1) + (grid.w - 1))

            obs_flat = obs.reshape(E * N, obs_dim)
            with torch.no_grad():
                a_t, lp_t = actor.sample(torch.from_numpy(obs_flat))
                v_t = critic(torch.from_numpy(obs_flat))
            actions = a_t.numpy().astype(np.int64).reshape(E, N)
            logp = lp_t.numpy().astype(np.float32).reshape(E, N)
            vals = v_t.numpy().astype(np.float32).reshape(E, N)
            actions[reached] = 4

            exec_a = actions.copy().astype(np.int32)
            active = ~reached
            if noise_cfg.kind != "none" and p_noise > 0:
                idx = np.where(active.ravel())[0]
                if len(idx):
                    flat = exec_a.ravel()
                    slip = rng.random(len(idx)) < p_noise
                    si = idx[slip]
                    if len(si):
                        old = flat[si]
                        off = rng.integers(0, n_act - 1, size=len(si)).astype(np.int32)
                        flat[si] = off + (off >= old).astype(np.int32)

            new_r = pos_r.copy()
            new_c = pos_c.copy()
            dr = _ACTIONS_ARR[exec_a, 0]
            dc = _ACTIONS_ARR[exec_a, 1]
            new_r[active] = np.clip(pos_r[active] + dr[active], 0, grid.h - 1).astype(np.int32)
            new_c[active] = np.clip(pos_c[active] + dc[active], 0, grid.w - 1).astype(np.int32)

            just_reached = active & (new_r == goals_r) & (new_c == goals_c)
            still_active = active & (~just_reached)
            cost = np.zeros((E, N), dtype=np.float32)
            cost[just_reached] = -float(grid.goal_bonus)
            cost[still_active] = float(grid.step_cost)
            if t == H - 1:
                term_dist = np.abs(new_r - goals_r) + np.abs(new_c - goals_c)
                cost[still_active] += term_dist[still_active].astype(np.float32)
            rew = -cost

            s_obs[t] = obs
            s_acts[t] = actions
            s_logp[t] = logp
            s_val[t] = vals
            s_rew[t] = rew
            s_mask[t] = mask
            ep_rewards += rew

            reached = reached | just_reached
            pos_r, pos_c = new_r, new_c

        ep_reached = reached.astype(np.float32)

        s_rew_f = s_rew.reshape(H, E * N)
        s_val_f = s_val.reshape(H, E * N)
        adv, ret = _gae_batched(s_rew_f, s_val_f, cfg.gamma, cfg.gae_lambda)
        b_obs = torch.from_numpy(s_obs.reshape(H * E * N, obs_dim))
        b_acts = torch.from_numpy(s_acts.reshape(H * E * N))
        b_logp = torch.from_numpy(s_logp.reshape(H * E * N))
        b_adv = torch.from_numpy(adv.reshape(H * E * N))
        b_ret = torch.from_numpy(ret.reshape(H * E * N))
        b_mask = torch.from_numpy(s_mask.reshape(H * E * N))
        am = b_mask > 0.5
        if am.sum() > 1:
            b_adv = (b_adv - b_adv[am].mean()) / (b_adv[am].std() + 1e-8)

        actor_loss_v = np.nan
        critic_loss_v = np.nan
        entropy_v = np.nan
        ms = b_mask.sum()
        for _ in range(cfg.ppo_epochs):
            nlp, ent = actor.evaluate(b_obs, b_acts)
            ratio = torch.exp(nlp - b_logp)
            s1 = ratio * b_adv
            s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * b_adv
            a_loss = (
                -(torch.min(s1, s2) * b_mask).sum() / ms
                - cfg.entropy_coef * (ent * b_mask).sum() / ms
            )
            opt_a.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            opt_a.step()

            v = critic(b_obs)
            c_loss = ((v - b_ret).pow(2) * b_mask).sum() / ms
            opt_c.zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            opt_c.step()
            actor_loss_v = float(a_loss.item())
            critic_loss_v = float(c_loss.item())
            entropy_v = float(((ent * b_mask).sum() / ms).item())

        mean_rew = float(ep_rewards.sum(axis=1).mean())
        mean_reach = float(ep_reached.mean())
        rewards_log.append(mean_rew)
        reach_log.append(mean_reach)
        episodes_done = (batch_idx + 1) * E
        if episodes_done % print_every == 0:
            rec_rew = float(np.mean(rewards_log[-log_every:]))
            rec_reach = float(np.mean(reach_log[-log_every:]))
            print(
                f"[SepPPO-{noise_cfg.kind}] batch {batch_idx+1:4d}/{cfg.n_batches}"
                f"  episodes={episodes_done:6d}"
                f"  mean_reward={rec_rew:.2f}"
                f"  reach_rate={rec_reach:.2%}"
            )
        if (batch_idx + 1) % log_every == 0 and wb_run is not None:
            p = wb_prefix
            wb_run.log({
                f"{p}/iter": episodes_done,
                f"{p}/batch": batch_idx + 1,
                f"{p}/episodes": episodes_done,
                f"{p}/epsilon": float(_epsilon_by_episode(episodes_done - 1, cfg)),
                f"{p}/mean_reward": float(np.mean(rewards_log[-log_every:])),
                f"{p}/last_batch_reward": float(mean_rew),
                f"{p}/reach_rate": float(mean_reach),
                f"{p}/actor_loss": actor_loss_v,
                f"{p}/critic_loss": critic_loss_v,
                f"{p}/entropy": entropy_v,
            })

    policy = GoalConditionedPPOPolicy(actor=actor, h=grid.h, w=grid.w)
    metrics = {
        "algorithm": "SeparationPPO",
        "episodes": int(cfg.n_batches * E),
        "n_agents": int(N),
        "noise_kind": noise_cfg.kind,
        "p": noise_cfg.p,
        "sep_ppo_config": asdict(cfg),
        "mean_reward_last10": float(np.mean(rewards_log[-10:])),
        "mean_reach_last10": float(np.mean(reach_log[-10:])),
        "env_interactions": int(cfg.n_batches * E * N * H),
    }
    return policy, metrics
