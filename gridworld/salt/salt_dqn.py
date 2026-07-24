"""
Deep Double DQN (Van Hasselt et al.) for goal-conditioned separation policies.

Uses the same observation encoding as Sep-PPO / Sep-A2C. Q(s,z,·) outputs action values
for **reward** (higher is better), matching the PPO rollout sign convention.

Replay transitions use the **executed** action after action slip (same as the dynamics that
produce r and s′). Storing the pre-noise “intended” action would mismatch the Bellman backup
whenever p > 0.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import GridConfig, Pos, target_col_lo
from .noise import NoiseConfig
from .salt_ppo import _obs_sep

torch.set_num_threads(min(torch.get_num_threads(), 8))
_ACTIONS_ARR = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)], dtype=np.int32)


def _epsilon_ddqn(interactions_done: int, decay_interactions: int, cfg: SepDeepDQNConfig) -> float:
    """Exponential epsilon schedule driven by interaction count (same form as Sep-PPO)."""
    if interactions_done >= decay_interactions:
        return float(cfg.eps_end)
    rate = (cfg.eps_end / max(cfg.eps_start, 1e-8)) ** (
        1.0 / max(1, decay_interactions)
    )
    return float(max(cfg.eps_start * rate ** interactions_done, cfg.eps_end))


@dataclass
class SepDeepDQNConfig:
    gamma: float = 0.98
    lr: float = 3e-4
    hidden_size: int = 64
    max_grad_norm: float = 0.5
    batch_episodes: int = 64
    n_batches: int = 500
    replay_capacity: int = 500_000
    replay_warmup: int = 5_000
    train_steps_per_batch: int = 64
    minibatch_size: int = 256
    target_update_every: int = 500
    eps_start: float = 0.5
    eps_end: float = 0.05
    eps_decay_interactions: int | None = None
    early_stop_patience_batches: int = 0
    early_stop_min_rel_policy_update: float = 0.0
    early_stop_plateau_window_batches: int = 0
    early_stop_max_delta_reach_rate: float = 0.0
    early_stop_max_delta_mean_reward: float = 0.0


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 5, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.bool_)
        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """Add many transitions at once (each row one transition)."""
        m = obs.shape[0]
        for i in range(m):
            self._add_one(obs[i], int(a[i]), float(r[i]), next_obs[i], bool(done[i]))

    def _add_one(self, obs: np.ndarray, a: int, r: float, next_obs: np.ndarray, done: bool) -> None:
        p = self.ptr
        self.obs[p] = obs
        self.next_obs[p] = next_obs
        self.a[p] = a
        self.r[p] = r
        self.d[p] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[torch.Tensor, ...]:
        n = min(batch_size, self.size)
        idx = rng.integers(0, self.size, size=n)
        obs = torch.from_numpy(self.obs[idx])
        next_obs = torch.from_numpy(self.next_obs[idx])
        a = torch.from_numpy(self.a[idx]).long()
        r = torch.from_numpy(self.r[idx])
        d = torch.from_numpy(self.d[idx].astype(np.float32))
        return obs, a, r, next_obs, d


def _assign_goals_by_qmax(
    pos: List[Pos],
    targets: List[Pos],
    t: int,
    grid: GridConfig,
    q_online: QNetwork,
) -> List[Pos]:
    n = len(pos)
    m = len(targets)
    size = max(n, m)
    big = 1e6
    C = np.full((size, size), big, dtype=np.float32)
    pair_obs: List[np.ndarray] = []
    pair_ij: List[Tuple[int, int]] = []
    for i, s in enumerate(pos):
        for j, z in enumerate(targets):
            pair_obs.append(_obs_sep(s, z, t, grid))
            pair_ij.append((i, j))
    if pair_obs:
        obs_t = torch.from_numpy(np.asarray(pair_obs, dtype=np.float32))
        with torch.no_grad():
            qm = q_online(obs_t).max(dim=-1).values.numpy()
        for (i, j), v in zip(pair_ij, qm):
            C[i, j] = -float(v)
    row_ind, col_ind = linear_sum_assignment(C)
    row_to_col = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    assigned: List[Pos] = []
    for i in range(n):
        j = row_to_col.get(i, -1)
        if 0 <= j < m:
            assigned.append(targets[j])
        else:
            dists = [abs(pos[i][0] - z[0]) + abs(pos[i][1] - z[1]) for z in targets]
            assigned.append(targets[int(np.argmin(dists))])
    return assigned


class GoalConditionedDeepDQNPolicy:
    """Wrapper for eval (greedy) matching GoalConditionedPPOPolicy API."""

    def __init__(self, q_net: QNetwork, h: int, w: int):
        self.q_net = q_net
        self.h = int(h)
        self.w = int(w)
        self.obs_dim = 6

    def act_greedy(self, s: Pos, z: Pos, t: int, grid: GridConfig) -> int:
        obs = _obs_sep(s, z, t, grid)[None, :]
        with torch.no_grad():
            q = self.q_net(torch.from_numpy(obs))
            return int(q.argmax(-1).item())

    def save(self, path: str) -> None:
        lin0 = self.q_net.net[0]
        assert isinstance(lin0, nn.Linear)
        torch.save(
            {
                "q_online_state_dict": self.q_net.state_dict(),
                "h": self.h,
                "w": self.w,
                "obs_dim": self.obs_dim,
                "hidden_size": int(lin0.out_features),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, hidden_size: int = 64) -> "GoalConditionedDeepDQNPolicy":
        payload = torch.load(path, map_location="cpu")
        obs_dim = int(payload["obs_dim"])
        h = int(payload.get("hidden_size", hidden_size))
        q = QNetwork(obs_dim=obs_dim, hidden=h)
        q.load_state_dict(payload["q_online_state_dict"])
        q.eval()
        return cls(q_net=q, h=int(payload["h"]), w=int(payload["w"]))


def train_goal_deep_double_dqn(
    grid: GridConfig,
    noise_cfg: NoiseConfig,
    n_agents: int,
    cfg: SepDeepDQNConfig | None = None,
    seed: int = 0,
    log_every: int = 10,
    print_every: int = 1000,
    wb_run=None,
    wb_prefix: str = "train_sep_ddqn",
) -> Tuple[GoalConditionedDeepDQNPolicy, Dict[str, Any]]:
    if cfg is None:
        cfg = SepDeepDQNConfig()

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    obs_dim = 6
    n_act = 5
    H = grid.horizon
    E = cfg.batch_episodes
    N = n_agents
    total_interactions = int(cfg.n_batches * E * N * H)
    eps_decay_interactions = int(
        cfg.eps_decay_interactions
        if cfg.eps_decay_interactions is not None
        else max(1, total_interactions // 2)
    )

    q_online = QNetwork(obs_dim=obs_dim, n_actions=n_act, hidden=cfg.hidden_size)
    q_target = QNetwork(obs_dim=obs_dim, n_actions=n_act, hidden=cfg.hidden_size)
    q_target.load_state_dict(q_online.state_dict())
    opt = torch.optim.Adam(q_online.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_capacity, obs_dim)

    rewards_log: List[float] = []
    reach_log: List[float] = []
    train_curve: List[Dict[str, float]] = []
    consecutive_small_updates = 0
    early_stopped = False
    stop_reason = ""
    learner_steps = 0
    p_noise = float(noise_cfg.p)

    if wb_run is not None:
        p = wb_prefix
        wb_run.define_metric(f"{p}/iter")
        wb_run.define_metric(f"{p}/*", step_metric=f"{p}/iter")

    for batch_idx in range(cfg.n_batches):
        q_before = nn.utils.parameters_to_vector(q_online.parameters()).detach().clone()

        starts_r = rng.integers(0, grid.h, size=(E, N)).astype(np.int32)
        starts_c = np.zeros((E, N), dtype=np.int32)
        targets_r = rng.integers(0, grid.h, size=(E, N)).astype(np.int32)
        col_lo = target_col_lo(grid)
        targets_c = rng.integers(col_lo, grid.w, size=(E, N)).astype(np.int32)
        goals_r = np.empty((E, N), dtype=np.int32)
        goals_c = np.empty((E, N), dtype=np.int32)

        for e in range(E):
            episode_idx = batch_idx * E + e
            interactions_done = int(episode_idx * N * H)
            eps = _epsilon_ddqn(interactions_done, eps_decay_interactions, cfg)
            starts = [(int(starts_r[e, i]), 0) for i in range(N)]
            targets = [(int(targets_r[e, j]), int(targets_c[e, j])) for j in range(N)]
            if rng.random() < eps:
                perm = rng.permutation(len(targets))
                goals = [targets[int(j)] for j in perm[:N]]
            else:
                goals = _assign_goals_by_qmax(starts, targets, t=0, grid=grid, q_online=q_online)
            goals_r[e] = np.asarray([g[0] for g in goals], dtype=np.int32)
            goals_c[e] = np.asarray([g[1] for g in goals], dtype=np.int32)

        pos_r = starts_r.copy()
        pos_c = starts_c.copy()
        reached = np.zeros((E, N), dtype=bool)
        ep_rewards = np.zeros((E, N), dtype=np.float32)

        batch_obs: List[np.ndarray] = []
        batch_a: List[int] = []
        batch_r: List[float] = []
        batch_no: List[np.ndarray] = []
        batch_d: List[bool] = []

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
            interactions_done = int((batch_idx * E * N * H) + t * E * N)
            eps_act = _epsilon_ddqn(interactions_done, eps_decay_interactions, cfg)

            with torch.no_grad():
                qv = q_online(torch.from_numpy(obs_flat))
                greedy = qv.argmax(dim=-1).numpy().astype(np.int64).reshape(E, N)
            rand_a = rng.integers(0, n_act, size=(E, N)).astype(np.int64)
            u = rng.random((E, N))
            actions = np.where(u < eps_act, rand_a, greedy)
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

            t_next = min(t + 1, H - 1)
            obs_next = np.empty((E, N, obs_dim), dtype=np.float32)
            obs_next[:, :, 0] = new_r / max(1, grid.h - 1)
            obs_next[:, :, 1] = new_c / max(1, grid.w - 1)
            obs_next[:, :, 2] = goals_r / max(1, grid.h - 1)
            obs_next[:, :, 3] = goals_c / max(1, grid.w - 1)
            obs_next[:, :, 4] = t_next / max(1, grid.horizon - 1)
            manh2 = np.abs(new_r - goals_r) + np.abs(new_c - goals_c)
            obs_next[:, :, 5] = manh2 / max(1, (grid.h - 1) + (grid.w - 1))

            done_here = just_reached | (t == H - 1)

            # The Bellman target must use the action that actually drove the transition.
            for e in range(E):
                for n in range(N):
                    if mask[e, n] < 0.5:
                        continue
                    batch_obs.append(obs[e, n].copy())
                    batch_a.append(int(exec_a[e, n]))
                    batch_r.append(float(rew[e, n]))
                    batch_no.append(obs_next[e, n].copy())
                    batch_d.append(bool(done_here[e, n]))

            ep_rewards += rew
            reached = reached | just_reached
            pos_r, pos_c = new_r, new_c

        ep_reached = reached.astype(np.float32)

        if batch_obs:
            replay.add_batch(
                np.asarray(batch_obs, dtype=np.float32),
                np.asarray(batch_a, dtype=np.int64),
                np.asarray(batch_r, dtype=np.float32),
                np.asarray(batch_no, dtype=np.float32),
                np.asarray(batch_d, dtype=np.bool_),
            )

        mean_rew = float(ep_rewards.sum(axis=1).mean())
        mean_reach = float(ep_reached.mean())
        rewards_log.append(mean_rew)
        reach_log.append(mean_reach)

        actor_loss_v = float("nan")
        for _ in range(cfg.train_steps_per_batch):
            if replay.size < max(cfg.replay_warmup, cfg.minibatch_size):
                break
            o, a, r, no, d = replay.sample(cfg.minibatch_size, rng)
            q_sa = q_online(o).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_a = q_online(no).argmax(dim=1, keepdim=True)
                next_q = q_target(no).gather(1, next_a).squeeze(1)
                y = r + cfg.gamma * (1.0 - d) * next_q
            loss = F.smooth_l1_loss(q_sa, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_online.parameters(), cfg.max_grad_norm)
            opt.step()
            learner_steps += 1
            actor_loss_v = float(loss.item())
            if learner_steps % cfg.target_update_every == 0:
                q_target.load_state_dict(q_online.state_dict())

        q_after = nn.utils.parameters_to_vector(q_online.parameters()).detach()
        delta_norm = float(torch.linalg.vector_norm(q_after - q_before).item())
        base_norm = float(torch.linalg.vector_norm(q_before).item())
        rel_policy_update = delta_norm / max(base_norm, 1e-12)

        episodes_done = (batch_idx + 1) * E
        interactions_done = int((batch_idx + 1) * E * N * H)
        train_curve.append(
            {
                "batch": float(batch_idx + 1),
                "episodes": float(episodes_done),
                "env_interactions": float(interactions_done),
                "reach_rate": float(mean_reach),
                "rel_policy_update": float(rel_policy_update),
            }
        )

        if episodes_done % print_every == 0:
            rec_rew = float(np.mean(rewards_log[-log_every:]))
            rec_reach = float(np.mean(reach_log[-log_every:]))
            print(
                f"[SepDDQN-{noise_cfg.kind}] batch {batch_idx+1:4d}/{cfg.n_batches}"
                f"  episodes={episodes_done:6d}"
                f"  mean_reward={rec_rew:.2f}"
                f"  reach_rate={rec_reach:.2%}"
                f"  replay={replay.size}"
            )
        if (batch_idx + 1) % log_every == 0 and wb_run is not None:
            p = wb_prefix
            wb_run.log({
                f"{p}/iter": interactions_done,
                f"{p}/batch": batch_idx + 1,
                f"{p}/episodes": episodes_done,
                f"{p}/env_interactions": interactions_done,
                f"{p}/mean_reward": float(np.mean(rewards_log[-log_every:])),
                f"{p}/reach_rate": float(mean_reach),
                f"{p}/rel_policy_update": float(rel_policy_update),
                f"{p}/replay_size": replay.size,
                f"{p}/learner_steps": learner_steps,
                f"{p}/td_loss": actor_loss_v,
            })

        if cfg.early_stop_patience_batches > 0:
            small_update_ok = (
                cfg.early_stop_min_rel_policy_update <= 0.0
                or rel_policy_update < cfg.early_stop_min_rel_policy_update
            )
            plateau_ok = True
            if cfg.early_stop_plateau_window_batches > 0:
                w = cfg.early_stop_plateau_window_batches
                if len(reach_log) >= 2 * w and len(rewards_log) >= 2 * w:
                    prev_reach = float(np.mean(reach_log[-2 * w : -w]))
                    curr_reach = float(np.mean(reach_log[-w:]))
                    prev_rew = float(np.mean(rewards_log[-2 * w : -w]))
                    curr_rew = float(np.mean(rewards_log[-w:]))
                    plateau_ok = (
                        abs(curr_reach - prev_reach) <= cfg.early_stop_max_delta_reach_rate
                        and abs(curr_rew - prev_rew) <= cfg.early_stop_max_delta_mean_reward
                    )
                else:
                    plateau_ok = False

            if small_update_ok and plateau_ok:
                consecutive_small_updates += 1
            else:
                consecutive_small_updates = 0

            if consecutive_small_updates >= cfg.early_stop_patience_batches:
                early_stopped = True
                thr = cfg.early_stop_min_rel_policy_update
                stop_reason = (
                    f"stable for {cfg.early_stop_patience_batches} batches "
                    f"(rel_update<={thr if thr > 0 else 'disabled'})"
                )
                print(f"[SepDDQN-{noise_cfg.kind}] early stop at batch {batch_idx + 1}: {stop_reason}")
                break

    policy = GoalConditionedDeepDQNPolicy(q_net=q_online, h=grid.h, w=grid.w)
    executed_batches = len(train_curve)
    executed_episodes = int(executed_batches * E)
    executed_interactions = int(executed_batches * E * N * H)
    metrics = {
        "algorithm": "SeparationDeepDoubleDQN",
        "episodes": executed_episodes,
        "n_agents": int(N),
        "noise_kind": noise_cfg.kind,
        "p": noise_cfg.p,
        "sep_deep_dqn_config": asdict(cfg),
        "mean_reward_last10": float(np.mean(rewards_log[-10:])),
        "mean_reach_last10": float(np.mean(reach_log[-10:])),
        "env_interactions": executed_interactions,
        "eps_decay_interactions": eps_decay_interactions,
        "planned_batches": int(cfg.n_batches),
        "executed_batches": int(executed_batches),
        "learner_steps": int(learner_steps),
        "early_stopped": bool(early_stopped),
        "early_stop_reason": stop_reason,
        "train_curve": train_curve,
    }
    return policy, metrics
