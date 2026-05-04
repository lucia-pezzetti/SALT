from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from .env import SingleAgentGoalGrid, GridConfig, ACTIONS
from .noise import NoiseModel, NoiseConfig

Pos = Tuple[int, int]
Action = int

@dataclass
class QConfig:
    alpha: float = 0.25
    alpha_end: Optional[float] = 0.02
    alpha_decay_episodes: int = 20_000
    gamma: float = 0.98
    eps_start: float = 0.5
    eps_end: float = 0.05
    eps_decay_episodes: int = 20_000  # episodes over which ε decays (exponential)

class GoalConditionedTabularQ:
    """
    Q-table indexed by (s_row, s_col, z_row, z_col, action).
    We treat Q as expected **cost-to-go**, so control uses argmin_a Q.
    """
    def __init__(self, h: int, w: int, n_actions: int = 5):
        self.h, self.w, self.n_actions = h, w, n_actions
        self.Q = np.zeros((h, w, h, w, n_actions), dtype=np.float32)

    def act(self, s: Pos, z: Pos, eps: float, rng: np.random.Generator) -> Action:
        if rng.random() < eps:
            return int(rng.integers(0, self.n_actions))
        q = self.Q[s[0], s[1], z[0], z[1], :]
        return int(np.argmin(q))

    def update(
        self,
        s: Pos,
        z: Pos,
        a: Action,
        cost: float,
        s2: Pos,
        done: bool,
        cfg: QConfig,
        alpha: Optional[float] = None,
    ):
        q = self.Q[s[0], s[1], z[0], z[1], a]
        target = float(cost)
        if not done:
            target += cfg.gamma * float(np.min(self.Q[s2[0], s2[1], z[0], z[1], :]))
        lr = float(cfg.alpha if alpha is None else alpha)
        self.Q[s[0], s[1], z[0], z[1], a] = (1 - lr) * q + lr * target

    def save(self, path: str):
        np.save(path, self.Q)

    @classmethod
    def load(cls, path: str) -> "GoalConditionedTabularQ":
        Q = np.load(path)
        h, w, _, _, n_actions = Q.shape
        obj = cls(h, w, n_actions=n_actions)
        obj.Q = Q
        return obj

def epsilon_by_episode(ep: int, cfg: QConfig) -> float:
    """Exponential decay: eps_start * decay_rate^ep, clamped to eps_end."""
    if ep >= cfg.eps_decay_episodes:
        return float(cfg.eps_end)
    # decay_rate chosen so that eps_start * rate^decay_episodes = eps_end
    rate = (cfg.eps_end / max(cfg.eps_start, 1e-8)) ** (
        1.0 / max(1, cfg.eps_decay_episodes)
    )
    return float(max(cfg.eps_start * rate ** ep, cfg.eps_end))


def alpha_by_episode(ep: int, cfg: QConfig) -> float:
    """Linear alpha schedule for stable long-run tabular updates."""
    alpha_start = float(cfg.alpha)
    alpha_end = float(cfg.alpha if cfg.alpha_end is None else cfg.alpha_end)
    if cfg.alpha_decay_episodes <= 0:
        return alpha_end
    frac = min(1.0, ep / float(cfg.alpha_decay_episodes))
    return float(alpha_start + frac * (alpha_end - alpha_start))

def train_goal_q(
    env: SingleAgentGoalGrid,
    episodes: int,
    qcfg: QConfig,
    seed: int = 0,
    log_every: int = 500,
    print_every: int = 100000,
    wb_run=None,
    wb_prefix: str = "train",
) -> Tuple[GoalConditionedTabularQ, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    model = GoalConditionedTabularQ(env.grid.h, env.grid.w, n_actions=5)
    if wb_run is not None:
        p = wb_prefix
        wb_run.define_metric(f"{p}/iter")
        wb_run.define_metric(f"{p}/*", step_metric=f"{p}/iter")

    ep_costs = []
    ep_rewards = []
    for ep in range(episodes):
        s, z, _ = env.reset()
        done = False
        total = 0.0
        eps = epsilon_by_episode(ep, qcfg)
        alpha = alpha_by_episode(ep, qcfg)
        while not done:
            a = model.act(s, z, eps=eps, rng=rng)
            s2, cost, done, _ = env.step(a)
            model.update(s, z, a, cost, s2, done, qcfg, alpha=alpha)
            s = s2
            total += float(cost)
        ep_costs.append(total)
        ep_rewards.append(-float(total))

        if (ep + 1) % print_every == 0:
            mean = float(np.mean(ep_costs[-log_every:]))
            mean_reward = float(np.mean(ep_rewards[-log_every:]))
            print(
                f"[train] ep {ep+1:6d}/{episodes}  "
                f"eps={eps:.3f}  alpha={alpha:.4f}  "
                f"mean_reward={mean_reward:.3f}  mean_cost={mean:.3f}"
            )
        if (ep + 1) % log_every == 0 and wb_run is not None:
            mean = float(np.mean(ep_costs[-log_every:]))
            mean_reward = float(np.mean(ep_rewards[-log_every:]))
            ep_reward = -float(total)
            p = wb_prefix
            wb_run.log({
                f"{p}/iter": ep + 1,
                f"{p}/episode": ep + 1,
                f"{p}/epsilon": eps,
                f"{p}/alpha": alpha,
                f"{p}/mean_reward": mean_reward,
                f"{p}/episode_reward": ep_reward,
                f"{p}/mean_cost": mean,
                f"{p}/episode_cost": total,
            })

    metrics = {
        "episodes": int(episodes),
        "q_config": asdict(qcfg),
        "alpha_last": float(alpha_by_episode(max(episodes - 1, 0), qcfg)),
        "mean_cost_last_500": float(np.mean(ep_costs[-500:])) if len(ep_costs) >= 500 else float(np.mean(ep_costs)),
        "mean_reward_last_500": float(np.mean(ep_rewards[-500:])) if len(ep_rewards) >= 500 else float(np.mean(ep_rewards)),
    }
    return model, metrics


# ---------------------------------------------------------------------------
# Multi-agent Q-learning with ε-greedy optimal transport assignment
# (vectorised: all M agents processed in parallel per timestep)
# ---------------------------------------------------------------------------

# Pre-compute actions array once at module level
_ACTIONS_ARR = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)], dtype=np.int32)

def train_goal_q_multiagent(
    grid: GridConfig,
    noise_cfg: NoiseConfig,
    n_agents: int,
    episodes: int,
    qcfg: QConfig,
    seed: int = 0,
    log_every: int = 500,
    print_every: int = 100000,
    wb_run=None,
    wb_prefix: str = "train",
) -> Tuple[GoalConditionedTabularQ, Dict[str, Any]]:
    """
    Multi-agent Q-learning with ε-greedy OT assignment.

    All M agents are simulated in batch using NumPy array operations.
    Q-table is updated once per episode with concatenated experiences.
    """
    rng = np.random.default_rng(seed)
    noise = NoiseModel(noise_cfg, n_actions=len(ACTIONS))
    model = GoalConditionedTabularQ(grid.h, grid.w, n_actions=len(ACTIONS))
    n_act = len(ACTIONS)
    alpha = np.float32(qcfg.alpha)
    gamma = np.float32(qcfg.gamma)
    M = n_agents
    H = grid.horizon
    step_c = np.float32(grid.step_cost)
    bonus_c = np.float32(grid.goal_bonus)

    ep_costs: List[float] = []
    ep_rewards: List[float] = []
    if wb_run is not None:
        p = wb_prefix
        wb_run.define_metric(f"{p}/iter")
        wb_run.define_metric(f"{p}/*", step_metric=f"{p}/iter")

    for ep in range(episodes):
        eps = epsilon_by_episode(ep, qcfg)

        # --- 1. sample M starts (col 0) and M targets (last 10 cols) ---
        pos_r = rng.integers(0, grid.h, size=M).astype(np.int32)
        pos_c = np.zeros(M, dtype=np.int32)
        tgt_r = rng.integers(0, grid.h, size=M).astype(np.int32)
        col_lo = max(0, grid.w // 2)
        tgt_c = rng.integers(col_lo, grid.w, size=M).astype(np.int32)

        # --- 2. ε-greedy OT assignment ---
        if rng.random() < eps:
            perm = rng.permutation(M)
            goal_r = tgt_r[perm]
            goal_c = tgt_c[perm]
        else:
            C = np.min(
                model.Q[pos_r[:, None], pos_c[:, None],
                        tgt_r[None, :], tgt_c[None, :], :],
                axis=-1,
            )
            _, col_ind = linear_sum_assignment(C)
            goal_r = tgt_r[col_ind]
            goal_c = tgt_c[col_ind]

        # --- 3. simulate M agents in parallel with inline Q-updates ---
        reached = np.zeros(M, dtype=bool)
        total_cost = np.float32(0.0)

        for t in range(H):
            active = ~reached
            if not np.any(active):
                break

            noise.reset_timestep()
            idx = np.where(active)[0]
            na = len(idx)

            sr = pos_r[idx]; sc = pos_c[idx]
            zr = goal_r[idx]
            zc = goal_c[idx]

            # Vectorised ε-greedy action selection
            q_vals = model.Q[sr, sc, zr, zc, :]
            best = np.argmin(q_vals, axis=1).astype(np.int32)
            rand_a = rng.integers(0, n_act, size=na).astype(np.int32)
            actions = np.where(rng.random(na) < eps, rand_a, best)

            # Vectorised noise + movement
            exec_a = noise.apply_batch(actions, t=t, pos_r=sr, pos_c=sc,
                                       agent_ids=idx)
            new_r = np.clip(sr + _ACTIONS_ARR[exec_a, 0],
                            0, grid.h - 1).astype(np.int32)
            new_c = np.clip(sc + _ACTIONS_ARR[exec_a, 1],
                            0, grid.w - 1).astype(np.int32)

            # Vectorised cost computation
            just_reached = (new_r == zr) & (new_c == zc)
            last_step = (t == H - 1)

            costs = np.full(na, step_c)
            costs[just_reached] = -bonus_c
            if last_step:
                miss = ~just_reached
                costs[miss] = step_c + (
                    np.abs(new_r[miss] - zr[miss])
                    + np.abs(new_c[miss] - zc[miss])
                ).astype(np.float32)

            done_flags = just_reached | last_step

            # --- Inline Q-table update (per timestep) ---
            # Values propagate within the episode: step t+1 sees
            # the Q-table already updated by steps 0..t.
            q_cur = model.Q[sr, sc, zr, zc, actions]
            q_nxt = np.min(
                model.Q[new_r, new_c, zr, zc, :], axis=-1,
            )
            td_targets = costs + np.where(
                done_flags, np.float32(0), gamma * q_nxt
            )
            # With only M agents per batch, duplicate keys are
            # extremely rare, so np.add.at is safe here.
            np.add.at(
                model.Q,
                (sr, sc, zr, zc, actions),
                alpha * (td_targets - q_cur),
            )

            total_cost += costs.sum()

            pos_r[idx] = new_r
            pos_c[idx] = new_c
            reached[idx[just_reached]] = True

        ep_costs.append(float(total_cost))
        ep_rewards.append(float(-total_cost))

        if (ep + 1) % print_every == 0:
            mean = float(np.mean(ep_costs[-log_every:]))
            mean_reward = float(np.mean(ep_rewards[-log_every:]))
            print(
                f"[train] ep {ep+1:6d}/{episodes}  eps={eps:.3f}  "
                f"mean_reward={mean_reward:.3f}  mean_cost={mean:.3f}  "
                f"reached={int(reached.sum())}/{M}"
            )
        if (ep + 1) % log_every == 0 and wb_run is not None:
            mean = float(np.mean(ep_costs[-log_every:]))
            mean_reward = float(np.mean(ep_rewards[-log_every:]))
            p = wb_prefix
            wb_run.log({
                f"{p}/iter": ep + 1,
                f"{p}/episode": ep + 1,
                f"{p}/epsilon": eps,
                f"{p}/mean_reward": mean_reward,
                f"{p}/episode_reward": float(-total_cost),
                f"{p}/mean_cost": mean,
                f"{p}/episode_cost": float(total_cost),
                f"{p}/reached": int(reached.sum()),
                f"{p}/reach_rate": float(reached.sum()) / M,
            })

    metrics = {
        "episodes": int(episodes),
        "n_agents": int(n_agents),
        "q_config": asdict(qcfg),
        "mean_cost_last_500": (
            float(np.mean(ep_costs[-500:]))
            if len(ep_costs) >= 500
            else float(np.mean(ep_costs))
        ),
        "mean_reward_last_500": (
            float(np.mean(ep_rewards[-500:]))
            if len(ep_rewards) >= 500
            else float(np.mean(ep_rewards))
        ),
    }
    return model, metrics
