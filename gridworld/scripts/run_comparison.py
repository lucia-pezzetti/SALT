#!/usr/bin/env python3
"""
Compare  Separation Principle  vs  MAPPO  on the multi-agent gridworld
under different noise correlation structures (individual / local / global).

Every evaluation episode uses N agents and N targets randomly sampled
(with repetition) from the last half of columns. Both algorithms are
evaluated on the **same** target configuration per seed for fairness.

Usage
-----
    python scripts/run_comparison.py --p 0.15 --agents 7

Outputs
-------
    runs/comparison/<stamp>/
        comparison.json
        <noise_kind>/
            sep_metrics.json
            mappo_dynamic_metrics.json
            mappo_fixed_targets_metrics.json
            mappo_full_metrics.json
            terminal_hist_sep.png
            terminal_hist_mappo_dynamic.png
            terminal_hist_mappo_fixed_targets.png
            terminal_hist_mappo_full.png
"""
from __future__ import annotations

import argparse, os, json, time, sys, subprocess
from dataclasses import replace
from datetime import datetime
from typing import List, Any, Dict

import numpy as np

from salt.env import GridConfig, MultiAgentGrid, Pos
from salt.noise import NoiseConfig
from salt.q_learning import (
    GoalConditionedTabularQ,
    QConfig,
    train_goal_q_multiagent,
)
from salt.salt_dqn import (
    GoalConditionedDeepDQNPolicy,
    SepDeepDQNConfig,
    train_goal_deep_double_dqn,
)
from salt.salt_ppo import (
    GoalConditionedPPOPolicy,
    SepPPOConfig,
    train_goal_ppo,
    train_goal_a2c,
)
from salt.matching import assign_goals, terminal_ot_cost, target_coverage_rate
from salt.mappo import (
    MAPPOConfig,
    train_mappo,
    rollout_mappo,
)
from salt.ippo import (
    IPPOConfig,
    train_ippo,
    rollout_ippo,
)
from salt.happo import (
    HAPPOConfig,
    train_happo,
    rollout_happo,
)
from salt.vdn import (
    VDNConfig,
    train_vdn,
    rollout_vdn,
)
from salt.qmix import (
    QMIXConfig,
    train_qmix,
    rollout_qmix,
)
from salt.mfq import (
    MFQConfig,
    train_mfq,
    rollout_mfq,
)
from salt.mfq_local import (
    MFQLocalConfig,
    train_mfq_local,
    rollout_mfq_local,
)
from salt.viz import plot_terminal_hist


def _sample_targets(
    seed: int,
    n: int,
    h: int,
    w: int,
    region: str = "last_half",
) -> List[Pos]:
    """Sample *n* random targets from the requested grid region."""
    rng = np.random.default_rng(seed + 50_000)
    rows = rng.integers(0, h, size=n)
    if region == "last_ten":
        col_lo = max(0, w - 10)
        cols = rng.integers(col_lo, w, size=n)
    elif region == "last_half":
        col_lo = max(0, w // 2)
        cols = rng.integers(col_lo, w, size=n)
    elif region == "full_grid":
        cols = rng.integers(0, w, size=n)
    else:
        raise ValueError(f"Unknown target sampling region: {region}")
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


_EVAL_KEYS = [
    "terminal_ot_cost", "target_coverage", "reach_rate",
    "mean_time_to_reach", "mean_time_to_reach_including_unreached", "total_cost",
]


def _aggregate(results: List[dict]) -> dict:
    """Compute mean ± std for key metrics over multiple rollout seeds."""
    agg: dict = {}
    for k in _EVAL_KEYS:
        vals = [
            r[k] for r in results
            if not (isinstance(r[k], float) and r[k] != r[k])  # skip NaN
        ]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        else:
            agg[f"{k}_mean"] = float("nan")
            agg[f"{k}_std"] = float("nan")
    return agg


def _fmt(val: float, std: float, prec: int = 1) -> str:
    """Format  mean±std."""
    if val != val:  # NaN
        return "N/A"
    return f"{val:.{prec}f}±{std:.{prec}f}"


def _parse_reach_thresholds(s: str) -> List[float]:
    vals = [v.strip() for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("At least one reach-rate threshold must be provided.")
    out: List[float] = []
    for v in vals:
        thr = float(v)
        if thr < 0.0 or thr > 1.0:
            raise ValueError(f"Reach-rate threshold must be in [0,1], got {thr}.")
        out.append(thr)
    return out


def _stable_reach_milestones(
    train_curve: List[dict],
    thresholds: List[float],
    stability_window: int,
) -> Dict[str, dict]:
    if stability_window <= 0:
        raise ValueError("stability_window must be positive.")

    # Fixed string keys keep downstream plots stable across threshold choices.
    keys = [f"{thr:.2f}" for thr in thresholds]
    out: Dict[str, dict] = {
        k: {
            "env_interactions": float("nan"),
            "epochs": float("nan"),
            "episodes": float("nan"),
        }
        for k in keys
    }
    if not train_curve:
        return out

    reaches = [float(x.get("reach_rate", float("nan"))) for x in train_curve]
    interactions = [float(x.get("env_interactions", float("nan"))) for x in train_curve]
    batches = [float(x.get("batch", float("nan"))) for x in train_curve]
    episodes = [float(x.get("episodes", float("nan"))) for x in train_curve]
    n = len(reaches)

    for thr in thresholds:
        k = f"{thr:.2f}"
        idx = None
        max_start = n - stability_window
        for i in range(max_start + 1):
            segment = reaches[i : i + stability_window]
            if all((r == r) and (r >= thr) for r in segment):
                idx = i
                break
        if idx is None:
            continue
        out[k] = {
            "env_interactions": interactions[idx],
            "epochs": batches[idx],
            "episodes": episodes[idx],
        }
    return out


def _print_eval_trace(seed: int, label: str, result: dict) -> None:
    print(f"\n[eval-trace] seed={seed} algorithm={label}")
    targets = result.get("targets", [])
    trajectories = result.get("trajectories", [])
    starts = [traj[0] for traj in trajectories] if trajectories else []
    print(f"  starts: {starts}")
    print(f"  targets: {targets}")
    print("  trajectories:")
    for i, traj in enumerate(trajectories):
        print(f"    agent[{i}]: {traj}")

    step_traces = result.get("step_traces", [])
    if not step_traces:
        print("  step_traces: <not collected>")
        return

    print("  step_traces:")
    for st in step_traces:
        print(
            f"    t={st['t']} "
            f"states_before={st['states_before']} "
            f"greedy_actions={st['greedy_actions']} "
            f"executed_actions={st['executed_actions']} "
            f"action_source={st['action_source']} "
            f"rewards_no_collision={st['rewards_no_collision']} "
            f"states_after={st['states_after']}"
        )


_RELATIVE_MODE = "relative_targets"


def _parse_relative_only_modes(s: str, label: str) -> List[str]:
    """Parse comma-separated modes and enforce relative-targets-only policy."""
    modes = [m.strip() for m in s.split(",") if m.strip()]
    if not modes:
        raise ValueError(f"At least one {label} mode must be provided.")
    out: List[str] = []
    for m in modes:
        if m not in out:
            out.append(m)
    invalid = [m for m in out if m != _RELATIVE_MODE]
    if invalid:
        raise ValueError(
            f"{label} now supports only '{_RELATIVE_MODE}'. Got: {invalid}."
        )
    return out


def _parse_sep_methods(s: str) -> List[str]:
    methods = [m.strip() for m in s.split(",") if m.strip()]
    if not methods:
        raise ValueError("At least one separation method must be provided.")
    valid = {"q", "ppo", "ddqn", "a2c"}
    unknown = [m for m in methods if m not in valid]
    if unknown:
        raise ValueError(
            f"Unknown separation method(s): {unknown}. Valid: {sorted(valid)}."
        )
    out: List[str] = []
    for m in methods:
        if m not in out:
            out.append(m)
    return out


def _parse_p_values(s: str) -> List[float]:
    vals = [v.strip() for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("At least one noise probability must be provided.")
    out: List[float] = []
    for v in vals:
        pv = float(v)
        if pv < 0.0 or pv > 1.0:
            raise ValueError(f"Noise probability must be in [0,1], got {pv}.")
        out.append(pv)
    return out


def _parse_fixed_targets(s: str, h: int, w: int) -> List[Pos]:
    """
    Parse fixed targets from "r:c;r:c;..." format.
    Example: "0:29;4:29;11:29"
    """
    items = [it.strip() for it in s.split(";") if it.strip()]
    if not items:
        raise ValueError("Fixed targets string is empty.")
    targets: List[Pos] = []
    for it in items:
        if ":" not in it:
            raise ValueError(
                f"Invalid target '{it}'. Expected 'row:col' entries separated by ';'."
            )
        r_s, c_s = it.split(":", 1)
        r, c = int(r_s), int(c_s)
        if not (0 <= r < h and 0 <= c < w):
            raise ValueError(
                f"Target ({r},{c}) out of bounds for grid {h}x{w}."
            )
        targets.append((r, c))
    return targets


def ensure_sep_q(
    grid: GridConfig, noise_cfg: NoiseConfig, q_path: str | None,
    n_agents: int, seed: int, sep_episodes: int = 250_000,
    save_dir: str | None = None, wb_run=None, wb_prefix: str = "train",
    log_every: int = 5000,
) -> tuple[GoalConditionedTabularQ, dict]:
    """Train or load a noise-specific goal-conditioned Q-table."""
    if q_path and os.path.exists(q_path):
        Q = GoalConditionedTabularQ.load(q_path)
        if Q.h == grid.h and Q.w == grid.w:
            print(f"[sep] Loaded Q from {q_path}")
            return Q, {"loaded_from": q_path}
        print("[warn] Q shape mismatch, retraining …")

    qcfg = QConfig(eps_decay_episodes=max(1, sep_episodes // 2))
    model, metrics = train_goal_q_multiagent(
        grid, noise_cfg, n_agents=n_agents,
        episodes=sep_episodes, qcfg=qcfg, seed=seed, log_every=log_every,
        wb_run=wb_run, wb_prefix=wb_prefix,
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"Q_sep_{noise_cfg.kind}.npy")
        model.save(out)
        print(f"[sep] Saved Q to {out}")
    return model, {"trained": True, "train_metrics": metrics}


def ensure_sep_ddqn(
    grid: GridConfig,
    noise_cfg: NoiseConfig,
    policy_path: str | None,
    n_agents: int,
    seed: int,
    sep_ppo_batches: int = 500,
    sep_ppo_batch_eps: int = 64,
    early_stop_patience_batches: int = 0,
    early_stop_min_rel_policy_update: float = 0.0,
    early_stop_plateau_window_batches: int = 0,
    early_stop_max_delta_reach_rate: float = 0.0,
    early_stop_max_delta_mean_reward: float = 0.0,
    save_dir: str | None = None,
    wb_run=None,
    wb_prefix: str = "train",
    log_every: int = 10,
    print_every: int = 10_000,
) -> tuple[GoalConditionedDeepDQNPolicy, dict]:
    """Train or load a noise-specific deep Double DQN policy (.pt)."""
    if policy_path and os.path.exists(policy_path):
        policy = GoalConditionedDeepDQNPolicy.load(policy_path)
        if policy.h == grid.h and policy.w == grid.w:
            print(f"[sep-ddqn] Loaded policy from {policy_path}")
            return policy, {"loaded_from": policy_path}
        print("[warn] Sep-DDQN shape mismatch, retraining …")

    total_ix = max(1, sep_ppo_batches * sep_ppo_batch_eps * n_agents * grid.horizon)
    dqcfg = SepDeepDQNConfig(
        n_batches=sep_ppo_batches,
        batch_episodes=sep_ppo_batch_eps,
        eps_decay_interactions=max(1, total_ix // 2),
        early_stop_patience_batches=early_stop_patience_batches,
        early_stop_min_rel_policy_update=early_stop_min_rel_policy_update,
        early_stop_plateau_window_batches=early_stop_plateau_window_batches,
        early_stop_max_delta_reach_rate=early_stop_max_delta_reach_rate,
        early_stop_max_delta_mean_reward=early_stop_max_delta_mean_reward,
    )
    model, metrics = train_goal_deep_double_dqn(
        grid,
        noise_cfg,
        n_agents=n_agents,
        cfg=dqcfg,
        seed=seed,
        log_every=log_every,
        print_every=print_every,
        wb_run=wb_run,
        wb_prefix=wb_prefix,
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"sep_ddqn_policy_{noise_cfg.kind}.pt")
        model.save(out)
        print(f"[sep-ddqn] Saved policy to {out}")
    return model, {"trained": True, "train_metrics": metrics}


def ensure_sep_a2c(
    grid: GridConfig,
    noise_cfg: NoiseConfig,
    policy_path: str | None,
    n_agents: int,
    seed: int,
    sep_ppo_batches: int = 500,
    sep_ppo_batch_eps: int = 64,
    early_stop_patience_batches: int = 0,
    early_stop_min_rel_policy_update: float = 0.0,
    early_stop_plateau_window_batches: int = 0,
    early_stop_max_delta_reach_rate: float = 0.0,
    early_stop_max_delta_mean_reward: float = 0.0,
    early_stop_reach_target: float = 0.0,
    save_dir: str | None = None,
    wb_run=None,
    wb_prefix: str = "train",
    log_every: int = 10,
    print_every: int = 10_000,
) -> tuple[GoalConditionedPPOPolicy, dict]:
    """Train or load a noise-specific Separation A2C policy (.pt, same format as Sep-PPO)."""
    if policy_path and os.path.exists(policy_path):
        policy = GoalConditionedPPOPolicy.load(policy_path)
        if policy.h == grid.h and policy.w == grid.w:
            print(f"[sep-a2c] Loaded policy from {policy_path}")
            return policy, {"loaded_from": policy_path}
        print("[warn] Sep A2C shape mismatch, retraining …")

    a2c_cfg = SepPPOConfig(
        n_batches=sep_ppo_batches,
        batch_episodes=sep_ppo_batch_eps,
        eps_decay_episodes=max(1, (sep_ppo_batches * sep_ppo_batch_eps) // 2),
        eps_decay_interactions=max(
            1, (sep_ppo_batches * sep_ppo_batch_eps * n_agents * grid.horizon) // 2
        ),
        early_stop_patience_batches=early_stop_patience_batches,
        early_stop_min_rel_policy_update=early_stop_min_rel_policy_update,
        early_stop_plateau_window_batches=early_stop_plateau_window_batches,
        early_stop_max_delta_reach_rate=early_stop_max_delta_reach_rate,
        early_stop_max_delta_mean_reward=early_stop_max_delta_mean_reward,
        early_stop_reach_target=early_stop_reach_target,
    )
    model, metrics = train_goal_a2c(
        grid,
        noise_cfg,
        n_agents=n_agents,
        cfg=a2c_cfg,
        seed=seed,
        log_every=log_every,
        print_every=print_every,
        wb_run=wb_run,
        wb_prefix=wb_prefix,
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"sep_a2c_policy_{noise_cfg.kind}.pt")
        model.save(out)
        print(f"[sep-a2c] Saved policy to {out}")
    return model, {"trained": True, "train_metrics": metrics}


def rollout_separation(
    Q: GoalConditionedTabularQ,
    grid: GridConfig,
    noise_cfg: NoiseConfig,
    n_agents: int,
    targets: List[Pos],
    rematch_every: int = 0,
    seed: int = 0,
    start_rows: List[int] | None = None,
    collect_step_traces: bool = False,
) -> dict:
    """Greedy rollout with the separation principle (Q + Hungarian)."""
    eval_grid = replace(grid, rng_seed=seed)
    eval_noise = NoiseConfig(
        kind=noise_cfg.kind, p=noise_cfg.p, rng_seed=seed + 10_000,
    )
    env = MultiAgentGrid(
        eval_grid, eval_noise, n_agents=n_agents, targets=targets,
    )
    env.reset(start_rows=start_rows)
    env.goals = assign_goals(env.pos, env.targets)

    arrival_time: list[int | None] = [None] * n_agents
    trajectories: List[List[Pos]] = [
        [tuple(env.pos[i])] for i in range(n_agents)
    ]
    step_traces: List[dict] = []
    total_cost = 0.0

    def _rematch_unreached_only() -> None:
        active_idx = [i for i, r in enumerate(env.reached) if not r]
        if not active_idx:
            return
        reached_targets = {tuple(env.pos[i]) for i, r in enumerate(env.reached) if r}
        available_targets = [t for t in env.targets if tuple(t) not in reached_targets]
        if len(available_targets) < len(active_idx):
            available_targets = list(env.targets)
        active_agents = [env.pos[i] for i in active_idx]
        matched = assign_goals(active_agents, available_targets)
        for i, g in zip(active_idx, matched):
            env.goals[i] = g

    for t in range(grid.horizon):
        states_before = [tuple(p) for p in env.pos]
        if rematch_every > 0 and t > 0 and t % rematch_every == 0:
            _rematch_unreached_only()

        greedy_actions = [
            4 if env.reached[i]
            else int(np.argmin(Q.Q[s[0], s[1], z[0], z[1], :]))
            for i, (s, z) in enumerate(zip(env.pos, env.goals))
        ]
        step = env.step(greedy_actions)
        total_cost += float(step["step_cost"])
        exec_actions = [int(a) for a in step["exec_actions"]]

        for i, nr in enumerate(step["newly_reached"]):
            if nr and arrival_time[i] is None:
                arrival_time[i] = t + 1
        for i in range(n_agents):
            trajectories[i].append(tuple(env.pos[i]))
        if collect_step_traces:
            rewards = []
            action_source = []
            for i in range(n_agents):
                if env.reached[i] and arrival_time[i] is not None and arrival_time[i] <= t:
                    if arrival_time[i] == t + 1:
                        rewards.append(float(grid.goal_bonus))
                    else:
                        rewards.append(0.0)
                elif arrival_time[i] == t + 1:
                    rewards.append(float(grid.goal_bonus))
                else:
                    rewards.append(float(-grid.step_cost))
                action_source.append(
                    "noise_override"
                    if exec_actions[i] != int(greedy_actions[i])
                    else "greedy"
                )
            step_traces.append(
                {
                    "t": int(t),
                    "states_before": [[int(r), int(c)] for r, c in states_before],
                    "greedy_actions": [int(a) for a in greedy_actions],
                    "executed_actions": exec_actions,
                    "action_source": action_source,
                    "rewards_no_collision": rewards,
                    "states_after": [[int(p[0]), int(p[1])] for p in env.pos],
                }
            )

    fp = [list(p) for p in env.pos]
    term_cost = terminal_ot_cost(env.pos, env.targets)
    cov = target_coverage_rate(env.pos, env.targets)
    n_reached = sum(1 for at in arrival_time if at is not None)
    reached_times = [at for at in arrival_time if at is not None]
    reached_or_horizon = [
        (at if at is not None else grid.horizon) for at in arrival_time
    ]

    out = {
        "algorithm": "Separation",
        "noise_kind": noise_cfg.kind, "p": noise_cfg.p,
        "agents": n_agents, "horizon": grid.horizon,
        "rematch_every": rematch_every,
        "target_coverage": cov,
        "reach_rate": n_reached / n_agents,
        "mean_time_to_reach": (
            float(np.mean(reached_times)) if reached_times else float("nan")
        ),
        "mean_time_to_reach_including_unreached": float(np.mean(reached_or_horizon)),
        "total_cost": float(total_cost),
        "terminal_ot_cost": float(term_cost),
        "final_positions": fp,
        "arrival_times": arrival_time,
        "targets": [list(t) for t in targets],
        "trajectories": [[list(p) for p in traj] for traj in trajectories],
    }
    if collect_step_traces:
        out["step_traces"] = step_traces
    return out


def ensure_sep_ppo(
    grid: GridConfig, noise_cfg: NoiseConfig, policy_path: str | None,
    n_agents: int, seed: int,
    sep_ppo_batches: int = 500, sep_ppo_batch_eps: int = 64,
    early_stop_patience_batches: int = 0,
    early_stop_min_rel_policy_update: float = 0.0,
    early_stop_plateau_window_batches: int = 0,
    early_stop_max_delta_reach_rate: float = 0.0,
    early_stop_max_delta_mean_reward: float = 0.0,
    early_stop_reach_target: float = 0.0,
    save_dir: str | None = None, wb_run=None, wb_prefix: str = "train",
    log_every: int = 10,
    print_every: int = 10_000,
) -> tuple[GoalConditionedPPOPolicy, dict]:
    """Train or load a noise-specific goal-conditioned PPO policy."""
    if policy_path and os.path.exists(policy_path):
        policy = GoalConditionedPPOPolicy.load(policy_path)
        if policy.h == grid.h and policy.w == grid.w:
            print(f"[sep-ppo] Loaded policy from {policy_path}")
            return policy, {"loaded_from": policy_path}
        print("[warn] SepPPO shape mismatch, retraining …")

    ppo_cfg = SepPPOConfig(
        n_batches=sep_ppo_batches,
        batch_episodes=sep_ppo_batch_eps,
        eps_decay_episodes=max(1, (sep_ppo_batches * sep_ppo_batch_eps) // 2),
        eps_decay_interactions=max(
            1, (sep_ppo_batches * sep_ppo_batch_eps * n_agents * grid.horizon) // 2
        ),
        early_stop_patience_batches=early_stop_patience_batches,
        early_stop_min_rel_policy_update=early_stop_min_rel_policy_update,
        early_stop_plateau_window_batches=early_stop_plateau_window_batches,
        early_stop_max_delta_reach_rate=early_stop_max_delta_reach_rate,
        early_stop_max_delta_mean_reward=early_stop_max_delta_mean_reward,
        early_stop_reach_target=early_stop_reach_target,
    )
    model, metrics = train_goal_ppo(
        grid, noise_cfg, n_agents=n_agents,
        cfg=ppo_cfg, seed=seed, log_every=log_every, print_every=print_every,
        wb_run=wb_run, wb_prefix=wb_prefix,
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"sep_policy_{noise_cfg.kind}.pt")
        model.save(out)
        print(f"[sep-ppo] Saved policy to {out}")
    return model, {"trained": True, "train_metrics": metrics}


def rollout_separation_ppo(
    policy: GoalConditionedPPOPolicy | GoalConditionedDeepDQNPolicy,
    grid: GridConfig,
    noise_cfg: NoiseConfig,
    n_agents: int,
    targets: List[Pos],
    rematch_every: int = 0,
    seed: int = 0,
    start_rows: List[int] | None = None,
    collect_step_traces: bool = False,
) -> dict:
    """Greedy rollout with separation principle (goal-conditioned PPO + Hungarian)."""
    eval_grid = replace(grid, rng_seed=seed)
    eval_noise = NoiseConfig(
        kind=noise_cfg.kind, p=noise_cfg.p, rng_seed=seed + 10_000,
    )
    env = MultiAgentGrid(
        eval_grid, eval_noise, n_agents=n_agents, targets=targets,
    )
    env.reset(start_rows=start_rows)
    env.goals = assign_goals(env.pos, env.targets)

    arrival_time: list[int | None] = [None] * n_agents
    trajectories: List[List[Pos]] = [
        [tuple(env.pos[i])] for i in range(n_agents)
    ]
    step_traces: List[dict] = []
    total_cost = 0.0

    def _rematch_unreached_only() -> None:
        active_idx = [i for i, r in enumerate(env.reached) if not r]
        if not active_idx:
            return
        reached_targets = {tuple(env.pos[i]) for i, r in enumerate(env.reached) if r}
        available_targets = [t for t in env.targets if tuple(t) not in reached_targets]
        if len(available_targets) < len(active_idx):
            available_targets = list(env.targets)
        active_agents = [env.pos[i] for i in active_idx]
        matched = assign_goals(active_agents, available_targets)
        for i, g in zip(active_idx, matched):
            env.goals[i] = g

    for t in range(grid.horizon):
        states_before = [tuple(p) for p in env.pos]
        if rematch_every > 0 and t > 0 and t % rematch_every == 0:
            _rematch_unreached_only()

        greedy_actions = [
            4 if env.reached[i]
            else policy.act_greedy(s, z, t, grid)
            for i, (s, z) in enumerate(zip(env.pos, env.goals))
        ]
        step = env.step(greedy_actions)
        total_cost += float(step["step_cost"])
        exec_actions = [int(a) for a in step["exec_actions"]]

        for i, nr in enumerate(step["newly_reached"]):
            if nr and arrival_time[i] is None:
                arrival_time[i] = t + 1
        for i in range(n_agents):
            trajectories[i].append(tuple(env.pos[i]))
        if collect_step_traces:
            rewards = []
            action_source = []
            for i in range(n_agents):
                if env.reached[i] and arrival_time[i] is not None and arrival_time[i] <= t:
                    if arrival_time[i] == t + 1:
                        rewards.append(float(grid.goal_bonus))
                    else:
                        rewards.append(0.0)
                elif arrival_time[i] == t + 1:
                    rewards.append(float(grid.goal_bonus))
                else:
                    rewards.append(float(-grid.step_cost))
                action_source.append(
                    "noise_override"
                    if exec_actions[i] != int(greedy_actions[i])
                    else "greedy"
                )
            step_traces.append(
                {
                    "t": int(t),
                    "states_before": [[int(r), int(c)] for r, c in states_before],
                    "greedy_actions": [int(a) for a in greedy_actions],
                    "executed_actions": exec_actions,
                    "action_source": action_source,
                    "rewards_no_collision": rewards,
                    "states_after": [[int(p[0]), int(p[1])] for p in env.pos],
                }
            )

    fp = [list(p) for p in env.pos]
    term_cost = terminal_ot_cost(env.pos, env.targets)
    cov = target_coverage_rate(env.pos, env.targets)
    n_reached = sum(1 for at in arrival_time if at is not None)
    reached_times = [at for at in arrival_time if at is not None]
    reached_or_horizon = [
        (at if at is not None else grid.horizon) for at in arrival_time
    ]

    out = {
        "algorithm": "SeparationPPO",
        "noise_kind": noise_cfg.kind, "p": noise_cfg.p,
        "agents": n_agents, "horizon": grid.horizon,
        "rematch_every": rematch_every,
        "target_coverage": cov,
        "reach_rate": n_reached / n_agents,
        "mean_time_to_reach": (
            float(np.mean(reached_times)) if reached_times else float("nan")
        ),
        "mean_time_to_reach_including_unreached": float(np.mean(reached_or_horizon)),
        "total_cost": float(total_cost),
        "terminal_ot_cost": float(term_cost),
        "final_positions": fp,
        "arrival_times": arrival_time,
        "targets": [list(t) for t in targets],
        "trajectories": [[list(p) for p in traj] for traj in trajectories],
    }
    if collect_step_traces:
        out["step_traces"] = step_traces
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Compare Separation Principle vs MAPPO",
    )
    ap.add_argument("--grid_h", type=int, default=11)
    ap.add_argument("--grid_w", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--agents", type=int, default=7)
    ap.add_argument("--noise_mode", type=str, default="individual,local,global",
                    help="Comma-separated noise modes to evaluate")
    ap.add_argument(
        "--p",
        type=str,
        default="0,0.05,0.1,0.25",
        help="Noise probability p, or comma-separated list (e.g. 0,0.05,0.1,0.25)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_seeds", type=int, default=20,
                    help="Number of evaluation seeds for multi-seed averaging")
    ap.add_argument(
        "--reach_thresholds",
        type=str,
        default="0.7,0.8,0.9,0.95,0.99",
        help=(
            "Comma-separated reach-rate thresholds used to compute "
            "stable-learning interaction milestones."
        ),
    )
    ap.add_argument(
        "--stability_window",
        type=int,
        default=10,
        help=(
            "Number of consecutive training batches required above a threshold "
            "to declare a stable reach level."
        ),
    )
    ap.add_argument(
        "--print_eval_traces",
        action="store_true",
        help=(
            "Print starts, targets, trajectories, and per-step "
            "(state, greedy action, executed action, reward) traces during evaluation."
        ),
    )
    ap.add_argument(
        "--print_eval_max_seeds",
        type=int,
        default=1,
        help="Maximum number of eval seeds to print when --print_eval_traces is enabled.",
    )
    ap.add_argument(
        "--fixed_targets",
        type=str,
        default=None,
        help=(
            "Optional fixed evaluation targets as 'row:col;row:col;...'. "
            "If set, the same targets are used for every evaluation seed and algorithm."
        ),
    )
    ap.add_argument(
        "--eval_target_region",
        type=str,
        choices=["last_ten", "last_half", "full_grid"],
        default="last_ten",
        help=(
            "Region used to sample both training and evaluation targets for all "
            "methods (when --fixed_targets is not set). "
            "'last_ten' samples from columns [grid_w-10, grid_w) (paper's Fig. 2: "
            "cols 20-29 on the 11x30 grid), "
            "'last_half' samples from columns [grid_w//2, grid_w), "
            "'full_grid' samples from all columns."
        ),
    )
    ap.add_argument(
        "--eval_start_mode",
        type=str,
        choices=["random_rows", "bottom_left"],
        default="random_rows",
        help=(
            "Evaluation start placement mode. "
            "'random_rows' starts each agent at (random_row, 0); "
            "'bottom_left' starts all agents at (grid_h-1, 0)."
        ),
    )
    ap.add_argument(
        "--rematch_every",
        type=int,
        default=1,
        help=(
            "Evaluation-only goal rematching cadence for separation rollouts. "
            "1 means rematch every timestep; 0 disables rematching."
        ),
    )
    ap.add_argument(
        "--also_eval_rematch_zero",
        action="store_true",
        help=(
            "Run an additional separation-only evaluation with rematch_every=0 "
            "(same eval seeds/targets) for direct comparison."
        ),
    )
    ap.add_argument("--collision_penalty", type=float, default=0.0)
    ap.add_argument("--goal_bonus", type=float, default=0.0)
    # Separation policy settings.
    ap.add_argument(
        "--sep_method",
        type=str,
        default="q",
        help="Comma-separated separation backends. Choices: q, ddqn, ppo, a2c",
    )
    ap.add_argument("--q_path_individual", type=str, default=None,
                    help="Pre-trained Q for individual noise (.npy)")
    ap.add_argument("--q_path_local", type=str, default=None,
                    help="Pre-trained Q for local noise (.npy)")
    ap.add_argument("--q_path_global", type=str, default=None,
                    help="Pre-trained Q for global noise (.npy)")
    ap.add_argument("--q_ddqn_path_individual", type=str, default=None,
                    help="Pre-trained Sep deep Double DQN policy for individual noise (.pt)")
    ap.add_argument("--q_ddqn_path_local", type=str, default=None,
                    help="Pre-trained Sep deep Double DQN policy for local noise (.pt)")
    ap.add_argument("--q_ddqn_path_global", type=str, default=None,
                    help="Pre-trained Sep deep Double DQN policy for global noise (.pt)")
    ap.add_argument("--sep_policy_path_individual", type=str, default=None,
                    help="Pre-trained Separation PPO policy for individual noise (.pt)")
    ap.add_argument("--sep_policy_path_local", type=str, default=None,
                    help="Pre-trained Separation PPO policy for local noise (.pt)")
    ap.add_argument("--sep_policy_path_global", type=str, default=None,
                    help="Pre-trained Separation PPO policy for global noise (.pt)")
    ap.add_argument("--sep_a2c_policy_path_individual", type=str, default=None,
                    help="Pre-trained Separation A2C policy for individual noise (.pt)")
    ap.add_argument("--sep_a2c_policy_path_local", type=str, default=None,
                    help="Pre-trained Separation A2C policy for local noise (.pt)")
    ap.add_argument("--sep_a2c_policy_path_global", type=str, default=None,
                    help="Pre-trained Separation A2C policy for global noise (.pt)")
    ap.add_argument("--sep_episodes", type=int, default=320_000,
                    help="Training episodes for separation Q-learning")
    ap.add_argument("--sep_ppo_batches", type=int, default=5000,
                    help="PPO batches for Separation-PPO training")
    ap.add_argument("--sep_ppo_batch_eps", type=int, default=64,
                    help="Episodes per PPO batch for Separation-PPO training")
    ap.add_argument(
        "--sep_ppo_early_stop_patience",
        type=int,
        default=0,
        help=(
            "Early-stop Sep-PPO when relative policy update stays below threshold "
            "for this many consecutive batches (0 disables)."
        ),
    )
    ap.add_argument(
        "--sep_ppo_early_stop_min_rel_update",
        type=float,
        default=0.0,
        help="Minimum relative Sep-PPO policy update for early-stop check (0 disables).",
    )
    ap.add_argument(
        "--sep_ppo_early_stop_plateau_window",
        type=int,
        default=0,
        help="Sep-PPO plateau window (batches) for hybrid early stop (0 disables plateau check).",
    )
    ap.add_argument(
        "--sep_ppo_early_stop_max_delta_reach",
        type=float,
        default=0.0,
        help="Max abs delta of Sep-PPO reach-rate moving averages between consecutive windows.",
    )
    ap.add_argument(
        "--sep_ppo_early_stop_max_delta_reward",
        type=float,
        default=0.0,
        help="Max abs delta of Sep-PPO mean-reward moving averages between consecutive windows.",
    )
    # MAPPO settings.
    ap.add_argument("--mappo_batches", type=int, default=5000,
                    help="Number of PPO update batches")
    ap.add_argument("--mappo_batch_eps", type=int, default=64,
                    help="Episodes per PPO batch")
    ap.add_argument(
        "--mappo_early_stop_patience",
        type=int,
        default=0,
        help=(
            "Early-stop MAPPO when relative policy update stays below threshold "
            "for this many consecutive batches (0 disables)."
        ),
    )
    ap.add_argument(
        "--mappo_early_stop_min_rel_update",
        type=float,
        default=0.0,
        help="Minimum relative MAPPO policy update for early-stop check (0 disables).",
    )
    ap.add_argument(
        "--mappo_early_stop_plateau_window",
        type=int,
        default=0,
        help="MAPPO plateau window (batches) for hybrid early stop (0 disables plateau check).",
    )
    ap.add_argument(
        "--mappo_early_stop_max_delta_reach",
        type=float,
        default=0.0,
        help="Max abs delta of MAPPO reach-rate moving averages between consecutive windows.",
    )
    ap.add_argument(
        "--mappo_early_stop_max_delta_reward",
        type=float,
        default=0.0,
        help="Max abs delta of MAPPO mean-reward moving averages between consecutive windows.",
    )
    ap.add_argument(
        "--mappo_obs_modes",
        type=str,
        default="relative_targets",
        help=(
            "Comma-separated MAPPO observation modes. "
            "Choices: relative_targets"
        ),
    )
    ap.add_argument("--wandb_log_points", type=int, default=1000,
                    help="Approximate number of points per training curve in wandb")
    ap.add_argument(
        "--train_print_every_batches",
        type=int,
        default=500,
        help="Console progress print cadence in training batches (larger = less frequent).",
    )
    # Optional baseline settings.
    ap.add_argument("--mappo", dest="run_mappo", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Train/evaluate the MAPPO baseline (default on; use --no-mappo "
                         "to skip it, e.g. to compare only Sep-PPO vs MFQ vs MFQ-local).")
    ap.add_argument("--run_ippo", action="store_true", help="Also train/evaluate IPPO baseline")
    ap.add_argument("--ippo_batches", type=int, default=5000,
                    help="Number of IPPO update batches")
    ap.add_argument("--ippo_batch_eps", type=int, default=64,
                    help="Episodes per IPPO batch")
    ap.add_argument(
        "--ippo_obs_modes",
        type=str,
        default="relative_targets",
        help=(
            "Comma-separated IPPO observation modes. "
            "Choices: relative_targets"
        ),
    )
    ap.add_argument("--run_happo", action="store_true",
                    help="Also train/evaluate HAPPO baseline (Kuba et al., 2022)")
    ap.add_argument("--happo_batches", type=int, default=5000,
                    help="Number of HAPPO update batches")
    ap.add_argument("--happo_batch_eps", type=int, default=64,
                    help="Episodes per HAPPO batch")
    ap.add_argument(
        "--happo_obs_modes",
        type=str,
        default="relative_targets",
        help=(
            "Comma-separated HAPPO observation modes. "
            "Choices: relative_targets"
        ),
    )
    ap.add_argument("--run_qmix", action="store_true", help="Also train/evaluate QMIX baseline")
    ap.add_argument("--qmix_batches", type=int, default=5000,
                    help="Number of QMIX update batches")
    ap.add_argument("--qmix_batch_eps", type=int, default=64,
                    help="Episodes per QMIX batch")
    ap.add_argument(
        "--qmix_obs_modes",
        type=str,
        default="relative_targets",
        help=(
            "Comma-separated QMIX observation modes. "
            "Choices: relative_targets"
        ),
    )
    ap.add_argument("--run_vdn", action="store_true", help="Also train/evaluate VDN baseline")
    ap.add_argument("--vdn_batches", type=int, default=5000,
                    help="Number of VDN update batches")
    ap.add_argument("--vdn_batch_eps", type=int, default=64,
                    help="Episodes per VDN batch")
    ap.add_argument(
        "--vdn_obs_modes",
        type=str,
        default="relative_targets",
        help=(
            "Comma-separated VDN observation modes. "
            "Choices: relative_targets"
        ),
    )
    ap.add_argument("--run_mfq", action="store_true",
                    help="Also train/evaluate Mean-Field Q-learning (MFQ) baseline")
    ap.add_argument("--mfq_batches", type=int, default=5000,
                    help="Number of MFQ update batches")
    ap.add_argument("--mfq_batch_eps", type=int, default=64,
                    help="Episodes per MFQ batch")
    ap.add_argument(
        "--mfq_obs_modes",
        type=str,
        default="relative_targets",
        help=(
            "Comma-separated MFQ observation modes. "
            "Choices: relative_targets"
        ),
    )
    ap.add_argument("--mfq_early_stop_patience", type=int, default=0,
                    help="MFQ plateau early-stop patience in batches (0 disables).")
    ap.add_argument("--mfq_early_stop_plateau_window", type=int, default=0,
                    help="MFQ plateau/target window in batches.")
    ap.add_argument("--mfq_early_stop_max_delta_reach", type=float, default=0.0,
                    help="MFQ max reach-rate moving-average delta for the plateau stop.")
    ap.add_argument("--mfq_early_stop_max_delta_reward", type=float, default=0.0,
                    help="MFQ max mean-reward moving-average delta for the plateau stop "
                         "(<=0 uses a reach-rate-only plateau).")
    ap.add_argument("--mfq_explore_decay_cap_batches", type=int, default=5000,
                    help="Decay the MFQ tau/eps exploration schedule to its floor over "
                         "this many batches (default 5000), or the whole run if shorter. "
                         "0 means decay over the full batch budget.")
    # MFQ-local: faithful Mean-Field Q-learning variant with an N-independent
    # egocentric target-density observation and a *local* mean field.
    ap.add_argument("--run_mfq_local", action="store_true",
                    help="Also train/evaluate the faithful MFQ-local baseline "
                         "(density-map obs + local mean field)")
    ap.add_argument("--mfq_local_batches", type=int, default=5000,
                    help="Number of MFQ-local update batches")
    ap.add_argument("--mfq_local_batch_eps", type=int, default=64,
                    help="Episodes per MFQ-local batch")
    ap.add_argument("--mfq_local_obs_mode", type=str, default="density_map",
                    choices=["density_map", "fov"],
                    help="MFQ-local target encoding: 'density_map' (coarse K*K egocentric "
                         "histogram) or 'fov' ((2R+1)^2 egocentric presence window). Both "
                         "are independent of the number of agents.")
    ap.add_argument("--mfq_local_density_bins", type=int, default=5,
                    help="density_map resolution K (obs_dim = 5 + K*K)")
    ap.add_argument("--mfq_local_fov_radius", type=int, default=2,
                    help="fov window half-width R (obs_dim = 5 + (2R+1)^2)")
    ap.add_argument("--mfq_local_mf_radius", type=int, default=3,
                    help="Local mean-field neighbourhood radius (Chebyshev, grid cells)")
    ap.add_argument("--mfq_local_early_stop_patience", type=int, default=0,
                    help="MFQ-local plateau early-stop patience in batches (0 disables).")
    ap.add_argument("--mfq_local_early_stop_plateau_window", type=int, default=0,
                    help="MFQ-local plateau/target window in batches.")
    ap.add_argument("--mfq_local_early_stop_max_delta_reach", type=float, default=0.0,
                    help="MFQ-local max reach-rate moving-average delta for the plateau stop.")
    ap.add_argument("--mfq_local_early_stop_max_delta_reward", type=float, default=0.0,
                    help="MFQ-local max mean-reward moving-average delta for the plateau stop "
                         "(<=0 uses a reach-rate-only plateau).")
    ap.add_argument("--mfq_local_explore_decay_cap_batches", type=int, default=5000,
                    help="Decay the MFQ-local tau/eps exploration schedule to its floor over "
                         "this many batches (default 5000), or the whole run if shorter. "
                         "0 means decay over the full batch budget.")
    # Shared success stop: halt a method once its training reach rate is
    # sustained >= target for the plateau window. Applies to MAPPO, Sep-PPO, MFQ,
    # MFQ-local.
    ap.add_argument("--early_stop_reach_target", type=float, default=0.0,
                    help="Stop training once training reach rate is sustained >= this "
                         "target (e.g. 0.99). 0 disables. Applies to MAPPO, Sep-PPO, MFQ, "
                         "MFQ-local.")
    ap.add_argument("--early_stop_min_reach", type=float, default=0.0,
                    help="MFQ / MFQ-local plateau stop only fires once the reach-rate "
                         "moving average is >= this floor, preventing a false stop during "
                         "early low-reach exploration. 0 disables the guard.")
    # Output and logging.
    ap.add_argument("--outdir", type=str, default="runs/comparison")
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="marl-separation-noise",
                    help="W&B project name")
    ap.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team/account)")
    args = ap.parse_args()

    p_values = _parse_p_values(args.p)
    if len(p_values) > 1:
        raw_args = sys.argv[1:]
        stripped_args: list[str] = []
        i = 0
        while i < len(raw_args):
            tok = raw_args[i]
            if tok == "--p":
                i += 2
                continue
            if tok.startswith("--p="):
                i += 1
                continue
            stripped_args.append(tok)
            i += 1
        for pval in p_values:
            cmd = [sys.executable, os.path.abspath(__file__), *stripped_args, "--p", str(pval)]
            print(f"\n[multi-p] launching p={pval}: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        return
    args.p = float(p_values[0])
    reach_thresholds = _parse_reach_thresholds(args.reach_thresholds)
    if args.stability_window <= 0:
        raise ValueError("--stability_window must be >= 1.")
    if args.sep_ppo_early_stop_patience < 0 or args.mappo_early_stop_patience < 0:
        raise ValueError("Early-stop patience values must be >= 0.")
    if args.sep_ppo_early_stop_min_rel_update < 0.0 or args.mappo_early_stop_min_rel_update < 0.0:
        raise ValueError("Early-stop relative-update thresholds must be >= 0.")
    if args.sep_ppo_early_stop_plateau_window < 0 or args.mappo_early_stop_plateau_window < 0:
        raise ValueError("Early-stop plateau windows must be >= 0.")
    if args.sep_ppo_early_stop_max_delta_reach < 0.0 or args.mappo_early_stop_max_delta_reach < 0.0:
        raise ValueError("Early-stop reach-rate deltas must be >= 0.")
    if args.sep_ppo_early_stop_max_delta_reward < 0.0 or args.mappo_early_stop_max_delta_reward < 0.0:
        raise ValueError("Early-stop reward deltas must be >= 0.")

    wb_run = None
    if args.wandb:
        import wandb
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"comparison_p{args.p}_N{args.agents}",
            tags=["comparison"],
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_outdir = os.path.join(args.outdir, f"p{args.p}_{stamp}")
    os.makedirs(base_outdir, exist_ok=True)

    grid = GridConfig(
        h=args.grid_h, w=args.grid_w, horizon=args.horizon,
        rng_seed=args.seed, collision_penalty=args.collision_penalty,
        goal_bonus=args.goal_bonus, target_region=args.eval_target_region,
    )
    N = args.agents
    n_eval = args.eval_seeds
    fixed_targets = (
        _parse_fixed_targets(args.fixed_targets, grid.h, grid.w)
        if args.fixed_targets else None
    )
    if fixed_targets is not None:
        print(
            f"[viz] using fixed targets ({len(fixed_targets)}): "
            f"{[list(t) for t in fixed_targets]}"
        )

    sep_methods = _parse_sep_methods(args.sep_method)
    primary_sep_method = sep_methods[0]
    q_paths = {
        "individual": args.q_path_individual,
        "local": args.q_path_local,
        "global": args.q_path_global,
    }
    sep_policy_paths = {
        "individual": args.sep_policy_path_individual,
        "local": args.sep_policy_path_local,
        "global": args.sep_policy_path_global,
    }
    q_ddqn_paths = {
        "individual": args.q_ddqn_path_individual,
        "local": args.q_ddqn_path_local,
        "global": args.q_ddqn_path_global,
    }
    sep_a2c_policy_paths = {
        "individual": args.sep_a2c_policy_path_individual,
        "local": args.sep_a2c_policy_path_local,
        "global": args.sep_a2c_policy_path_global,
    }

    mappo_modes = _parse_relative_only_modes(args.mappo_obs_modes, "MAPPO") if args.run_mappo else []
    mappo_cfg = MAPPOConfig(
        n_batches=args.mappo_batches,
        batch_episodes=args.mappo_batch_eps,
        early_stop_patience_batches=args.mappo_early_stop_patience,
        early_stop_min_rel_policy_update=args.mappo_early_stop_min_rel_update,
        early_stop_plateau_window_batches=args.mappo_early_stop_plateau_window,
        early_stop_max_delta_reach_rate=args.mappo_early_stop_max_delta_reach,
        early_stop_max_delta_mean_reward=args.mappo_early_stop_max_delta_reward,
        early_stop_reach_target=args.early_stop_reach_target,
    )
    ippo_modes = (
        _parse_relative_only_modes(args.ippo_obs_modes, "IPPO")
        if args.run_ippo else []
    )
    ippo_cfg = IPPOConfig(
        n_batches=args.ippo_batches,
        batch_episodes=args.ippo_batch_eps,
    )
    happo_modes = (
        _parse_relative_only_modes(args.happo_obs_modes, "HAPPO")
        if args.run_happo else []
    )
    happo_cfg = HAPPOConfig(
        n_batches=args.happo_batches,
        batch_episodes=args.happo_batch_eps,
    )
    qmix_modes = (
        _parse_relative_only_modes(args.qmix_obs_modes, "QMIX")
        if args.run_qmix else []
    )
    qmix_cfg = QMIXConfig(
        n_batches=args.qmix_batches,
        batch_episodes=args.qmix_batch_eps,
        eps_decay_episodes=max(1, (args.qmix_batches * args.qmix_batch_eps) // 2),
    )
    vdn_modes = (
        _parse_relative_only_modes(args.vdn_obs_modes, "VDN")
        if args.run_vdn else []
    )
    vdn_cfg = VDNConfig(
        n_batches=args.vdn_batches,
        batch_episodes=args.vdn_batch_eps,
        eps_decay_episodes=max(1, (args.vdn_batches * args.vdn_batch_eps) // 2),
    )
    mfq_modes = (
        _parse_relative_only_modes(args.mfq_obs_modes, "MFQ")
        if args.run_mfq else []
    )
    # Exploration-decay horizon: half the episode budget, optionally capped so it
    # does not blow up when the interaction-matched budget yields a huge batch
    # count (e.g. low-agent scalability runs). See --mfq_explore_decay_cap_batches.
    # Decay tau/eps to their floors over the first `explore_decay_cap_batches`
    # batches (default 5000), or the whole run if it is shorter.
    _mfq_decay_batches = min(args.mfq_batches,
                             args.mfq_explore_decay_cap_batches or args.mfq_batches)
    mfq_decay_episodes = max(1, _mfq_decay_batches * args.mfq_batch_eps)
    mfq_cfg = MFQConfig(
        n_batches=args.mfq_batches,
        batch_episodes=args.mfq_batch_eps,
        tau_decay_episodes=mfq_decay_episodes,
        eps_decay_episodes=mfq_decay_episodes,
        early_stop_patience_batches=args.mfq_early_stop_patience,
        early_stop_plateau_window_batches=args.mfq_early_stop_plateau_window,
        early_stop_max_delta_reach_rate=args.mfq_early_stop_max_delta_reach,
        early_stop_max_delta_mean_reward=args.mfq_early_stop_max_delta_reward,
        early_stop_reach_target=args.early_stop_reach_target,
        early_stop_min_reach=args.early_stop_min_reach,
    )
    mfq_local_modes = [args.mfq_local_obs_mode] if args.run_mfq_local else []
    _mfq_local_decay_batches = min(args.mfq_local_batches,
                                   args.mfq_local_explore_decay_cap_batches or args.mfq_local_batches)
    mfq_local_decay_episodes = max(1, _mfq_local_decay_batches * args.mfq_local_batch_eps)
    mfq_local_cfg = MFQLocalConfig(
        n_batches=args.mfq_local_batches,
        batch_episodes=args.mfq_local_batch_eps,
        obs_mode=args.mfq_local_obs_mode,
        density_bins=args.mfq_local_density_bins,
        fov_radius=args.mfq_local_fov_radius,
        mf_radius=args.mfq_local_mf_radius,
        tau_decay_episodes=mfq_local_decay_episodes,
        eps_decay_episodes=mfq_local_decay_episodes,
        early_stop_patience_batches=args.mfq_local_early_stop_patience,
        early_stop_plateau_window_batches=args.mfq_local_early_stop_plateau_window,
        early_stop_max_delta_reach_rate=args.mfq_local_early_stop_max_delta_reach,
        early_stop_max_delta_mean_reward=args.mfq_local_early_stop_max_delta_reward,
        early_stop_reach_target=args.early_stop_reach_target,
        early_stop_min_reach=args.early_stop_min_reach,
    )
    # Keep logged curves comparable without storing every training batch.
    sep_log_every_q = max(1, args.sep_episodes // max(1, args.wandb_log_points))
    sep_log_every_ddqn = max(1, args.sep_ppo_batches // max(1, args.wandb_log_points))
    sep_log_every_ppo = max(1, args.sep_ppo_batches // max(1, args.wandb_log_points))
    sep_log_every_a2c = sep_log_every_ppo
    mappo_log_every = max(1, mappo_cfg.n_batches // max(1, args.wandb_log_points))
    ippo_log_every = max(1, ippo_cfg.n_batches // max(1, args.wandb_log_points))
    happo_log_every = max(1, happo_cfg.n_batches // max(1, args.wandb_log_points))
    qmix_log_every = max(1, qmix_cfg.n_batches // max(1, args.wandb_log_points))
    vdn_log_every = max(1, vdn_cfg.n_batches // max(1, args.wandb_log_points))
    mfq_log_every = max(1, mfq_cfg.n_batches // max(1, args.wandb_log_points))
    mfq_local_log_every = max(1, mfq_local_cfg.n_batches // max(1, args.wandb_log_points))
    sep_ppo_print_every = max(1, args.train_print_every_batches * args.sep_ppo_batch_eps)
    mappo_print_every = max(1, args.train_print_every_batches * args.mappo_batch_eps)
    mfq_print_every = max(1, args.train_print_every_batches * args.mfq_batch_eps)
    mfq_local_print_every = max(1, args.train_print_every_batches * args.mfq_local_batch_eps)
    all_results: List[dict] = []

    print(
        f"[wandb] log cadence: "
        f"sep_methods={sep_methods} "
        f"(q:{sep_log_every_q} episodes, ddqn/ppo/a2c:{sep_log_every_ppo} batches) | "
        f"mappo every {mappo_log_every} batches "
        f"({mappo_log_every * mappo_cfg.batch_episodes} episodes); "
        f"mappo_modes={mappo_modes}; "
        f"ippo={'on' if args.run_ippo else 'off'}"
        f"{f' ippo_modes={ippo_modes} every {ippo_log_every} batches;' if args.run_ippo else ';'} "
        f"happo={'on' if args.run_happo else 'off'}"
        f"{f' happo_modes={happo_modes} every {happo_log_every} batches;' if args.run_happo else ';'} "
        f" qmix={'on' if args.run_qmix else 'off'}"
        f"{f' qmix_modes={qmix_modes} every {qmix_log_every} batches;' if args.run_qmix else ';'} "
        f"vdn={'on' if args.run_vdn else 'off'}"
        f"{f' vdn_modes={vdn_modes} every {vdn_log_every} batches;' if args.run_vdn else ''} "
        f"mfq={'on' if args.run_mfq else 'off'}"
        f"{f' mfq_modes={mfq_modes} every {mfq_log_every} batches' if args.run_mfq else ''} "
        f"mfq_local={'on' if args.run_mfq_local else 'off'}"
        f"{f' every {mfq_local_log_every} batches' if args.run_mfq_local else ''}"
    )

    selected_noise_kinds = [k.strip() for k in args.noise_mode.split(",") if k.strip()]
    valid_noise_kinds = {"individual", "local", "global"}
    invalid_noise_kinds = [k for k in selected_noise_kinds if k not in valid_noise_kinds]
    if invalid_noise_kinds:
        raise ValueError(
            f"Unknown noise_mode values: {invalid_noise_kinds}. "
            f"Valid options are: {sorted(valid_noise_kinds)}."
        )

    for kind in selected_noise_kinds:
        noise_cfg = NoiseConfig(kind=kind, p=args.p, rng_seed=args.seed)
        kind_dir = os.path.join(base_outdir, kind)
        os.makedirs(kind_dir, exist_ok=True)

        # Train separation backends.
        sep_by_method: dict[str, dict] = {}
        for sep_method in sep_methods:
            print(f"\n{'=' * 60}")
            print(f"  SEPARATION[{sep_method}] — {kind} noise  (p={args.p})")
            print(f"{'=' * 60}")
            t0 = time.time()
            if sep_method == "q":
                sep_model, sep_train_info = ensure_sep_q(
                    grid, noise_cfg, q_paths[kind], n_agents=N,
                    seed=args.seed, sep_episodes=args.sep_episodes,
                    save_dir=kind_dir, wb_run=wb_run,
                    wb_prefix=f"train_sep/{kind}",
                    log_every=sep_log_every_q,
                )
                sep_env_interactions = args.sep_episodes * N * grid.horizon
            elif sep_method == "ddqn":
                sep_model, sep_train_info = ensure_sep_ddqn(
                    grid, noise_cfg, q_ddqn_paths[kind], n_agents=N,
                    seed=args.seed,
                    sep_ppo_batches=args.sep_ppo_batches,
                    sep_ppo_batch_eps=args.sep_ppo_batch_eps,
                    early_stop_patience_batches=args.sep_ppo_early_stop_patience,
                    early_stop_min_rel_policy_update=args.sep_ppo_early_stop_min_rel_update,
                    early_stop_plateau_window_batches=args.sep_ppo_early_stop_plateau_window,
                    early_stop_max_delta_reach_rate=args.sep_ppo_early_stop_max_delta_reach,
                    early_stop_max_delta_mean_reward=args.sep_ppo_early_stop_max_delta_reward,
                    save_dir=kind_dir, wb_run=wb_run,
                    wb_prefix=f"train_sep_ddqn/{kind}",
                    log_every=sep_log_every_ddqn,
                    print_every=sep_ppo_print_every,
                )
                sep_env_interactions = (
                    sep_train_info.get("train_metrics", {}).get("env_interactions")
                    if isinstance(sep_train_info, dict) else None
                )
                if sep_env_interactions is None:
                    sep_env_interactions = (
                        args.sep_ppo_batches * args.sep_ppo_batch_eps * N * grid.horizon
                    )
            elif sep_method == "ppo":
                sep_model, sep_train_info = ensure_sep_ppo(
                    grid, noise_cfg, sep_policy_paths[kind],
                    n_agents=N,
                    seed=args.seed,
                    sep_ppo_batches=args.sep_ppo_batches,
                    sep_ppo_batch_eps=args.sep_ppo_batch_eps,
                    early_stop_patience_batches=args.sep_ppo_early_stop_patience,
                    early_stop_min_rel_policy_update=args.sep_ppo_early_stop_min_rel_update,
                    early_stop_plateau_window_batches=args.sep_ppo_early_stop_plateau_window,
                    early_stop_max_delta_reach_rate=args.sep_ppo_early_stop_max_delta_reach,
                    early_stop_max_delta_mean_reward=args.sep_ppo_early_stop_max_delta_reward,
                    early_stop_reach_target=args.early_stop_reach_target,
                    save_dir=kind_dir, wb_run=wb_run,
                    wb_prefix=f"train_sep_ppo/{kind}",
                    log_every=sep_log_every_ppo,
                    print_every=sep_ppo_print_every,
                )
                sep_env_interactions = (
                    sep_train_info.get("train_metrics", {}).get("env_interactions")
                    if isinstance(sep_train_info, dict) else None
                )
                if sep_env_interactions is None:
                    sep_env_interactions = (
                        args.sep_ppo_batches * args.sep_ppo_batch_eps * grid.horizon
                    )
            elif sep_method == "a2c":
                sep_model, sep_train_info = ensure_sep_a2c(
                    grid, noise_cfg, sep_a2c_policy_paths[kind],
                    n_agents=N,
                    seed=args.seed,
                    sep_ppo_batches=args.sep_ppo_batches,
                    sep_ppo_batch_eps=args.sep_ppo_batch_eps,
                    early_stop_patience_batches=args.sep_ppo_early_stop_patience,
                    early_stop_min_rel_policy_update=args.sep_ppo_early_stop_min_rel_update,
                    early_stop_plateau_window_batches=args.sep_ppo_early_stop_plateau_window,
                    early_stop_max_delta_reach_rate=args.sep_ppo_early_stop_max_delta_reach,
                    early_stop_max_delta_mean_reward=args.sep_ppo_early_stop_max_delta_reward,
                    early_stop_reach_target=args.early_stop_reach_target,
                    save_dir=kind_dir, wb_run=wb_run,
                    wb_prefix=f"train_sep_a2c/{kind}",
                    log_every=sep_log_every_a2c,
                    print_every=sep_ppo_print_every,
                )
                sep_env_interactions = (
                    sep_train_info.get("train_metrics", {}).get("env_interactions")
                    if isinstance(sep_train_info, dict) else None
                )
                if sep_env_interactions is None:
                    sep_env_interactions = (
                        args.sep_ppo_batches * args.sep_ppo_batch_eps * grid.horizon
                    )
            else:
                raise ValueError(f"Unhandled sep_method: {sep_method}")
            sep_train_time = time.time() - t0
            sep_by_method[sep_method] = {
                "model": sep_model,
                "train_info": sep_train_info,
                "train_time": sep_train_time,
                "env_interactions": sep_env_interactions,
            }
            if wb_run is not None:
                wb_run.log({
                    f"timing/{kind}/sep/{sep_method}/train_time_s": sep_train_time,
                    f"timing/{kind}/sep/{sep_method}/env_interactions": sep_env_interactions,
                })

        # Train MAPPO variants (optional; --no-mappo skips it).
        mappo_by_mode: dict[str, dict] = {}
        primary_mode = mappo_modes[0] if mappo_modes else None
        if args.run_mappo:
            print(f"\n{'=' * 60}")
            print(f"  MAPPO variants — {kind} noise  (p={args.p})  "
                  f"({mappo_cfg.n_batches}×{mappo_cfg.batch_episodes} eps each)")
            print(f"{'=' * 60}")
        for mode in mappo_modes:
            cfg_mode = replace(mappo_cfg, obs_mode=mode)
            t0 = time.time()
            actor, mappo_train = train_mappo(
                grid, noise_cfg, N,
                cfg=cfg_mode,
                seed=args.seed,
                log_every=mappo_log_every,
                print_every=mappo_print_every,
                wb_run=wb_run, wb_prefix=f"train_mappo/{kind}/{mode}",
            )
            mappo_train_time = time.time() - t0
            mappo_env_interactions = mappo_train["env_interactions"]
            mappo_by_mode[mode] = {
                "actor": actor,
                "train": mappo_train,
                "train_time": mappo_train_time,
                "env_interactions": mappo_env_interactions,
            }

            import torch
            torch.save(actor.state_dict(), os.path.join(kind_dir, f"actor_{mode}.pt"))
            if mode == primary_mode:
                torch.save(actor.state_dict(), os.path.join(kind_dir, "actor.pt"))

            if wb_run is not None:
                wb_run.log({
                    f"timing/{kind}/mappo/{mode}/train_time_s": mappo_train_time,
                    f"timing/{kind}/mappo/{mode}/env_interactions": mappo_env_interactions,
                })

        # Train optional IPPO variants.
        ippo_by_mode: dict[str, dict] = {}
        ippo_primary_mode = None
        if args.run_ippo:
            print(f"\n{'=' * 60}")
            print(f"  IPPO variants — {kind} noise  (p={args.p})  "
                  f"({ippo_cfg.n_batches}×{ippo_cfg.batch_episodes} eps each)")
            print(f"{'=' * 60}")
            ippo_primary_mode = ippo_modes[0]
            for mode in ippo_modes:
                cfg_mode = replace(ippo_cfg, obs_mode=mode)
                t0 = time.time()
                ippo_actor, ippo_train = train_ippo(
                    grid, noise_cfg, N,
                    cfg=cfg_mode, seed=args.seed, log_every=ippo_log_every,
                    wb_run=wb_run, wb_prefix=f"train_ippo/{kind}/{mode}",
                )
                ippo_train_time = time.time() - t0
                ippo_env_interactions = ippo_train["env_interactions"]
                ippo_by_mode[mode] = {
                    "actor": ippo_actor,
                    "train": ippo_train,
                    "train_time": ippo_train_time,
                    "env_interactions": ippo_env_interactions,
                }

                import torch
                torch.save(ippo_actor.state_dict(), os.path.join(kind_dir, f"ippo_actor_{mode}.pt"))
                if mode == ippo_primary_mode:
                    torch.save(ippo_actor.state_dict(), os.path.join(kind_dir, "ippo_actor.pt"))

                if wb_run is not None:
                    wb_run.log({
                        f"timing/{kind}/ippo/{mode}/train_time_s": ippo_train_time,
                        f"timing/{kind}/ippo/{mode}/env_interactions": ippo_env_interactions,
                    })

        # Train optional HAPPO variants (Kuba et al., 2022).
        happo_by_mode: dict[str, dict] = {}
        happo_primary_mode = None
        if args.run_happo:
            print(f"\n{'=' * 60}")
            print(f"  HAPPO variants — {kind} noise  (p={args.p})  "
                  f"({happo_cfg.n_batches}×{happo_cfg.batch_episodes} eps each)")
            print(f"{'=' * 60}")
            happo_primary_mode = happo_modes[0]
            for mode in happo_modes:
                cfg_mode = replace(happo_cfg, obs_mode=mode)
                t0 = time.time()
                happo_actor, happo_train = train_happo(
                    grid, noise_cfg, N,
                    cfg=cfg_mode, seed=args.seed, log_every=happo_log_every,
                    wb_run=wb_run, wb_prefix=f"train_happo/{kind}/{mode}",
                )
                happo_train_time = time.time() - t0
                happo_env_interactions = happo_train["env_interactions"]
                happo_by_mode[mode] = {
                    "actor": happo_actor,
                    "train": happo_train,
                    "train_time": happo_train_time,
                    "env_interactions": happo_env_interactions,
                }

                import torch
                torch.save(happo_actor.state_dict(), os.path.join(kind_dir, f"happo_actor_{mode}.pt"))
                if mode == happo_primary_mode:
                    torch.save(happo_actor.state_dict(), os.path.join(kind_dir, "happo_actor.pt"))

                if wb_run is not None:
                    wb_run.log({
                        f"timing/{kind}/happo/{mode}/train_time_s": happo_train_time,
                        f"timing/{kind}/happo/{mode}/env_interactions": happo_env_interactions,
                    })

        # Train optional QMIX variants.
        qmix_by_mode: dict[str, dict] = {}
        qmix_primary_mode = None
        if args.run_qmix:
            print(f"\n{'=' * 60}")
            print(f"  QMIX variants — {kind} noise  (p={args.p})  "
                  f"({qmix_cfg.n_batches}×{qmix_cfg.batch_episodes} eps each)")
            print(f"{'=' * 60}")
            qmix_primary_mode = qmix_modes[0]
            for mode in qmix_modes:
                cfg_mode = replace(qmix_cfg, obs_mode=mode)
                t0 = time.time()
                qmix_q, qmix_train = train_qmix(
                    grid, noise_cfg, N,
                    cfg=cfg_mode, seed=args.seed, log_every=qmix_log_every,
                    wb_run=wb_run, wb_prefix=f"train_qmix/{kind}/{mode}",
                )
                qmix_train_time = time.time() - t0
                qmix_env_interactions = qmix_train["env_interactions"]
                qmix_by_mode[mode] = {
                    "q_net": qmix_q,
                    "train": qmix_train,
                    "train_time": qmix_train_time,
                    "env_interactions": qmix_env_interactions,
                }

                import torch
                torch.save(qmix_q.state_dict(), os.path.join(kind_dir, f"qmix_q_{mode}.pt"))
                if mode == qmix_primary_mode:
                    torch.save(qmix_q.state_dict(), os.path.join(kind_dir, "qmix_q.pt"))

                if wb_run is not None:
                    wb_run.log({
                        f"timing/{kind}/qmix/{mode}/train_time_s": qmix_train_time,
                        f"timing/{kind}/qmix/{mode}/env_interactions": qmix_env_interactions,
                    })

        # Train optional VDN variants.
        vdn_by_mode: dict[str, dict] = {}
        vdn_primary_mode = None
        if args.run_vdn:
            print(f"\n{'=' * 60}")
            print(f"  VDN variants — {kind} noise  (p={args.p})  "
                  f"({vdn_cfg.n_batches}×{vdn_cfg.batch_episodes} eps each)")
            print(f"{'=' * 60}")
            vdn_primary_mode = vdn_modes[0]
            for mode in vdn_modes:
                cfg_mode = replace(vdn_cfg, obs_mode=mode)
                t0 = time.time()
                vdn_q, vdn_train = train_vdn(
                    grid, noise_cfg, N,
                    cfg=cfg_mode, seed=args.seed, log_every=vdn_log_every,
                    wb_run=wb_run, wb_prefix=f"train_vdn/{kind}/{mode}",
                )
                vdn_train_time = time.time() - t0
                vdn_env_interactions = vdn_train["env_interactions"]
                vdn_by_mode[mode] = {
                    "q_net": vdn_q,
                    "train": vdn_train,
                    "train_time": vdn_train_time,
                    "env_interactions": vdn_env_interactions,
                }

                import torch
                torch.save(vdn_q.state_dict(), os.path.join(kind_dir, f"vdn_q_{mode}.pt"))
                if mode == vdn_primary_mode:
                    torch.save(vdn_q.state_dict(), os.path.join(kind_dir, "vdn_q.pt"))

                if wb_run is not None:
                    wb_run.log({
                        f"timing/{kind}/vdn/{mode}/train_time_s": vdn_train_time,
                        f"timing/{kind}/vdn/{mode}/env_interactions": vdn_env_interactions,
                    })

        # Train optional MFQ variants.
        mfq_by_mode: dict[str, dict] = {}
        mfq_primary_mode = None
        if args.run_mfq:
            print(f"\n{'=' * 60}")
            print(f"  MFQ variants — {kind} noise  (p={args.p})  "
                  f"({mfq_cfg.n_batches}×{mfq_cfg.batch_episodes} eps each)")
            print(f"{'=' * 60}")
            mfq_primary_mode = mfq_modes[0]
            for mode in mfq_modes:
                cfg_mode = replace(mfq_cfg, obs_mode=mode)
                t0 = time.time()
                mfq_q, mfq_train = train_mfq(
                    grid, noise_cfg, N,
                    cfg=cfg_mode, seed=args.seed, log_every=mfq_log_every,
                    print_every=mfq_print_every,
                    wb_run=wb_run, wb_prefix=f"train_mfq/{kind}/{mode}",
                )
                mfq_train_time = time.time() - t0
                mfq_env_interactions = mfq_train["env_interactions"]
                mfq_by_mode[mode] = {
                    "q_net": mfq_q,
                    "train": mfq_train,
                    "train_time": mfq_train_time,
                    "env_interactions": mfq_env_interactions,
                }

                import torch
                torch.save(mfq_q.state_dict(), os.path.join(kind_dir, f"mfq_q_{mode}.pt"))
                if mode == mfq_primary_mode:
                    torch.save(mfq_q.state_dict(), os.path.join(kind_dir, "mfq_q.pt"))

                if wb_run is not None:
                    wb_run.log({
                        f"timing/{kind}/mfq/{mode}/train_time_s": mfq_train_time,
                        f"timing/{kind}/mfq/{mode}/env_interactions": mfq_env_interactions,
                    })

        # Train optional MFQ-local variant (faithful: density-map obs + local mean field).
        mfq_local_by_mode: dict[str, dict] = {}
        mfq_local_primary_mode = None
        if args.run_mfq_local:
            print(f"\n{'=' * 60}")
            print(f"  MFQ-local — {kind} noise  (p={args.p})  "
                  f"({mfq_local_cfg.n_batches}×{mfq_local_cfg.batch_episodes} eps each)")
            print(f"{'=' * 60}")
            mfq_local_primary_mode = mfq_local_modes[0]
            for mode in mfq_local_modes:
                cfg_mode = replace(mfq_local_cfg, obs_mode=mode)
                t0 = time.time()
                mfq_local_q, mfq_local_train = train_mfq_local(
                    grid, noise_cfg, N,
                    cfg=cfg_mode, seed=args.seed, log_every=mfq_local_log_every,
                    print_every=mfq_local_print_every,
                    wb_run=wb_run, wb_prefix=f"train_mfq_local/{kind}/{mode}",
                )
                mfq_local_train_time = time.time() - t0
                mfq_local_env_interactions = mfq_local_train["env_interactions"]
                mfq_local_by_mode[mode] = {
                    "q_net": mfq_local_q,
                    "train": mfq_local_train,
                    "train_time": mfq_local_train_time,
                    "env_interactions": mfq_local_env_interactions,
                }

                import torch
                torch.save(mfq_local_q.state_dict(),
                           os.path.join(kind_dir, f"mfq_local_q_{mode}.pt"))
                if mode == mfq_local_primary_mode:
                    torch.save(mfq_local_q.state_dict(),
                               os.path.join(kind_dir, "mfq_local_q.pt"))

                if wb_run is not None:
                    wb_run.log({
                        f"timing/{kind}/mfq_local/{mode}/train_time_s": mfq_local_train_time,
                        f"timing/{kind}/mfq_local/{mode}/env_interactions": mfq_local_env_interactions,
                    })

        # Estimate when training curves first stay above each reach-rate threshold.
        sep_reach_stabilization_by_method: dict[str, dict] = {}
        for sep_method in sep_methods:
            train_curve = (
                sep_by_method[sep_method]
                .get("train_info", {})
                .get("train_metrics", {})
                .get("train_curve", [])
            )
            sep_reach_stabilization_by_method[sep_method] = _stable_reach_milestones(
                train_curve=train_curve,
                thresholds=reach_thresholds,
                stability_window=args.stability_window,
            )
        mappo_reach_stabilization_by_mode = {
            mode: _stable_reach_milestones(
                train_curve=mappo_by_mode[mode]["train"].get("train_curve", []),
                thresholds=reach_thresholds,
                stability_window=args.stability_window,
            )
            for mode in mappo_modes
        }
        mfq_reach_stabilization_by_mode = {
            mode: _stable_reach_milestones(
                train_curve=mfq_by_mode[mode]["train"].get("train_curve", []),
                thresholds=reach_thresholds,
                stability_window=args.stability_window,
            )
            for mode in mfq_modes
        } if args.run_mfq else {}
        mfq_local_reach_stabilization_by_mode = {
            mode: _stable_reach_milestones(
                train_curve=mfq_local_by_mode[mode]["train"].get("train_curve", []),
                thresholds=reach_thresholds,
                stability_window=args.stability_window,
            )
            for mode in mfq_local_modes
        } if args.run_mfq_local else {}

        # Paired multi-seed evaluation with shared starts and targets.
        print(f"\n  Evaluating on {n_eval} seeds "
              f"(N={N} random targets each) …")
        sep_all_by_method: dict[str, List[dict]] = {m: [] for m in sep_methods}
        sep_all_by_method_rematch0: dict[str, List[dict]] = (
            {m: [] for m in sep_methods} if args.also_eval_rematch_zero else {}
        )
        mappo_all_by_mode: dict[str, List[dict]] = {m: [] for m in mappo_modes}
        ippo_all_by_mode: dict[str, List[dict]] = {m: [] for m in ippo_modes} if args.run_ippo else {}
        happo_all_by_mode: dict[str, List[dict]] = {m: [] for m in happo_modes} if args.run_happo else {}
        qmix_all_by_mode: dict[str, List[dict]] = {m: [] for m in qmix_modes} if args.run_qmix else {}
        vdn_all_by_mode: dict[str, List[dict]] = {m: [] for m in vdn_modes} if args.run_vdn else {}
        mfq_all_by_mode: dict[str, List[dict]] = {m: [] for m in mfq_modes} if args.run_mfq else {}
        mfq_local_all_by_mode: dict[str, List[dict]] = (
            {m: [] for m in mfq_local_modes} if args.run_mfq_local else {}
        )

        for s in range(n_eval):
            targets = _sample_targets(
                s, N, grid.h, grid.w, region=args.eval_target_region
            )
            start_rows = (
                [grid.h - 1] * N if args.eval_start_mode == "bottom_left" else None
            )
            collect_step_traces = args.print_eval_traces and (s < args.print_eval_max_seeds)

            for sep_method in sep_methods:
                sep_model = sep_by_method[sep_method]["model"]
                if sep_method == "q":
                    sep_res = rollout_separation(
                        sep_model, grid, noise_cfg, N, targets=targets,
                        rematch_every=args.rematch_every, seed=s,
                        start_rows=start_rows,
                        collect_step_traces=collect_step_traces,
                    )
                else:
                    sep_res = rollout_separation_ppo(
                        sep_model, grid, noise_cfg, N, targets=targets,
                        rematch_every=args.rematch_every, seed=s,
                        start_rows=start_rows,
                        collect_step_traces=collect_step_traces,
                    )
                sep_all_by_method[sep_method].append(sep_res)
                if args.also_eval_rematch_zero:
                    if sep_method == "q":
                        sep_res_r0 = rollout_separation(
                            sep_model, grid, noise_cfg, N, targets=targets,
                            rematch_every=0, seed=s,
                            start_rows=start_rows,
                            collect_step_traces=False,
                        )
                    else:
                        sep_res_r0 = rollout_separation_ppo(
                            sep_model, grid, noise_cfg, N, targets=targets,
                            rematch_every=0, seed=s,
                            start_rows=start_rows,
                            collect_step_traces=False,
                        )
                    sep_all_by_method_rematch0[sep_method].append(sep_res_r0)
            for mode in mappo_modes:
                mappo_res = rollout_mappo(
                    mappo_by_mode[mode]["actor"],
                    grid,
                    noise_cfg,
                    N,
                    targets=targets,
                    seed=s,
                    obs_mode=mode,
                    start_rows=start_rows,
                    collect_step_traces=collect_step_traces,
                )
                mappo_all_by_mode[mode].append(mappo_res)
            if args.run_ippo:
                for mode in ippo_modes:
                    ippo_res = rollout_ippo(
                        ippo_by_mode[mode]["actor"],
                        grid,
                        noise_cfg,
                        N,
                        targets=targets,
                        seed=s,
                        obs_mode=mode,
                        start_rows=start_rows,
                    )
                    ippo_all_by_mode[mode].append(ippo_res)
            if args.run_happo:
                for mode in happo_modes:
                    happo_res = rollout_happo(
                        happo_by_mode[mode]["actor"],
                        grid,
                        noise_cfg,
                        N,
                        targets=targets,
                        seed=s,
                        obs_mode=mode,
                        start_rows=start_rows,
                    )
                    happo_all_by_mode[mode].append(happo_res)
            if args.run_qmix:
                for mode in qmix_modes:
                    qmix_res = rollout_qmix(
                        qmix_by_mode[mode]["q_net"],
                        grid,
                        noise_cfg,
                        N,
                        targets=targets,
                        seed=s,
                        obs_mode=mode,
                        start_rows=start_rows,
                    )
                    qmix_all_by_mode[mode].append(qmix_res)
            if args.run_vdn:
                for mode in vdn_modes:
                    vdn_res = rollout_vdn(
                        vdn_by_mode[mode]["q_net"],
                        grid,
                        noise_cfg,
                        N,
                        targets=targets,
                        seed=s,
                        obs_mode=mode,
                        start_rows=start_rows,
                    )
                    vdn_all_by_mode[mode].append(vdn_res)
            if args.run_mfq:
                for mode in mfq_modes:
                    mfq_res = rollout_mfq(
                        mfq_by_mode[mode]["q_net"],
                        grid,
                        noise_cfg,
                        N,
                        targets=targets,
                        seed=s,
                        obs_mode=mode,
                        start_rows=start_rows,
                    )
                    mfq_all_by_mode[mode].append(mfq_res)
            if args.run_mfq_local:
                for mode in mfq_local_modes:
                    mfq_local_res = rollout_mfq_local(
                        mfq_local_by_mode[mode]["q_net"],
                        grid,
                        noise_cfg,
                        N,
                        targets=targets,
                        seed=s,
                        obs_mode=mode,
                        start_rows=start_rows,
                    )
                    mfq_local_all_by_mode[mode].append(mfq_local_res)

            if collect_step_traces:
                for sep_method in sep_methods:
                    _print_eval_trace(
                        s,
                        f"Separation[{sep_method}]",
                        sep_all_by_method[sep_method][-1],
                    )
                for mode in mappo_modes:
                    _print_eval_trace(
                        s,
                        f"MAPPO[{mode}]",
                        mappo_all_by_mode[mode][-1],
                    )
                if args.run_ippo:
                    for mode in ippo_modes:
                        _print_eval_trace(
                            s,
                            f"IPPO[{mode}]",
                            ippo_all_by_mode[mode][-1],
                        )
                if args.run_happo:
                    for mode in happo_modes:
                        _print_eval_trace(
                            s,
                            f"HAPPO[{mode}]",
                            happo_all_by_mode[mode][-1],
                        )
                if args.run_qmix:
                    for mode in qmix_modes:
                        _print_eval_trace(
                            s,
                            f"QMIX[{mode}]",
                            qmix_all_by_mode[mode][-1],
                        )
                if args.run_vdn:
                    for mode in vdn_modes:
                        _print_eval_trace(
                            s,
                            f"VDN[{mode}]",
                            vdn_all_by_mode[mode][-1],
                        )
                if args.run_mfq:
                    for mode in mfq_modes:
                        _print_eval_trace(
                            s,
                            f"MFQ[{mode}]",
                            mfq_all_by_mode[mode][-1],
                        )
                if args.run_mfq_local:
                    for mode in mfq_local_modes:
                        _print_eval_trace(
                            s,
                            f"MFQ-local[{mode}]",
                            mfq_local_all_by_mode[mode][-1],
                        )
                # Traces are for console debugging only; result files stay compact.
                for sep_method in sep_methods:
                    sep_all_by_method[sep_method][-1].pop("step_traces", None)
                for mode in mappo_modes:
                    mappo_all_by_mode[mode][-1].pop("step_traces", None)

        sep_agg_by_method = {
            m: _aggregate(sep_all_by_method[m]) for m in sep_methods
        }
        sep_agg_by_method_rematch0 = (
            {m: _aggregate(sep_all_by_method_rematch0[m]) for m in sep_methods}
            if args.also_eval_rematch_zero
            else {}
        )
        mappo_agg_by_mode = {
            mode: _aggregate(mappo_all_by_mode[mode]) for mode in mappo_modes
        }
        ippo_agg_by_mode = (
            {mode: _aggregate(ippo_all_by_mode[mode]) for mode in ippo_modes}
            if args.run_ippo else {}
        )
        happo_agg_by_mode = (
            {mode: _aggregate(happo_all_by_mode[mode]) for mode in happo_modes}
            if args.run_happo else {}
        )
        qmix_agg_by_mode = (
            {mode: _aggregate(qmix_all_by_mode[mode]) for mode in qmix_modes}
            if args.run_qmix else {}
        )
        vdn_agg_by_mode = (
            {mode: _aggregate(vdn_all_by_mode[mode]) for mode in vdn_modes}
            if args.run_vdn else {}
        )
        mfq_agg_by_mode = (
            {mode: _aggregate(mfq_all_by_mode[mode]) for mode in mfq_modes}
            if args.run_mfq else {}
        )
        mfq_local_agg_by_mode = (
            {mode: _aggregate(mfq_local_all_by_mode[mode]) for mode in mfq_local_modes}
            if args.run_mfq_local else {}
        )

        # Seed 0 is kept for plots and detailed per-algorithm outputs.
        sep_seed0_by_method = {m: sep_all_by_method[m][0] for m in sep_methods}
        sep_agg = sep_agg_by_method[primary_sep_method]
        sep_m = sep_seed0_by_method[primary_sep_method]
        mappo_seed0_by_mode = {mode: mappo_all_by_mode[mode][0] for mode in mappo_modes}
        mappo_agg = mappo_agg_by_mode.get(primary_mode, {})
        mappo_m = mappo_seed0_by_mode.get(primary_mode, {})
        ippo_seed0_by_mode = (
            {mode: ippo_all_by_mode[mode][0] for mode in ippo_modes}
            if args.run_ippo else {}
        )
        ippo_agg = ippo_agg_by_mode.get(ippo_primary_mode, {})
        ippo_m = ippo_seed0_by_mode.get(ippo_primary_mode, {})
        happo_seed0_by_mode = (
            {mode: happo_all_by_mode[mode][0] for mode in happo_modes}
            if args.run_happo else {}
        )
        happo_agg = happo_agg_by_mode.get(happo_primary_mode, {})
        happo_m = happo_seed0_by_mode.get(happo_primary_mode, {})
        qmix_seed0_by_mode = (
            {mode: qmix_all_by_mode[mode][0] for mode in qmix_modes}
            if args.run_qmix else {}
        )
        qmix_agg = qmix_agg_by_mode.get(qmix_primary_mode, {})
        qmix_m = qmix_seed0_by_mode.get(qmix_primary_mode, {})
        vdn_seed0_by_mode = (
            {mode: vdn_all_by_mode[mode][0] for mode in vdn_modes}
            if args.run_vdn else {}
        )
        vdn_agg = vdn_agg_by_mode.get(vdn_primary_mode, {})
        vdn_m = vdn_seed0_by_mode.get(vdn_primary_mode, {})
        mfq_seed0_by_mode = (
            {mode: mfq_all_by_mode[mode][0] for mode in mfq_modes}
            if args.run_mfq else {}
        )
        mfq_agg = mfq_agg_by_mode.get(mfq_primary_mode, {})
        mfq_m = mfq_seed0_by_mode.get(mfq_primary_mode, {})
        mfq_local_seed0_by_mode = (
            {mode: mfq_local_all_by_mode[mode][0] for mode in mfq_local_modes}
            if args.run_mfq_local else {}
        )
        mfq_local_agg = mfq_local_agg_by_mode.get(mfq_local_primary_mode, {})
        mfq_local_m = mfq_local_seed0_by_mode.get(mfq_local_primary_mode, {})

        for sep_method in sep_methods:
            sagg = sep_agg_by_method[sep_method]
            stime = sep_by_method[sep_method]["train_time"]
            print(f"  Sep[{sep_method}] → OT={sagg['terminal_ot_cost_mean']:.1f}"
                  f"±{sagg['terminal_ot_cost_std']:.1f}"
                  f"  reach={sagg['reach_rate_mean']:.1%}"
                  f"  cover={sagg['target_coverage_mean']:.1%}"
                  f"  t_incl={sagg['mean_time_to_reach_including_unreached_mean']:.1f}"
                  f"±{sagg['mean_time_to_reach_including_unreached_std']:.1f}"
                  f"  ({stime:.1f}s train)")
            if args.also_eval_rematch_zero:
                sagg0 = sep_agg_by_method_rematch0[sep_method]
                print(
                    f"      [rematch=0] OT={sagg0['terminal_ot_cost_mean']:.1f}"
                    f"±{sagg0['terminal_ot_cost_std']:.1f}"
                    f"  reach={sagg0['reach_rate_mean']:.1%}"
                    f"  cover={sagg0['target_coverage_mean']:.1%}"
                    f"  t_incl={sagg0['mean_time_to_reach_including_unreached_mean']:.1f}"
                    f"±{sagg0['mean_time_to_reach_including_unreached_std']:.1f}"
                )
        for mode in mappo_modes:
            agg = mappo_agg_by_mode[mode]
            tmode = mappo_by_mode[mode]["train_time"]
            print(f"  MAPPO[{mode}] → OT={agg['terminal_ot_cost_mean']:.1f}"
                  f"±{agg['terminal_ot_cost_std']:.1f}"
                  f"  reach={agg['reach_rate_mean']:.1%}"
                  f"  cover={agg['target_coverage_mean']:.1%}"
                  f"  t_incl={agg['mean_time_to_reach_including_unreached_mean']:.1f}"
                  f"±{agg['mean_time_to_reach_including_unreached_std']:.1f}"
                  f"  ({tmode:.1f}s train)")
        if args.run_ippo:
            for mode in ippo_modes:
                agg = ippo_agg_by_mode[mode]
                tmode = ippo_by_mode[mode]["train_time"]
                print(f"  IPPO[{mode}] → OT={agg['terminal_ot_cost_mean']:.1f}"
                      f"±{agg['terminal_ot_cost_std']:.1f}"
                      f"  reach={agg['reach_rate_mean']:.1%}"
                      f"  cover={agg['target_coverage_mean']:.1%}"
                      f"  t_incl={agg['mean_time_to_reach_including_unreached_mean']:.1f}"
                      f"±{agg['mean_time_to_reach_including_unreached_std']:.1f}"
                      f"  ({tmode:.1f}s train)")
        if args.run_happo:
            for mode in happo_modes:
                agg = happo_agg_by_mode[mode]
                tmode = happo_by_mode[mode]["train_time"]
                print(f"  HAPPO[{mode}] → OT={agg['terminal_ot_cost_mean']:.1f}"
                      f"±{agg['terminal_ot_cost_std']:.1f}"
                      f"  reach={agg['reach_rate_mean']:.1%}"
                      f"  cover={agg['target_coverage_mean']:.1%}"
                      f"  t_incl={agg['mean_time_to_reach_including_unreached_mean']:.1f}"
                      f"±{agg['mean_time_to_reach_including_unreached_std']:.1f}"
                      f"  ({tmode:.1f}s train)")
        if args.run_qmix:
            for mode in qmix_modes:
                agg = qmix_agg_by_mode[mode]
                tmode = qmix_by_mode[mode]["train_time"]
                print(f"  QMIX[{mode}] → OT={agg['terminal_ot_cost_mean']:.1f}"
                      f"±{agg['terminal_ot_cost_std']:.1f}"
                      f"  reach={agg['reach_rate_mean']:.1%}"
                      f"  cover={agg['target_coverage_mean']:.1%}"
                      f"  t_incl={agg['mean_time_to_reach_including_unreached_mean']:.1f}"
                      f"±{agg['mean_time_to_reach_including_unreached_std']:.1f}"
                      f"  ({tmode:.1f}s train)")
        if args.run_vdn:
            for mode in vdn_modes:
                agg = vdn_agg_by_mode[mode]
                tmode = vdn_by_mode[mode]["train_time"]
                print(f"  VDN[{mode}] → OT={agg['terminal_ot_cost_mean']:.1f}"
                      f"±{agg['terminal_ot_cost_std']:.1f}"
                      f"  reach={agg['reach_rate_mean']:.1%}"
                      f"  cover={agg['target_coverage_mean']:.1%}"
                      f"  t_incl={agg['mean_time_to_reach_including_unreached_mean']:.1f}"
                      f"±{agg['mean_time_to_reach_including_unreached_std']:.1f}"
                      f"  ({tmode:.1f}s train)")
        if args.run_mfq:
            for mode in mfq_modes:
                agg = mfq_agg_by_mode[mode]
                tmode = mfq_by_mode[mode]["train_time"]
                print(f"  MFQ[{mode}] → OT={agg['terminal_ot_cost_mean']:.1f}"
                      f"±{agg['terminal_ot_cost_std']:.1f}"
                      f"  reach={agg['reach_rate_mean']:.1%}"
                      f"  cover={agg['target_coverage_mean']:.1%}"
                      f"  t_incl={agg['mean_time_to_reach_including_unreached_mean']:.1f}"
                      f"±{agg['mean_time_to_reach_including_unreached_std']:.1f}"
                      f"  ({tmode:.1f}s train)")
        if args.run_mfq_local:
            for mode in mfq_local_modes:
                agg = mfq_local_agg_by_mode[mode]
                tmode = mfq_local_by_mode[mode]["train_time"]
                print(f"  MFQ-local[{mode}] → OT={agg['terminal_ot_cost_mean']:.1f}"
                      f"±{agg['terminal_ot_cost_std']:.1f}"
                      f"  reach={agg['reach_rate_mean']:.1%}"
                      f"  cover={agg['target_coverage_mean']:.1%}"
                      f"  t_incl={agg['mean_time_to_reach_including_unreached_mean']:.1f}"
                      f"±{agg['mean_time_to_reach_including_unreached_std']:.1f}"
                      f"  ({tmode:.1f}s train)")

        if wb_run is not None:
            import wandb
            for sep_method in sep_methods:
                sagg = sep_agg_by_method[sep_method]
                for k in _EVAL_KEYS:
                    wb_run.log({
                        f"eval/{kind}/sep/{sep_method}/{k}_mean": sagg[f"{k}_mean"],
                        f"eval/{kind}/sep/{sep_method}/{k}_std": sagg[f"{k}_std"],
                    })
            for mode in mappo_modes:
                agg = mappo_agg_by_mode[mode]
                for k in _EVAL_KEYS:
                    wb_run.log({
                        f"eval/{kind}/mappo/{mode}/{k}_mean": agg[f"{k}_mean"],
                        f"eval/{kind}/mappo/{mode}/{k}_std": agg[f"{k}_std"],
                    })
            if args.run_ippo:
                for mode in ippo_modes:
                    agg = ippo_agg_by_mode[mode]
                    for k in _EVAL_KEYS:
                        wb_run.log({
                            f"eval/{kind}/ippo/{mode}/{k}_mean": agg[f"{k}_mean"],
                            f"eval/{kind}/ippo/{mode}/{k}_std": agg[f"{k}_std"],
                        })
            if args.run_happo:
                for mode in happo_modes:
                    agg = happo_agg_by_mode[mode]
                    for k in _EVAL_KEYS:
                        wb_run.log({
                            f"eval/{kind}/happo/{mode}/{k}_mean": agg[f"{k}_mean"],
                            f"eval/{kind}/happo/{mode}/{k}_std": agg[f"{k}_std"],
                        })
            if args.run_qmix:
                for mode in qmix_modes:
                    agg = qmix_agg_by_mode[mode]
                    for k in _EVAL_KEYS:
                        wb_run.log({
                            f"eval/{kind}/qmix/{mode}/{k}_mean": agg[f"{k}_mean"],
                            f"eval/{kind}/qmix/{mode}/{k}_std": agg[f"{k}_std"],
                        })
            if args.run_vdn:
                for mode in vdn_modes:
                    agg = vdn_agg_by_mode[mode]
                    for k in _EVAL_KEYS:
                        wb_run.log({
                            f"eval/{kind}/vdn/{mode}/{k}_mean": agg[f"{k}_mean"],
                            f"eval/{kind}/vdn/{mode}/{k}_std": agg[f"{k}_std"],
                        })
            if args.run_mfq:
                for mode in mfq_modes:
                    agg = mfq_agg_by_mode[mode]
                    for k in _EVAL_KEYS:
                        wb_run.log({
                            f"eval/{kind}/mfq/{mode}/{k}_mean": agg[f"{k}_mean"],
                            f"eval/{kind}/mfq/{mode}/{k}_std": agg[f"{k}_std"],
                        })
            if args.run_mfq_local:
                for mode in mfq_local_modes:
                    agg = mfq_local_agg_by_mode[mode]
                    for k in _EVAL_KEYS:
                        wb_run.log({
                            f"eval/{kind}/mfq_local/{mode}/{k}_mean": agg[f"{k}_mean"],
                            f"eval/{kind}/mfq_local/{mode}/{k}_std": agg[f"{k}_std"],
                        })

        # Save per-noise-kind outputs.
        for sep_method in sep_methods:
            sm = sep_seed0_by_method[sep_method]
            sa = sep_agg_by_method[sep_method]
            sall = sep_all_by_method[sep_method]
            with open(os.path.join(kind_dir, f"sep_{sep_method}_metrics.json"), "w") as f:
                json.dump({"seed0": sm, "aggregate": sa, "all_seeds": sall}, f, indent=2)
            if sep_method == primary_sep_method:
                with open(os.path.join(kind_dir, "sep_metrics.json"), "w") as f:
                    json.dump({"seed0": sm, "aggregate": sa, "all_seeds": sall}, f, indent=2)

        for mode in mappo_modes:
            m_seed0 = mappo_seed0_by_mode[mode]
            m_agg = mappo_agg_by_mode[mode]
            m_all = mappo_all_by_mode[mode]
            m_train = mappo_by_mode[mode]["train"]
            with open(os.path.join(kind_dir, f"mappo_{mode}_metrics.json"), "w") as f:
                json.dump({"seed0": m_seed0, "aggregate": m_agg,
                           "all_seeds": m_all}, f, indent=2)
            with open(os.path.join(kind_dir, f"mappo_{mode}_train.json"), "w") as f:
                json.dump(m_train, f, indent=2, default=str)
            if mode == primary_mode:
                with open(os.path.join(kind_dir, "mappo_metrics.json"), "w") as f:
                    json.dump({"seed0": m_seed0, "aggregate": m_agg,
                               "all_seeds": m_all}, f, indent=2)
                with open(os.path.join(kind_dir, "mappo_train.json"), "w") as f:
                    json.dump(m_train, f, indent=2, default=str)
        if args.run_ippo:
            for mode in ippo_modes:
                i_seed0 = ippo_seed0_by_mode[mode]
                i_agg = ippo_agg_by_mode[mode]
                i_all = ippo_all_by_mode[mode]
                i_train = ippo_by_mode[mode]["train"]
                with open(os.path.join(kind_dir, f"ippo_{mode}_metrics.json"), "w") as f:
                    json.dump({"seed0": i_seed0, "aggregate": i_agg,
                               "all_seeds": i_all}, f, indent=2)
                with open(os.path.join(kind_dir, f"ippo_{mode}_train.json"), "w") as f:
                    json.dump(i_train, f, indent=2, default=str)
                if mode == ippo_primary_mode:
                    with open(os.path.join(kind_dir, "ippo_metrics.json"), "w") as f:
                        json.dump({"seed0": i_seed0, "aggregate": i_agg,
                                   "all_seeds": i_all}, f, indent=2)
                    with open(os.path.join(kind_dir, "ippo_train.json"), "w") as f:
                        json.dump(i_train, f, indent=2, default=str)
        if args.run_happo:
            for mode in happo_modes:
                h_seed0 = happo_seed0_by_mode[mode]
                h_agg = happo_agg_by_mode[mode]
                h_all = happo_all_by_mode[mode]
                h_train = happo_by_mode[mode]["train"]
                with open(os.path.join(kind_dir, f"happo_{mode}_metrics.json"), "w") as f:
                    json.dump({"seed0": h_seed0, "aggregate": h_agg,
                               "all_seeds": h_all}, f, indent=2)
                with open(os.path.join(kind_dir, f"happo_{mode}_train.json"), "w") as f:
                    json.dump(h_train, f, indent=2, default=str)
                if mode == happo_primary_mode:
                    with open(os.path.join(kind_dir, "happo_metrics.json"), "w") as f:
                        json.dump({"seed0": h_seed0, "aggregate": h_agg,
                                   "all_seeds": h_all}, f, indent=2)
                    with open(os.path.join(kind_dir, "happo_train.json"), "w") as f:
                        json.dump(h_train, f, indent=2, default=str)
        if args.run_qmix:
            for mode in qmix_modes:
                q_seed0 = qmix_seed0_by_mode[mode]
                q_agg = qmix_agg_by_mode[mode]
                q_all = qmix_all_by_mode[mode]
                q_train = qmix_by_mode[mode]["train"]
                with open(os.path.join(kind_dir, f"qmix_{mode}_metrics.json"), "w") as f:
                    json.dump({"seed0": q_seed0, "aggregate": q_agg,
                               "all_seeds": q_all}, f, indent=2)
                with open(os.path.join(kind_dir, f"qmix_{mode}_train.json"), "w") as f:
                    json.dump(q_train, f, indent=2, default=str)
                if mode == qmix_primary_mode:
                    with open(os.path.join(kind_dir, "qmix_metrics.json"), "w") as f:
                        json.dump({"seed0": q_seed0, "aggregate": q_agg,
                                   "all_seeds": q_all}, f, indent=2)
                    with open(os.path.join(kind_dir, "qmix_train.json"), "w") as f:
                        json.dump(q_train, f, indent=2, default=str)
        if args.run_vdn:
            for mode in vdn_modes:
                v_seed0 = vdn_seed0_by_mode[mode]
                v_agg = vdn_agg_by_mode[mode]
                v_all = vdn_all_by_mode[mode]
                v_train = vdn_by_mode[mode]["train"]
                with open(os.path.join(kind_dir, f"vdn_{mode}_metrics.json"), "w") as f:
                    json.dump({"seed0": v_seed0, "aggregate": v_agg,
                               "all_seeds": v_all}, f, indent=2)
                with open(os.path.join(kind_dir, f"vdn_{mode}_train.json"), "w") as f:
                    json.dump(v_train, f, indent=2, default=str)
                if mode == vdn_primary_mode:
                    with open(os.path.join(kind_dir, "vdn_metrics.json"), "w") as f:
                        json.dump({"seed0": v_seed0, "aggregate": v_agg,
                                   "all_seeds": v_all}, f, indent=2)
                    with open(os.path.join(kind_dir, "vdn_train.json"), "w") as f:
                        json.dump(v_train, f, indent=2, default=str)
        if args.run_mfq:
            for mode in mfq_modes:
                mf_seed0 = mfq_seed0_by_mode[mode]
                mf_agg = mfq_agg_by_mode[mode]
                mf_all = mfq_all_by_mode[mode]
                mf_train = mfq_by_mode[mode]["train"]
                with open(os.path.join(kind_dir, f"mfq_{mode}_metrics.json"), "w") as f:
                    json.dump({"seed0": mf_seed0, "aggregate": mf_agg,
                               "all_seeds": mf_all}, f, indent=2)
                with open(os.path.join(kind_dir, f"mfq_{mode}_train.json"), "w") as f:
                    json.dump(mf_train, f, indent=2, default=str)
                if mode == mfq_primary_mode:
                    with open(os.path.join(kind_dir, "mfq_metrics.json"), "w") as f:
                        json.dump({"seed0": mf_seed0, "aggregate": mf_agg,
                                   "all_seeds": mf_all}, f, indent=2)
                    with open(os.path.join(kind_dir, "mfq_train.json"), "w") as f:
                        json.dump(mf_train, f, indent=2, default=str)
        if args.run_mfq_local:
            for mode in mfq_local_modes:
                mfl_seed0 = mfq_local_seed0_by_mode[mode]
                mfl_agg = mfq_local_agg_by_mode[mode]
                mfl_all = mfq_local_all_by_mode[mode]
                mfl_train = mfq_local_by_mode[mode]["train"]
                with open(os.path.join(kind_dir, f"mfq_local_{mode}_metrics.json"), "w") as f:
                    json.dump({"seed0": mfl_seed0, "aggregate": mfl_agg,
                               "all_seeds": mfl_all}, f, indent=2)
                with open(os.path.join(kind_dir, f"mfq_local_{mode}_train.json"), "w") as f:
                    json.dump(mfl_train, f, indent=2, default=str)
                if mode == mfq_local_primary_mode:
                    with open(os.path.join(kind_dir, "mfq_local_metrics.json"), "w") as f:
                        json.dump({"seed0": mfl_seed0, "aggregate": mfl_agg,
                                   "all_seeds": mfl_all}, f, indent=2)
                    with open(os.path.join(kind_dir, "mfq_local_train.json"), "w") as f:
                        json.dump(mfl_train, f, indent=2, default=str)

        # Dedicated visualization rollout with shared starts and targets.
        viz_seed = args.seed + 900_000
        viz_rng = np.random.default_rng(viz_seed)
        viz_start_rows = (
            [grid.h - 1] * N
            if args.eval_start_mode == "bottom_left"
            else viz_rng.integers(0, grid.h, size=N).astype(int).tolist()
        )
        viz_targets = (
            list(fixed_targets)
            if fixed_targets is not None
            else _sample_targets(
                viz_seed, N, grid.h, grid.w, region=args.eval_target_region
            )
        )

        viz_sep_by_method: dict[str, dict] = {}
        for sep_method in sep_methods:
            sep_model = sep_by_method[sep_method]["model"]
            if sep_method == "q":
                viz_sep_by_method[sep_method] = rollout_separation(
                    sep_model,
                    grid,
                    noise_cfg,
                    N,
                    targets=viz_targets,
                    rematch_every=args.rematch_every,
                    seed=viz_seed,
                    start_rows=viz_start_rows,
                )
            else:
                viz_sep_by_method[sep_method] = rollout_separation_ppo(
                    sep_model,
                    grid,
                    noise_cfg,
                    N,
                    targets=viz_targets,
                    rematch_every=args.rematch_every,
                    seed=viz_seed,
                    start_rows=viz_start_rows,
                )

        viz_mappo = (
            rollout_mappo(
                mappo_by_mode[primary_mode]["actor"],
                grid,
                noise_cfg,
                N,
                targets=viz_targets,
                seed=viz_seed,
                obs_mode=primary_mode,
                start_rows=viz_start_rows,
            )
            if args.run_mappo and primary_mode is not None
            else None
        )
        viz_ippo = (
            rollout_ippo(
                ippo_by_mode[ippo_primary_mode]["actor"],
                grid,
                noise_cfg,
                N,
                targets=viz_targets,
                seed=viz_seed,
                obs_mode=ippo_primary_mode,
                start_rows=viz_start_rows,
            )
            if args.run_ippo and ippo_primary_mode is not None
            else None
        )
        viz_happo = (
            rollout_happo(
                happo_by_mode[happo_primary_mode]["actor"],
                grid,
                noise_cfg,
                N,
                targets=viz_targets,
                seed=viz_seed,
                obs_mode=happo_primary_mode,
                start_rows=viz_start_rows,
            )
            if args.run_happo and happo_primary_mode is not None
            else None
        )
        viz_qmix = (
            rollout_qmix(
                qmix_by_mode[qmix_primary_mode]["q_net"],
                grid,
                noise_cfg,
                N,
                targets=viz_targets,
                seed=viz_seed,
                obs_mode=qmix_primary_mode,
                start_rows=viz_start_rows,
            )
            if args.run_qmix and qmix_primary_mode is not None
            else None
        )
        viz_vdn = (
            rollout_vdn(
                vdn_by_mode[vdn_primary_mode]["q_net"],
                grid,
                noise_cfg,
                N,
                targets=viz_targets,
                seed=viz_seed,
                obs_mode=vdn_primary_mode,
                start_rows=viz_start_rows,
            )
            if args.run_vdn and vdn_primary_mode is not None
            else None
        )
        viz_mfq = (
            rollout_mfq(
                mfq_by_mode[mfq_primary_mode]["q_net"],
                grid,
                noise_cfg,
                N,
                targets=viz_targets,
                seed=viz_seed,
                obs_mode=mfq_primary_mode,
                start_rows=viz_start_rows,
            )
            if args.run_mfq and mfq_primary_mode is not None
            else None
        )
        viz_mfq_local = (
            rollout_mfq_local(
                mfq_local_by_mode[mfq_local_primary_mode]["q_net"],
                grid,
                noise_cfg,
                N,
                targets=viz_targets,
                seed=viz_seed,
                obs_mode=mfq_local_primary_mode,
                start_rows=viz_start_rows,
            )
            if args.run_mfq_local and mfq_local_primary_mode is not None
            else None
        )

        # Terminal histograms from seed-0 rollout.
        targets_0 = [tuple(t) for t in sep_m["targets"]]
        trajectories_overlay: Dict[str, List[List[Pos]]] = {}
        for sep_method in sep_methods:
            trajectories_overlay[f"Separation[{sep_method}]"] = [
                [(int(p[0]), int(p[1])) for p in traj]
                for traj in viz_sep_by_method[sep_method]["trajectories"]
            ]
        if viz_mappo is not None:
            trajectories_overlay[f"MAPPO[{primary_mode}]"] = [
                [(int(p[0]), int(p[1])) for p in traj]
                for traj in viz_mappo["trajectories"]
            ]
        if viz_ippo is not None:
            trajectories_overlay[f"IPPO[{ippo_primary_mode}]"] = [
                [(int(p[0]), int(p[1])) for p in traj]
                for traj in viz_ippo["trajectories"]
            ]
        if viz_happo is not None:
            trajectories_overlay[f"HAPPO[{happo_primary_mode}]"] = [
                [(int(p[0]), int(p[1])) for p in traj]
                for traj in viz_happo["trajectories"]
            ]
        if viz_qmix is not None:
            trajectories_overlay[f"QMIX[{qmix_primary_mode}]"] = [
                [(int(p[0]), int(p[1])) for p in traj]
                for traj in viz_qmix["trajectories"]
            ]
        if viz_vdn is not None:
            trajectories_overlay[f"VDN[{vdn_primary_mode}]"] = [
                [(int(p[0]), int(p[1])) for p in traj]
                for traj in viz_vdn["trajectories"]
            ]
        if viz_mfq is not None:
            trajectories_overlay[f"MFQ[{mfq_primary_mode}]"] = [
                [(int(p[0]), int(p[1])) for p in traj]
                for traj in viz_mfq["trajectories"]
            ]
        if viz_mfq_local is not None:
            trajectories_overlay[f"MFQ-local[{mfq_local_primary_mode}]"] = [
                [(int(p[0]), int(p[1])) for p in traj]
                for traj in viz_mfq_local["trajectories"]
            ]
        eval_traj_data_path = os.path.join(kind_dir, "eval_trajectories_viz.json")
        viz_payload = {
            "grid_h": int(grid.h),
            "grid_w": int(grid.w),
            "noise_kind": kind,
            "noise_p": float(args.p),
            "viz_seed": int(viz_seed),
            "viz_start_rows": [int(x) for x in viz_start_rows],
            "viz_targets": [[int(t[0]), int(t[1])] for t in viz_targets],
            "trajectories_by_algo": {
                algo: [
                    [[int(p[0]), int(p[1])] for p in traj]
                    for traj in trajs
                ]
                for algo, trajs in trajectories_overlay.items()
            },
        }
        with open(eval_traj_data_path, "w") as f:
            json.dump(viz_payload, f, indent=2)

        sep_hist_paths: dict[str, str] = {}
        for sep_method in sep_methods:
            out = os.path.join(kind_dir, f"terminal_hist_sep_{sep_method}.png")
            sep_hist_paths[sep_method] = out
            plot_terminal_hist(
                grid.h, grid.w,
                agents_terminal=sep_seed0_by_method[sep_method]["final_positions"], targets=targets_0,
                outpath=out,
                title=f"Separation[{sep_method}] — {kind} noise (p={args.p})",
            )
            if sep_method == primary_sep_method:
                plot_terminal_hist(
                    grid.h, grid.w,
                    agents_terminal=sep_seed0_by_method[sep_method]["final_positions"], targets=targets_0,
                    outpath=os.path.join(kind_dir, "terminal_hist_sep.png"),
                    title=f"Separation — {kind} noise (p={args.p})",
                )
        mappo_hist_paths: dict[str, str] = {}
        for mode in mappo_modes:
            mappo_hist_path = os.path.join(kind_dir, f"terminal_hist_mappo_{mode}.png")
            mappo_hist_paths[mode] = mappo_hist_path
            plot_terminal_hist(
                grid.h, grid.w,
                agents_terminal=mappo_seed0_by_mode[mode]["final_positions"], targets=targets_0,
                outpath=mappo_hist_path,
                title=f"MAPPO[{mode}] — {kind} noise (p={args.p})",
            )
            if mode == primary_mode:
                plot_terminal_hist(
                    grid.h, grid.w,
                    agents_terminal=mappo_seed0_by_mode[mode]["final_positions"], targets=targets_0,
                    outpath=os.path.join(kind_dir, "terminal_hist_mappo.png"),
                    title=f"MAPPO — {kind} noise (p={args.p})",
                )
        ippo_hist_paths: dict[str, str] = {}
        if args.run_ippo:
            for mode in ippo_modes:
                ippo_hist_path = os.path.join(kind_dir, f"terminal_hist_ippo_{mode}.png")
                ippo_hist_paths[mode] = ippo_hist_path
                plot_terminal_hist(
                    grid.h, grid.w,
                    agents_terminal=ippo_seed0_by_mode[mode]["final_positions"], targets=targets_0,
                    outpath=ippo_hist_path,
                    title=f"IPPO[{mode}] — {kind} noise (p={args.p})",
                )
                if mode == ippo_primary_mode:
                    plot_terminal_hist(
                        grid.h, grid.w,
                        agents_terminal=ippo_seed0_by_mode[mode]["final_positions"], targets=targets_0,
                        outpath=os.path.join(kind_dir, "terminal_hist_ippo.png"),
                        title=f"IPPO — {kind} noise (p={args.p})",
                    )
        happo_hist_paths: dict[str, str] = {}
        if args.run_happo:
            for mode in happo_modes:
                happo_hist_path = os.path.join(kind_dir, f"terminal_hist_happo_{mode}.png")
                happo_hist_paths[mode] = happo_hist_path
                plot_terminal_hist(
                    grid.h, grid.w,
                    agents_terminal=happo_seed0_by_mode[mode]["final_positions"], targets=targets_0,
                    outpath=happo_hist_path,
                    title=f"HAPPO[{mode}] — {kind} noise (p={args.p})",
                )
                if mode == happo_primary_mode:
                    plot_terminal_hist(
                        grid.h, grid.w,
                        agents_terminal=happo_seed0_by_mode[mode]["final_positions"], targets=targets_0,
                        outpath=os.path.join(kind_dir, "terminal_hist_happo.png"),
                        title=f"HAPPO — {kind} noise (p={args.p})",
                    )
        qmix_hist_paths: dict[str, str] = {}
        if args.run_qmix:
            for mode in qmix_modes:
                qmix_hist_path = os.path.join(kind_dir, f"terminal_hist_qmix_{mode}.png")
                qmix_hist_paths[mode] = qmix_hist_path
                plot_terminal_hist(
                    grid.h, grid.w,
                    agents_terminal=qmix_seed0_by_mode[mode]["final_positions"], targets=targets_0,
                    outpath=qmix_hist_path,
                    title=f"QMIX[{mode}] — {kind} noise (p={args.p})",
                )
                if mode == qmix_primary_mode:
                    plot_terminal_hist(
                        grid.h, grid.w,
                        agents_terminal=qmix_seed0_by_mode[mode]["final_positions"], targets=targets_0,
                        outpath=os.path.join(kind_dir, "terminal_hist_qmix.png"),
                        title=f"QMIX — {kind} noise (p={args.p})",
                    )
        vdn_hist_paths: dict[str, str] = {}
        if args.run_vdn:
            for mode in vdn_modes:
                vdn_hist_path = os.path.join(kind_dir, f"terminal_hist_vdn_{mode}.png")
                vdn_hist_paths[mode] = vdn_hist_path
                plot_terminal_hist(
                    grid.h, grid.w,
                    agents_terminal=vdn_seed0_by_mode[mode]["final_positions"], targets=targets_0,
                    outpath=vdn_hist_path,
                    title=f"VDN[{mode}] — {kind} noise (p={args.p})",
                )
                if mode == vdn_primary_mode:
                    plot_terminal_hist(
                        grid.h, grid.w,
                        agents_terminal=vdn_seed0_by_mode[mode]["final_positions"], targets=targets_0,
                        outpath=os.path.join(kind_dir, "terminal_hist_vdn.png"),
                        title=f"VDN — {kind} noise (p={args.p})",
                    )
        mfq_hist_paths: dict[str, str] = {}
        if args.run_mfq:
            for mode in mfq_modes:
                mfq_hist_path = os.path.join(kind_dir, f"terminal_hist_mfq_{mode}.png")
                mfq_hist_paths[mode] = mfq_hist_path
                plot_terminal_hist(
                    grid.h, grid.w,
                    agents_terminal=mfq_seed0_by_mode[mode]["final_positions"], targets=targets_0,
                    outpath=mfq_hist_path,
                    title=f"MFQ[{mode}] — {kind} noise (p={args.p})",
                )
                if mode == mfq_primary_mode:
                    plot_terminal_hist(
                        grid.h, grid.w,
                        agents_terminal=mfq_seed0_by_mode[mode]["final_positions"], targets=targets_0,
                        outpath=os.path.join(kind_dir, "terminal_hist_mfq.png"),
                        title=f"MFQ — {kind} noise (p={args.p})",
                    )

        if wb_run is not None:
            import wandb
            img_payload = {}
            for sep_method in sep_methods:
                img_payload[f"plots/{kind}/terminal_hist_sep_{sep_method}"] = wandb.Image(
                    sep_hist_paths[sep_method]
                )
            for mode in mappo_modes:
                img_payload[f"plots/{kind}/terminal_hist_mappo_{mode}"] = wandb.Image(
                    mappo_hist_paths[mode]
                )
            if args.run_ippo:
                for mode in ippo_modes:
                    img_payload[f"plots/{kind}/terminal_hist_ippo_{mode}"] = wandb.Image(
                        ippo_hist_paths[mode]
                    )
            if args.run_happo:
                for mode in happo_modes:
                    img_payload[f"plots/{kind}/terminal_hist_happo_{mode}"] = wandb.Image(
                        happo_hist_paths[mode]
                    )
            if args.run_qmix:
                for mode in qmix_modes:
                    img_payload[f"plots/{kind}/terminal_hist_qmix_{mode}"] = wandb.Image(
                        qmix_hist_paths[mode]
                    )
            if args.run_vdn:
                for mode in vdn_modes:
                    img_payload[f"plots/{kind}/terminal_hist_vdn_{mode}"] = wandb.Image(
                        vdn_hist_paths[mode]
                    )
            if args.run_mfq:
                for mode in mfq_modes:
                    img_payload[f"plots/{kind}/terminal_hist_mfq_{mode}"] = wandb.Image(
                        mfq_hist_paths[mode]
                    )
            wb_run.log(img_payload)

        all_results.append({
            "noise": kind,
            "reach_thresholds": reach_thresholds,
            "stability_window_batches": int(args.stability_window),
            "sep_method": primary_sep_method,
            "sep_methods": sep_methods,
            "sep_reach_stabilization": sep_reach_stabilization_by_method.get(
                primary_sep_method, {}
            ),
            "sep_reach_stabilization_by_method": sep_reach_stabilization_by_method,
            "sep_train_info_by_method": {m: v["train_info"] for m, v in sep_by_method.items()},
            "sep_agg": sep_agg,
            "sep_agg_by_method": sep_agg_by_method,
            "sep_eval_rematch_every_primary": int(args.rematch_every),
            "sep_agg_by_method_rematch0": sep_agg_by_method_rematch0,
            "sep_seed0_by_method_rematch0": (
                {m: sep_all_by_method_rematch0[m][0] for m in sep_methods}
                if args.also_eval_rematch_zero
                else {}
            ),
            "mappo_agg": mappo_agg,
            "mappo_mode_primary": primary_mode,
            "mappo_reach_stabilization": mappo_reach_stabilization_by_mode.get(
                primary_mode, {}
            ),
            "mappo_reach_stabilization_by_mode": mappo_reach_stabilization_by_mode,
            "mfq_reach_stabilization": mfq_reach_stabilization_by_mode.get(
                mfq_primary_mode, {}
            ) if args.run_mfq else {},
            "mfq_reach_stabilization_by_mode": mfq_reach_stabilization_by_mode,
            "mfq_local_reach_stabilization": mfq_local_reach_stabilization_by_mode.get(
                mfq_local_primary_mode, {}
            ) if args.run_mfq_local else {},
            "mfq_local_reach_stabilization_by_mode": mfq_local_reach_stabilization_by_mode,
            "mappo_agg_by_mode": mappo_agg_by_mode,
            "ippo_agg": ippo_agg if args.run_ippo else {},
            "ippo_mode_primary": ippo_primary_mode if args.run_ippo else None,
            "ippo_agg_by_mode": ippo_agg_by_mode if args.run_ippo else {},
            "happo_agg": happo_agg if args.run_happo else {},
            "happo_mode_primary": happo_primary_mode if args.run_happo else None,
            "happo_agg_by_mode": happo_agg_by_mode if args.run_happo else {},
            "qmix_agg": qmix_agg if args.run_qmix else {},
            "qmix_mode_primary": qmix_primary_mode if args.run_qmix else None,
            "qmix_agg_by_mode": qmix_agg_by_mode if args.run_qmix else {},
            "vdn_agg": vdn_agg if args.run_vdn else {},
            "vdn_mode_primary": vdn_primary_mode if args.run_vdn else None,
            "vdn_agg_by_mode": vdn_agg_by_mode if args.run_vdn else {},
            "mfq_agg": mfq_agg if args.run_mfq else {},
            "mfq_mode_primary": mfq_primary_mode if args.run_mfq else None,
            "mfq_agg_by_mode": mfq_agg_by_mode if args.run_mfq else {},
            "mfq_local_agg": mfq_local_agg if args.run_mfq_local else {},
            "mfq_local_mode_primary": mfq_local_primary_mode if args.run_mfq_local else None,
            "mfq_local_agg_by_mode": mfq_local_agg_by_mode if args.run_mfq_local else {},
            "sep_seed0": sep_m,
            "sep_seed0_by_method": sep_seed0_by_method,
            "mappo_seed0": mappo_m,
            "mappo_seed0_by_mode": mappo_seed0_by_mode,
            "ippo_seed0": ippo_m if args.run_ippo else {},
            "ippo_seed0_by_mode": ippo_seed0_by_mode if args.run_ippo else {},
            "happo_seed0": happo_m if args.run_happo else {},
            "happo_seed0_by_mode": happo_seed0_by_mode if args.run_happo else {},
            "qmix_seed0": qmix_m if args.run_qmix else {},
            "qmix_seed0_by_mode": qmix_seed0_by_mode if args.run_qmix else {},
            "vdn_seed0": vdn_m if args.run_vdn else {},
            "vdn_seed0_by_mode": vdn_seed0_by_mode if args.run_vdn else {},
            "mfq_seed0": mfq_m if args.run_mfq else {},
            "mfq_seed0_by_mode": mfq_seed0_by_mode if args.run_mfq else {},
            "mfq_local_seed0": mfq_local_m if args.run_mfq_local else {},
            "mfq_local_seed0_by_mode": mfq_local_seed0_by_mode if args.run_mfq_local else {},
            "sep_train_time": sep_by_method[primary_sep_method]["train_time"],
            "sep_env_interactions": sep_by_method[primary_sep_method]["env_interactions"],
            "sep_train_time_by_method": {m: v["train_time"] for m, v in sep_by_method.items()},
            "sep_env_interactions_by_method": {
                m: v["env_interactions"] for m, v in sep_by_method.items()
            },
            "mappo_train_time": mappo_by_mode[primary_mode]["train_time"] if args.run_mappo and primary_mode is not None else float("nan"),
            "mappo_env_interactions": mappo_by_mode[primary_mode]["env_interactions"] if args.run_mappo and primary_mode is not None else 0,
            "mappo_train_by_mode": {m: v["train"] for m, v in mappo_by_mode.items()},
            "mappo_train_time_by_mode": {m: v["train_time"] for m, v in mappo_by_mode.items()},
            "mappo_env_interactions_by_mode": {
                m: v["env_interactions"] for m, v in mappo_by_mode.items()
            },
            "ippo_train_by_mode": {m: v["train"] for m, v in ippo_by_mode.items()} if args.run_ippo else {},
            "ippo_train_time_by_mode": {m: v["train_time"] for m, v in ippo_by_mode.items()} if args.run_ippo else {},
            "ippo_env_interactions_by_mode": {
                m: v["env_interactions"] for m, v in ippo_by_mode.items()
            } if args.run_ippo else {},
            "happo_train_by_mode": {m: v["train"] for m, v in happo_by_mode.items()} if args.run_happo else {},
            "happo_train_time_by_mode": {m: v["train_time"] for m, v in happo_by_mode.items()} if args.run_happo else {},
            "happo_env_interactions_by_mode": {
                m: v["env_interactions"] for m, v in happo_by_mode.items()
            } if args.run_happo else {},
            "qmix_train_by_mode": {m: v["train"] for m, v in qmix_by_mode.items()} if args.run_qmix else {},
            "qmix_train_time_by_mode": {m: v["train_time"] for m, v in qmix_by_mode.items()} if args.run_qmix else {},
            "qmix_env_interactions_by_mode": {
                m: v["env_interactions"] for m, v in qmix_by_mode.items()
            } if args.run_qmix else {},
            "vdn_train_by_mode": {m: v["train"] for m, v in vdn_by_mode.items()} if args.run_vdn else {},
            "vdn_train_time_by_mode": {m: v["train_time"] for m, v in vdn_by_mode.items()} if args.run_vdn else {},
            "vdn_env_interactions_by_mode": {
                m: v["env_interactions"] for m, v in vdn_by_mode.items()
            } if args.run_vdn else {},
            "mfq_train_by_mode": {m: v["train"] for m, v in mfq_by_mode.items()} if args.run_mfq else {},
            "mfq_train_time_by_mode": {m: v["train_time"] for m, v in mfq_by_mode.items()} if args.run_mfq else {},
            "mfq_env_interactions_by_mode": {
                m: v["env_interactions"] for m, v in mfq_by_mode.items()
            } if args.run_mfq else {},
            "mfq_local_train_by_mode": {m: v["train"] for m, v in mfq_local_by_mode.items()} if args.run_mfq_local else {},
            "mfq_local_train_time_by_mode": {m: v["train_time"] for m, v in mfq_local_by_mode.items()} if args.run_mfq_local else {},
            "mfq_local_env_interactions_by_mode": {
                m: v["env_interactions"] for m, v in mfq_local_by_mode.items()
            } if args.run_mfq_local else {},
        })

    hdr_w = 104
    print(f"\n{'=' * hdr_w}")
    print(f"  COMPARISON  ({n_eval}-seed eval, N={N} agents, N={N} random targets per episode)")
    print(f"{'=' * hdr_w}")
    print(f"{'noise':<12}  {'algorithm':<22} {'OT↓':>12} {'reach%↑':>12} {'cover%↑':>12} {'mean_t↓':>12}")
    print("─" * hdr_w)
    for r in all_results:
        kind = r["noise"]
        first_sep_row = True
        for sep_method in sep_methods:
            sa = r["sep_agg_by_method"][sep_method]
            s_ot = _fmt(sa["terminal_ot_cost_mean"], sa["terminal_ot_cost_std"])
            s_rr = _fmt(sa["reach_rate_mean"] * 100,
                        sa["reach_rate_std"] * 100, prec=0)
            s_cv = _fmt(sa["target_coverage_mean"] * 100,
                        sa["target_coverage_std"] * 100, prec=0)
            s_mt = _fmt(sa["mean_time_to_reach_mean"],
                        sa["mean_time_to_reach_std"])
            sep_label = f"Separation[{sep_method}]"
            if first_sep_row:
                print(f"{kind:<12}  {sep_label:<22} {s_ot:>12} {s_rr:>12} {s_cv:>12} {s_mt:>12}")
                first_sep_row = False
            else:
                print(f"{'':12}  {sep_label:<22} {s_ot:>12} {s_rr:>12} {s_cv:>12} {s_mt:>12}")
        for mode in mappo_modes:
            ma = r["mappo_agg_by_mode"][mode]
            m_ot = _fmt(ma["terminal_ot_cost_mean"], ma["terminal_ot_cost_std"])
            m_rr = _fmt(ma["reach_rate_mean"] * 100,
                        ma["reach_rate_std"] * 100, prec=0)
            m_cv = _fmt(ma["target_coverage_mean"] * 100,
                        ma["target_coverage_std"] * 100, prec=0)
            m_mt = _fmt(ma["mean_time_to_reach_mean"],
                        ma["mean_time_to_reach_std"])
            print(f"{'':12}  {f'MAPPO[{mode}]':<22} {m_ot:>12} {m_rr:>12} {m_cv:>12} {m_mt:>12}")
        if args.run_ippo:
            for mode in ippo_modes:
                ia = r["ippo_agg_by_mode"][mode]
                i_ot = _fmt(ia["terminal_ot_cost_mean"], ia["terminal_ot_cost_std"])
                i_rr = _fmt(ia["reach_rate_mean"] * 100,
                            ia["reach_rate_std"] * 100, prec=0)
                i_cv = _fmt(ia["target_coverage_mean"] * 100,
                            ia["target_coverage_std"] * 100, prec=0)
                i_mt = _fmt(ia["mean_time_to_reach_mean"],
                            ia["mean_time_to_reach_std"])
                print(f"{'':12}  {f'IPPO[{mode}]':<22} {i_ot:>12} {i_rr:>12} {i_cv:>12} {i_mt:>12}")
        if args.run_happo:
            for mode in happo_modes:
                ha = r["happo_agg_by_mode"][mode]
                h_ot = _fmt(ha["terminal_ot_cost_mean"], ha["terminal_ot_cost_std"])
                h_rr = _fmt(ha["reach_rate_mean"] * 100,
                            ha["reach_rate_std"] * 100, prec=0)
                h_cv = _fmt(ha["target_coverage_mean"] * 100,
                            ha["target_coverage_std"] * 100, prec=0)
                h_mt = _fmt(ha["mean_time_to_reach_mean"],
                            ha["mean_time_to_reach_std"])
                print(f"{'':12}  {f'HAPPO[{mode}]':<22} {h_ot:>12} {h_rr:>12} {h_cv:>12} {h_mt:>12}")
        if args.run_qmix:
            for mode in qmix_modes:
                qa = r["qmix_agg_by_mode"][mode]
                q_ot = _fmt(qa["terminal_ot_cost_mean"], qa["terminal_ot_cost_std"])
                q_rr = _fmt(qa["reach_rate_mean"] * 100,
                            qa["reach_rate_std"] * 100, prec=0)
                q_cv = _fmt(qa["target_coverage_mean"] * 100,
                            qa["target_coverage_std"] * 100, prec=0)
                q_mt = _fmt(qa["mean_time_to_reach_mean"],
                            qa["mean_time_to_reach_std"])
                print(f"{'':12}  {f'QMIX[{mode}]':<22} {q_ot:>12} {q_rr:>12} {q_cv:>12} {q_mt:>12}")
        if args.run_vdn:
            for mode in vdn_modes:
                va = r["vdn_agg_by_mode"][mode]
                v_ot = _fmt(va["terminal_ot_cost_mean"], va["terminal_ot_cost_std"])
                v_rr = _fmt(va["reach_rate_mean"] * 100,
                            va["reach_rate_std"] * 100, prec=0)
                v_cv = _fmt(va["target_coverage_mean"] * 100,
                            va["target_coverage_std"] * 100, prec=0)
                v_mt = _fmt(va["mean_time_to_reach_mean"],
                            va["mean_time_to_reach_std"])
                print(f"{'':12}  {f'VDN[{mode}]':<22} {v_ot:>12} {v_rr:>12} {v_cv:>12} {v_mt:>12}")
        if args.run_mfq:
            for mode in mfq_modes:
                fa = r["mfq_agg_by_mode"][mode]
                f_ot = _fmt(fa["terminal_ot_cost_mean"], fa["terminal_ot_cost_std"])
                f_rr = _fmt(fa["reach_rate_mean"] * 100,
                            fa["reach_rate_std"] * 100, prec=0)
                f_cv = _fmt(fa["target_coverage_mean"] * 100,
                            fa["target_coverage_std"] * 100, prec=0)
                f_mt = _fmt(fa["mean_time_to_reach_mean"],
                            fa["mean_time_to_reach_std"])
                print(f"{'':12}  {f'MFQ[{mode}]':<22} {f_ot:>12} {f_rr:>12} {f_cv:>12} {f_mt:>12}")
        if args.run_mfq_local:
            for mode in mfq_local_modes:
                la = r["mfq_local_agg_by_mode"][mode]
                l_ot = _fmt(la["terminal_ot_cost_mean"], la["terminal_ot_cost_std"])
                l_rr = _fmt(la["reach_rate_mean"] * 100,
                            la["reach_rate_std"] * 100, prec=0)
                l_cv = _fmt(la["target_coverage_mean"] * 100,
                            la["target_coverage_std"] * 100, prec=0)
                l_mt = _fmt(la["mean_time_to_reach_mean"],
                            la["mean_time_to_reach_std"])
                print(f"{'':12}  {f'MFQ-local[{mode}]':<22} {l_ot:>12} {l_rr:>12} {l_cv:>12} {l_mt:>12}")
        print("─" * hdr_w)

    print(f"\nTraining efficiency:")
    for r in all_results:
        kind = r["noise"]
        first_sep_row = True
        for sep_method in sep_methods:
            s_int = r["sep_env_interactions_by_method"][sep_method]
            s_t = r["sep_train_time_by_method"][sep_method]
            prefix = f"  {kind:12s}" if first_sep_row else " " * 14
            print(f"{prefix}  Sep[{sep_method}]: {s_int/1e6:.1f}M interactions, {s_t:.1f}s")
            first_sep_row = False
        for mode in mappo_modes:
            m_int = r["mappo_env_interactions_by_mode"][mode]
            m_t = r["mappo_train_time_by_mode"][mode]
            print(f"               MAPPO[{mode:<13}] {m_int/1e6:.1f}M interactions, {m_t:.1f}s")
        if args.run_ippo:
            for mode in ippo_modes:
                i_int = r["ippo_env_interactions_by_mode"][mode]
                i_t = r["ippo_train_time_by_mode"][mode]
                print(f"               IPPO[{mode:<14}] {i_int/1e6:.1f}M interactions, {i_t:.1f}s")
        if args.run_happo:
            for mode in happo_modes:
                h_int = r["happo_env_interactions_by_mode"][mode]
                h_t = r["happo_train_time_by_mode"][mode]
                print(f"               HAPPO[{mode:<13}] {h_int/1e6:.1f}M interactions, {h_t:.1f}s")
        if args.run_qmix:
            for mode in qmix_modes:
                q_int = r["qmix_env_interactions_by_mode"][mode]
                q_t = r["qmix_train_time_by_mode"][mode]
                print(f"               QMIX[{mode:<14}] {q_int/1e6:.1f}M interactions, {q_t:.1f}s")
        if args.run_vdn:
            for mode in vdn_modes:
                v_int = r["vdn_env_interactions_by_mode"][mode]
                v_t = r["vdn_train_time_by_mode"][mode]
                print(f"               VDN[{mode:<15}] {v_int/1e6:.1f}M interactions, {v_t:.1f}s")
        if args.run_mfq:
            for mode in mfq_modes:
                f_int = r["mfq_env_interactions_by_mode"][mode]
                f_t = r["mfq_train_time_by_mode"][mode]
                print(f"               MFQ[{mode:<15}] {f_int/1e6:.1f}M interactions, {f_t:.1f}s")
        if args.run_mfq_local:
            for mode in mfq_local_modes:
                l_int = r["mfq_local_env_interactions_by_mode"][mode]
                l_t = r["mfq_local_train_time_by_mode"][mode]
                print(f"               MFQ-local[{mode:<9}] {l_int/1e6:.1f}M interactions, {l_t:.1f}s")

    with open(os.path.join(base_outdir, "comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    if wb_run is not None:
        import wandb

        table = wandb.Table(columns=[
            "noise", "algorithm",
            "OT_mean", "OT_std",
            "reach_rate_mean", "reach_rate_std",
            "coverage_mean", "coverage_std",
            "mean_time_mean", "mean_time_std",
            "total_cost_mean", "total_cost_std",
            "train_time_s", "env_interactions",
        ])
        for r in all_results:
            kind = r["noise"]
            for sep_method in sep_methods:
                algo = f"Separation[{sep_method}]"
                agg = r["sep_agg_by_method"][sep_method]
                tt = r["sep_train_time_by_method"][sep_method]
                ei = r["sep_env_interactions_by_method"][sep_method]
                table.add_data(
                    kind, algo,
                    agg["terminal_ot_cost_mean"], agg["terminal_ot_cost_std"],
                    agg["reach_rate_mean"], agg["reach_rate_std"],
                    agg["target_coverage_mean"], agg["target_coverage_std"],
                    agg["mean_time_to_reach_mean"], agg["mean_time_to_reach_std"],
                    agg["total_cost_mean"], agg["total_cost_std"],
                    tt, ei,
                )
            for mode in mappo_modes:
                agg = r["mappo_agg_by_mode"][mode]
                table.add_data(
                    kind, f"MAPPO[{mode}]",
                    agg["terminal_ot_cost_mean"], agg["terminal_ot_cost_std"],
                    agg["reach_rate_mean"], agg["reach_rate_std"],
                    agg["target_coverage_mean"], agg["target_coverage_std"],
                    agg["mean_time_to_reach_mean"], agg["mean_time_to_reach_std"],
                    agg["total_cost_mean"], agg["total_cost_std"],
                    r["mappo_train_time_by_mode"][mode],
                    r["mappo_env_interactions_by_mode"][mode],
                )
            if args.run_ippo:
                for mode in ippo_modes:
                    agg = r["ippo_agg_by_mode"][mode]
                    table.add_data(
                        kind, f"IPPO[{mode}]",
                        agg["terminal_ot_cost_mean"], agg["terminal_ot_cost_std"],
                        agg["reach_rate_mean"], agg["reach_rate_std"],
                        agg["target_coverage_mean"], agg["target_coverage_std"],
                        agg["mean_time_to_reach_mean"], agg["mean_time_to_reach_std"],
                        agg["total_cost_mean"], agg["total_cost_std"],
                        r["ippo_train_time_by_mode"][mode],
                        r["ippo_env_interactions_by_mode"][mode],
                    )
            if args.run_happo:
                for mode in happo_modes:
                    agg = r["happo_agg_by_mode"][mode]
                    table.add_data(
                        kind, f"HAPPO[{mode}]",
                        agg["terminal_ot_cost_mean"], agg["terminal_ot_cost_std"],
                        agg["reach_rate_mean"], agg["reach_rate_std"],
                        agg["target_coverage_mean"], agg["target_coverage_std"],
                        agg["mean_time_to_reach_mean"], agg["mean_time_to_reach_std"],
                        agg["total_cost_mean"], agg["total_cost_std"],
                        r["happo_train_time_by_mode"][mode],
                        r["happo_env_interactions_by_mode"][mode],
                    )
            if args.run_qmix:
                for mode in qmix_modes:
                    agg = r["qmix_agg_by_mode"][mode]
                    table.add_data(
                        kind, f"QMIX[{mode}]",
                        agg["terminal_ot_cost_mean"], agg["terminal_ot_cost_std"],
                        agg["reach_rate_mean"], agg["reach_rate_std"],
                        agg["target_coverage_mean"], agg["target_coverage_std"],
                        agg["mean_time_to_reach_mean"], agg["mean_time_to_reach_std"],
                        agg["total_cost_mean"], agg["total_cost_std"],
                        r["qmix_train_time_by_mode"][mode],
                        r["qmix_env_interactions_by_mode"][mode],
                    )
            if args.run_vdn:
                for mode in vdn_modes:
                    agg = r["vdn_agg_by_mode"][mode]
                    table.add_data(
                        kind, f"VDN[{mode}]",
                        agg["terminal_ot_cost_mean"], agg["terminal_ot_cost_std"],
                        agg["reach_rate_mean"], agg["reach_rate_std"],
                        agg["target_coverage_mean"], agg["target_coverage_std"],
                        agg["mean_time_to_reach_mean"], agg["mean_time_to_reach_std"],
                        agg["total_cost_mean"], agg["total_cost_std"],
                        r["vdn_train_time_by_mode"][mode],
                        r["vdn_env_interactions_by_mode"][mode],
                    )
            if args.run_mfq:
                for mode in mfq_modes:
                    agg = r["mfq_agg_by_mode"][mode]
                    table.add_data(
                        kind, f"MFQ[{mode}]",
                        agg["terminal_ot_cost_mean"], agg["terminal_ot_cost_std"],
                        agg["reach_rate_mean"], agg["reach_rate_std"],
                        agg["target_coverage_mean"], agg["target_coverage_std"],
                        agg["mean_time_to_reach_mean"], agg["mean_time_to_reach_std"],
                        agg["total_cost_mean"], agg["total_cost_std"],
                        r["mfq_train_time_by_mode"][mode],
                        r["mfq_env_interactions_by_mode"][mode],
                    )
            if args.run_mfq_local:
                for mode in mfq_local_modes:
                    agg = r["mfq_local_agg_by_mode"][mode]
                    table.add_data(
                        kind, f"MFQ-local[{mode}]",
                        agg["terminal_ot_cost_mean"], agg["terminal_ot_cost_std"],
                        agg["reach_rate_mean"], agg["reach_rate_std"],
                        agg["target_coverage_mean"], agg["target_coverage_std"],
                        agg["mean_time_to_reach_mean"], agg["mean_time_to_reach_std"],
                        agg["total_cost_mean"], agg["total_cost_std"],
                        r["mfq_local_train_time_by_mode"][mode],
                        r["mfq_local_env_interactions_by_mode"][mode],
                    )
        wb_run.log({"comparison_table": table})

        artifact = wandb.Artifact(f"comparison_p{args.p}_{stamp}", type="results")
        artifact.add_file(os.path.join(base_outdir, "comparison.json"))
        for kind in selected_noise_kinds:
            kind_dir = os.path.join(base_outdir, kind)
            if not os.path.isdir(kind_dir):
                continue
            for fname in os.listdir(kind_dir):
                if fname.endswith((".json", ".png")):
                    artifact.add_file(os.path.join(kind_dir, fname),
                                      name=f"{kind}/{fname}")
        wb_run.log_artifact(artifact)

        wb_run.finish()

    print(f"\nAll outputs saved to: {base_outdir}")


if __name__ == "__main__":
    main()
