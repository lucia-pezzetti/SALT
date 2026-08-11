"""
Heterogeneous-Agent Proximal Policy Optimization (HAPPO) — Kuba et al., 2022,
"Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning" (ICLR).

HAPPO is a CTDE actor-critic that, unlike MAPPO, does **not** share actor
parameters and updates agents **sequentially**: within each iteration a random
agent order i_1, ..., i_N is drawn, and each agent i_m maximizes a clipped
surrogate whose advantage is scaled by the cumulative product of the
already-updated agents' policy ratios,

    M^{1:m}(s, a) = prod_{j<m}  pi_new^{i_j}(a^{i_j}|s) / pi_old^{i_j}(a^{i_j}|s),

which yields the multi-agent advantage decomposition and a monotonic-improvement
guarantee. We keep a **centralized** value function over the global state (as in
MAPPO) and estimate the joint advantage with GAE on the shared team reward.

For a clean, apples-to-apples comparison the environment dynamics, observation
encoding, reward shaping and GAE are reused verbatim from :mod:`salt.mappo`; the
*only* algorithmic difference from MAPPO is the per-agent (non-shared) actors and
the sequential cumulative-ratio update above. This isolates HAPPO's contribution.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List, Optional
import numpy as np

import torch
import torch.nn as nn

from .env import GridConfig, ACTIONS, Pos, target_col_lo
from .noise import NoiseConfig
from .matching import terminal_ot_cost, target_coverage_rate
# Reuse MAPPO's networks and env/rollout helpers so HAPPO differs from MAPPO
# *only* in its optimizer (non-shared actors + sequential update).
from .mappo import Actor, Critic, _gae_batched, _vec_slip, _linear_decay

_ACTIONS_ARR = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)], dtype=np.int32)
HAPPO_OBS_MODES = {"relative_targets"}

# The small tensor workloads here are faster with bounded intra-op parallelism.
torch.set_num_threads(min(torch.get_num_threads(), 8))


@dataclass
class HAPPOConfig:
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
    minibatch_size_critic: int = 1024


class HAPPOAgents(nn.Module):
    """Container of N non-shared actors (HAPPO is a heterogeneous-agent method).

    Mirrors the single-actor interface expected by the comparison runner: it is
    an ``nn.Module`` (so ``state_dict`` / ``load_state_dict`` work) and carries
    the ``obs_mode`` / ``n_targets`` / ``n_agents`` attributes that the runner and
    rollout read back.
    """

    def __init__(self, n_agents: int, obs_dim: int, n_actions: int = 5, hidden: int = 64):
        super().__init__()
        self.actors = nn.ModuleList(
            [Actor(obs_dim, n_actions, hidden) for _ in range(n_agents)]
        )
        self.n_agents = n_agents
        self.obs_mode = "relative_targets"
        self.n_targets = n_agents


def _validate_obs_mode(obs_mode: str) -> str:
    if obs_mode not in HAPPO_OBS_MODES:
        raise ValueError(
            f"Invalid HAPPO obs mode '{obs_mode}'. "
            f"Expected one of {sorted(HAPPO_OBS_MODES)}."
        )
    return obs_mode


def train_happo(
    grid: GridConfig,
    noise_cfg: NoiseConfig,
    n_agents: int,
    cfg: HAPPOConfig | None = None,
    seed: int = 0,
    log_every: int = 10,
    print_every: int = 1000,
    n_targets: int | None = None,
    wb_run=None,
    wb_prefix: str = "train_happo",
) -> Tuple[HAPPOAgents, Dict[str, Any]]:
    if cfg is None:
        cfg = HAPPOConfig()
    obs_mode = _validate_obs_mode(cfg.obs_mode)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n_act = len(ACTIONS)
    H, N, E = grid.horizon, n_agents, cfg.batch_episodes
    M = n_targets if n_targets is not None else N
    obs_dim = 5 + 3 * M
    state_dim = 3 * N + 1 + 3 * M

    agents = HAPPOAgents(N, obs_dim, n_act, cfg.hidden_size)
    critic = Critic(state_dim, cfg.hidden_size)
    opt_a = [torch.optim.Adam(agents.actors[i].parameters(), lr=cfg.lr_actor)
             for i in range(N)]
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    aid = np.arange(N, dtype=np.float32) / max(1, N - 1)
    kind = noise_cfg.kind
    p_noise = float(noise_cfg.p)
    batch_rewards_log: List[float] = []
    train_curve: List[Dict[str, float]] = []
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
        col_lo = target_col_lo(grid)
        tgt_c = rng.integers(col_lo, grid.w, size=(E, M)).astype(np.int32)

        reached = np.zeros((E, N), dtype=bool)
        tgt_active = np.ones((E, M), dtype=bool)
        s_obs_l = np.empty((H, E, N, obs_dim), dtype=np.float32)
        s_acts = np.empty((H, E, N), dtype=np.int64)
        s_logp = np.empty((H, E, N), dtype=np.float32)
        s_mask = np.empty((H, E, N), dtype=np.float32)
        s_obs_g = np.empty((H, E, state_dim), dtype=np.float32)
        s_rew = np.empty((H, E), dtype=np.float32)
        s_val = np.empty((H, E), dtype=np.float32)
        s_not_done = np.empty((H, E), dtype=np.float32)
        s_state_mask = np.empty((H, E), dtype=np.float32)

        obs_l = np.empty((E, N, obs_dim), dtype=np.float32)
        obs_g = np.empty((E, state_dim), dtype=np.float32)
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

            obs_g[:, 0:2 * N:2] = pos_r / max(1, grid.h - 1)
            obs_g[:, 1:2 * N:2] = pos_c / max(1, grid.w - 1)
            obs_g[:, 2 * N:3 * N] = active_f
            obs_g[:, 3 * N] = t / max(1, H - 1)
            tbase = 3 * N + 1
            obs_g[:, tbase:tbase + 2 * M:2] = tgt_r / max(1, grid.h - 1)
            obs_g[:, tbase + 1:tbase + 2 * M:2] = tgt_c / max(1, grid.w - 1)
            obs_g[:, tbase + 2 * M:] = tgt_active.astype(np.float32)

            mask = (~reached).astype(np.float32)
            s_state_mask[t] = (mask.sum(axis=1) > 0).astype(np.float32)

            # Per-agent action sampling under each agent's own (non-shared) actor.
            actions = np.empty((E, N), dtype=np.int64)
            logps = np.empty((E, N), dtype=np.float32)
            with torch.no_grad():
                for i in range(N):
                    a_i, lp_i = agents.actors[i].get_action(
                        torch.from_numpy(obs_l[:, i, :])
                    )
                    actions[:, i] = a_i.numpy()
                    logps[:, i] = lp_i.numpy()
                values = critic(torch.from_numpy(obs_g)).numpy()

            actions[reached] = 4

            exec_a = actions.copy().astype(np.int32)
            active = ~reached
            if kind != "none" and p_noise > 0:
                if kind == "individual":
                    flat = exec_a.ravel()
                    act_flat = active.ravel()
                    idx = np.where(act_flat)[0]
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
                            key = (e, int(pos_r[e, i]), int(pos_c[e, i]),
                                   int(actions[e, i]))
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
            new_r[active] = np.clip(
                pos_r[active] + dr[active], 0, grid.h - 1).astype(np.int32)
            new_c[active] = np.clip(
                pos_c[active] + dc[active], 0, grid.w - 1).astype(np.int32)

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

            rew = (-float(grid.step_cost) * n_still
                   + float(grid.goal_bonus) * n_just).astype(np.float32)

            if t == H - 1:
                for e in range(E):
                    fp = [(int(new_r[e, i]), int(new_c[e, i])) for i in range(N)]
                    tl = [(int(tgt_r[e, j]), int(tgt_c[e, j])) for j in range(M)]
                    rew[e] -= terminal_ot_cost(fp, tl)
            all_reached = reached.all(axis=1)
            s_not_done[t] = (((t < (H - 1)) & (~all_reached))).astype(np.float32)

            s_obs_l[t] = obs_l
            s_acts[t] = actions
            s_logp[t] = logps
            s_mask[t] = mask
            s_obs_g[t] = obs_g
            s_rew[t] = rew
            s_val[t] = values

            pos_r, pos_c = new_r, new_c

        # Joint advantage from the centralized critic (shared across agents).
        adv, ret = _gae_batched(s_rew, s_val, s_not_done, cfg.gamma, cfg.gae_lambda)

        HE = H * E
        b_obs = torch.from_numpy(s_obs_l.reshape(HE, N, obs_dim))
        b_acts = torch.from_numpy(s_acts.reshape(HE, N))
        b_logp = torch.from_numpy(s_logp.reshape(HE, N))
        b_mask = torch.from_numpy(s_mask.reshape(HE, N))
        b_adv = torch.from_numpy(adv.reshape(HE))
        c_obs = torch.from_numpy(s_obs_g.reshape(HE, state_dim))
        c_ret = torch.from_numpy(ret.reshape(HE))
        c_mask = torch.from_numpy(s_state_mask.reshape(HE))

        # Standardize the joint advantage over active states (as in MAPPO).
        sm = c_mask > 0.5
        if sm.sum() > 1:
            b_adv = (b_adv - b_adv[sm].mean()) / (b_adv[sm].std() + 1e-8)

        actor_losses: List[float] = []
        critic_losses: List[float] = []
        entropies: List[float] = []
        clip_fracs: List[float] = []
        mb_a = max(1, min(cfg.minibatch_size_actor, HE))
        mb_c = max(1, min(cfg.minibatch_size_critic, HE))

        # --- HAPPO sequential update over one random agent order ---
        order = rng.permutation(N)
        m_weight = torch.ones(HE, dtype=torch.float32)  # cumulative M^{1:m}
        for i in order:
            obs_i = b_obs[:, i, :]
            acts_i = b_acts[:, i]
            oldlp_i = b_logp[:, i]
            mask_i = b_mask[:, i]
            adv_i = (b_adv * m_weight).detach()  # heterogeneous-agent advantage weight

            for _ in range(cfg.ppo_epochs):
                perm = rng.permutation(HE)
                for start in range(0, HE, mb_a):
                    idx = torch.from_numpy(perm[start:start + mb_a])
                    mb_mask = mask_i[idx]
                    ms = mb_mask.sum()
                    if ms.item() <= 0:
                        continue
                    nlp, ent = agents.actors[i].evaluate(obs_i[idx], acts_i[idx])
                    ratio = torch.exp(nlp - oldlp_i[idx])
                    a_mb = adv_i[idx]
                    s1 = ratio * a_mb
                    s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * a_mb
                    clipped = ((ratio < (1 - cfg.clip_eps))
                               | (ratio > (1 + cfg.clip_eps))).float()
                    clip_frac = (clipped * mb_mask).sum() / ms
                    a_loss = (-(torch.min(s1, s2) * mb_mask).sum() / ms
                              - entropy_coef_t * (ent * mb_mask).sum() / ms)
                    opt_a[i].zero_grad()
                    a_loss.backward()
                    nn.utils.clip_grad_norm_(agents.actors[i].parameters(),
                                             cfg.max_grad_norm)
                    opt_a[i].step()
                    actor_losses.append(float(a_loss.item()))
                    entropies.append(float(((ent * mb_mask).sum() / ms).item()))
                    clip_fracs.append(float(clip_frac.item()))

            # Fold agent i's final updated policy ratio into the cumulative weight
            # before the next agent's update.
            with torch.no_grad():
                nlp_all, _ = agents.actors[i].evaluate(obs_i, acts_i)
                r_all = torch.exp(nlp_all - oldlp_i)
                # Inactive agents contribute a neutral factor of 1.
                r_all = torch.where(mask_i > 0.5, r_all, torch.ones_like(r_all))
                m_weight = m_weight * r_all

        # --- Centralized critic update (shared) ---
        for _ in range(cfg.ppo_epochs):
            perm_c = rng.permutation(HE)
            for start in range(0, HE, mb_c):
                idx = torch.from_numpy(perm_c[start:start + mb_c])
                mb_c_mask = c_mask[idx]
                ms_c = mb_c_mask.sum()
                if ms_c.item() <= 0:
                    continue
                v = critic(c_obs[idx])
                diff2 = (v - c_ret[idx]).pow(2)
                c_loss = (diff2 * mb_c_mask).sum() / ms_c
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
        interactions_done = int((batch_idx + 1) * E * N * H)
        train_curve.append({
            "batch": float(batch_idx + 1),
            "episodes": float(episodes_done),
            "env_interactions": float(interactions_done),
            "reach_rate": float(reach_rate),
        })
        if episodes_done % print_every == 0:
            rec = np.mean(batch_rewards_log[-log_every:])
            print(f"[HAPPO-{noise_cfg.kind}|{obs_mode}] batch {batch_idx+1:4d}/{cfg.n_batches}"
                  f"  episodes={episodes_done:6d}"
                  f"  mean_reward={rec:.2f}"
                  f"  reach_rate={reach_rate:.2%}")
        if (batch_idx + 1) % log_every == 0 and wb_run is not None:
            rec = np.mean(batch_rewards_log[-log_every:])
            p = wb_prefix
            wb_run.log({
                f"{p}/iter": interactions_done,
                f"{p}/batch": batch_idx + 1,
                f"{p}/episodes": episodes_done,
                f"{p}/env_interactions": interactions_done,
                f"{p}/mean_reward": float(rec),
                f"{p}/last_batch_reward": float(mean_rew),
                f"{p}/reach_rate": reach_rate,
                f"{p}/actor_loss": actor_loss_v,
                f"{p}/critic_loss": critic_loss_v,
                f"{p}/entropy": entropy_v,
                f"{p}/clip_fraction": clip_frac_v,
            })

    agents.obs_mode = obs_mode
    agents.n_targets = M
    executed_batches = len(train_curve)
    executed_episodes = int(executed_batches * E)
    executed_interactions = int(executed_batches * E * N * H)
    return agents, {
        "algorithm": "HAPPO", "episodes": executed_episodes,
        "n_agents": N, "n_targets": M, "noise_kind": noise_cfg.kind,
        "p": noise_cfg.p, "happo_config": asdict(cfg),
        "obs_mode": obs_mode,
        "env_interactions": executed_interactions,
        "mean_reward_last10": float(np.mean(batch_rewards_log[-10:])),
        "entropy_coef_start": float(cfg.entropy_coef),
        "entropy_coef_end": float(cfg.entropy_coef_end),
        "planned_batches": int(cfg.n_batches),
        "executed_batches": int(executed_batches),
        "train_curve": train_curve,
    }


def rollout_happo(
    agents: HAPPOAgents, grid: GridConfig, noise_cfg: NoiseConfig,
    n_agents: int, targets: List[Pos] | None = None, seed: int = 0,
    obs_mode: str | None = None,
    start_rows: List[int] | None = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    N, H = n_agents, grid.horizon
    n_act = len(ACTIONS)

    if obs_mode is None:
        obs_mode = getattr(agents, "obs_mode", "relative_targets")
    obs_mode = _validate_obs_mode(obs_mode)

    if N != agents.n_agents:
        raise ValueError(
            f"HAPPO uses non-shared per-agent actors trained with "
            f"{agents.n_agents} agents; rollout requested {N}. HAPPO, like "
            f"MAPPO, does not transfer across fleet sizes."
        )

    if targets is None:
        n_targets = int(getattr(agents, "n_targets", grid.h))
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
    arrival: List[Optional[int]] = [None] * N
    trajectories: List[List[Pos]] = [
        [(int(pos_r[i]), int(pos_c[i]))] for i in range(N)
    ]
    total_cost = 0.0
    aid = np.arange(N, dtype=np.float32) / max(1, N - 1)
    obs_dim = 5 + 3 * len(targets)
    p_noise = float(noise_cfg.p)
    kind = noise_cfg.kind
    tgt_r = np.array([int(r) for r, _ in targets], dtype=np.int32)
    tgt_c = np.array([int(c) for _, c in targets], dtype=np.int32)
    tgt_active = np.ones(len(targets), dtype=bool)

    for a in agents.actors:
        a.eval()
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

        actions_np = np.empty(N, dtype=np.int64)
        with torch.no_grad():
            for i in range(N):
                logits = agents.actors[i].net(torch.from_numpy(obs_l[i:i + 1]))
                actions_np[i] = int(logits.argmax(-1).item())

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
            new_r[idx_a] = np.clip(sr + _ACTIONS_ARR[exec_a[idx_a], 0],
                                   0, grid.h - 1).astype(np.int32)
            new_c[idx_a] = np.clip(sc + _ACTIONS_ARR[exec_a[idx_a], 1],
                                   0, grid.w - 1).astype(np.int32)

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
        "algorithm": "HAPPO", "noise_kind": noise_cfg.kind, "p": noise_cfg.p,
        "obs_mode": obs_mode,
        "agents": N, "horizon": H,
        "target_coverage": cov,
        "reach_rate": nr / N,
        "mean_time_to_reach": float(np.mean(rt)) if rt else float("nan"),
        "mean_time_to_reach_including_unreached": float(np.mean(rt_with_horizon)),
        "total_cost": float(total_cost), "terminal_ot_cost": float(tc),
        "final_positions": [list(p) for p in fp], "arrival_times": arrival,
        "targets": [list(t) for t in targets],
        "trajectories": [
            [list(p) for p in traj] for traj in trajectories
        ],
    }
