"""
Experimental JAX-native tabular Q-learning.

This module keeps the existing Python/dict-based implementation untouched and
provides a *separate* dense JAX prototype for single-agent training. The goal
is to make it easy to later extend to batched multi-agent training with a
shared Q-table.

Current status:
  - Single-agent tabular Q-learning in pure JAX (functional style).
  - Dense Q-table with indices: [num_nodes, num_nodes, max_time_slices, max_deg].
  - Same reward / dynamics as the existing discrete-time Q-learning:
    uses training.q_learning._jitted_step_with_discretization.
"""

from dataclasses import dataclass
from typing import Tuple, Dict

import jax
import jax.numpy as jnp
from jax import random as jax_random
import numpy as np

from taxi_env import TaxiState, TaxiEnv, init_env
from training.q_learning import _jitted_step_with_discretization


def _discretize_time(time: float, dt: float = 1.0) -> int:
    """Discretize time to the nearest multiple of dt."""
    return int(round(time / dt))


def _state_index(
    env: TaxiEnv,
    state: TaxiState,
    dt: float,
    max_time_slices: int,
) -> Tuple[int, int, int]:
    """
    Map TaxiState to dense Q-table indices: (curr_node, pickup_node, time_idx).

    Time index is clamped to [0, max_time_slices-1] to keep the table bounded.
    """
    curr = int(state.current_node)
    pickup = int(state.pickup_node)
    t_idx = _discretize_time(float(state.time), dt)
    t_idx = max(0, min(max_time_slices - 1, t_idx))
    # Safety clamps on nodes as well (in case of bad states)
    curr = max(0, min(env.num_nodes - 1, curr))
    pickup = max(0, min(env.num_nodes - 1, pickup))
    return curr, pickup, t_idx


@dataclass
class QConfig:
    """Configuration for JAX tabular Q-learning."""

    dt: float = 1.0
    gamma: float = 1.0
    learning_rate: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 10000
    max_time_slices: int = 100  # maximum discrete time index for the table


def make_q_table(
    env: TaxiEnv,
    cfg: QConfig,
    initial_q_value: float = 10.0,
) -> jnp.ndarray:
    """
    Create a dense Q-table with shape [num_nodes, num_nodes, max_time_slices, max_deg].

    This is an experimental dense representation. It roughly matches what the
    existing shortest-path initialization populates in the sparse dict-based table.
    """
    shape = (env.num_nodes, env.num_nodes, cfg.max_time_slices, env.max_deg)
    q_init = jnp.full(shape, initial_q_value, dtype=jnp.float32)
    return q_init


def epsilon_at_step(cfg: QConfig, step: int) -> float:
    """Linear epsilon decay from epsilon_start to epsilon_end over epsilon_decay_steps."""
    if step >= cfg.epsilon_decay_steps:
        return float(cfg.epsilon_end)
    progress = step / float(cfg.epsilon_decay_steps)
    return float(cfg.epsilon_start * (1.0 - progress) + cfg.epsilon_end * progress)


def q_greedy_action(
    q: jnp.ndarray,
    env: TaxiEnv,
    state: TaxiState,
    cfg: QConfig,
) -> int:
    """Greedy action with respect to the dense Q-table for a single state."""
    curr, pickup, t_idx = _state_index(env, state, cfg.dt, cfg.max_time_slices)
    # Q-values for this state over actions
    q_row = q[curr, pickup, t_idx, :]  # [max_deg]
    # Mask invalid actions
    valid_mask = np.array(state.neighbor_mask, dtype=bool)
    if not valid_mask.any():
        return 0
    q_np = np.array(q_row, dtype=float)
    q_np[~valid_mask] = -1e9
    best_idx = int(np.argmax(q_np))
    return best_idx


def q_epsilon_greedy_action(
    q: jnp.ndarray,
    env: TaxiEnv,
    state: TaxiState,
    cfg: QConfig,
    step: int,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray]:
    """Epsilon-greedy action selection using the dense Q-table."""
    eps = epsilon_at_step(cfg, step)
    key, subkey = jax_random.split(key)
    if float(jax_random.uniform(subkey)) < eps:
        # Random valid action
        valid_actions = [i for i, v in enumerate(np.array(state.neighbor_mask)) if v]
        if not valid_actions:
            return 0, key
        key, subkey2 = jax_random.split(key)
        idx = int(jax_random.randint(subkey2, (), 0, len(valid_actions)))
        return int(valid_actions[idx]), key
    else:
        return q_greedy_action(q, env, state, cfg), key


def q_update_single(
    q: jnp.ndarray,
    env: TaxiEnv,
    state: TaxiState,
    action: int,
    reward: float,
    next_state: TaxiState,
    done: bool,
    cfg: QConfig,
) -> jnp.ndarray:
    """
    Single-state Q-learning update for the dense Q-table.

    Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)].
    """
    curr, pickup, t_idx = _state_index(env, state, cfg.dt, cfg.max_time_slices)
    # Current Q(s,a)
    q_sa = q[curr, pickup, t_idx, action]

    if done:
        target = reward
    else:
        # Next state index and max_a' Q(s',a')
        n_curr, n_pickup, n_t = _state_index(env, next_state, cfg.dt, cfg.max_time_slices)
        q_next_row = q[n_curr, n_pickup, n_t, :]
        valid_mask = np.array(next_state.neighbor_mask, dtype=bool)
        if valid_mask.any():
            q_next_np = np.array(q_next_row, dtype=float)
            q_next_np[~valid_mask] = -1e9
            max_next = float(np.max(q_next_np))
        else:
            max_next = 0.0
        target = reward + cfg.gamma * max_next

    new_q_sa = q_sa + cfg.learning_rate * (target - q_sa)
    q = q.at[curr, pickup, t_idx, action].set(new_q_sa)
    return q


def run_episode_jax(
    q: jnp.ndarray,
    env: TaxiEnv,
    cfg: QConfig,
    fixed_starts: jnp.ndarray,
    fixed_pickups: jnp.ndarray,
    max_steps: int,
    episode_index: int,
    key: jnp.ndarray,
) -> Tuple[jnp.ndarray, Dict, jnp.ndarray]:
    """
    Run a single Q-learning episode with the dense JAX Q-table.

    This mirrors the structure of the existing train_q_learning loop but:
      - Uses Q as a dense array.
      - Returns a new Q (no in-place Python mutation).
    """
    # Sample random start and pickup
    key, key1, key2, init_key = jax_random.split(key, 4)
    start = int(jax_random.choice(key1, fixed_starts))
    pickup = int(jax_random.choice(key2, fixed_pickups))

    state, _ = init_env(init_key, start, pickup, env.neighbor_mask_static)

    episode_reward = 0.0
    episode_length = 0
    episode_done = False

    # We still track steps globally via (episode_index, step)
    for step in range(max_steps):
        if state.done:
            episode_done = True
            break

        training_step = episode_index * max_steps + step
        key, action_key = jax_random.split(key)
        action, key = q_epsilon_greedy_action(q, env, state, cfg, training_step, key)

        # Discrete step using existing jitted helper
        next_state, reward, done_flag, info = _jitted_step_with_discretization(
            env,
            state,
            int(action),
            env.travel_times,  # discrete_travel_times is not exposed; we reuse env.travel_times here
            cfg.dt,
        )

        reward_f = float(reward)
        done_bool = bool(done_flag)

        q = q_update_single(
            q,
            env,
            state,
            int(action),
            reward_f,
            next_state,
            done_bool,
            cfg,
        )

        episode_reward += reward_f
        episode_length += 1
        state = next_state

        if done_bool:
            episode_done = True
            break

    metrics = {
        "episode_reward": float(episode_reward),
        "episode_length": int(episode_length),
        "episode_done": bool(episode_done),
    }
    return q, metrics, key


def train_q_learning_jax(
    env: TaxiEnv,
    fixed_starts: jnp.ndarray,
    fixed_pickups: jnp.ndarray,
    num_episodes: int,
    max_steps_per_episode: int,
    cfg: QConfig,
    initial_q_value: float = 10.0,
    seed: int = 0,
) -> Tuple[jnp.ndarray, Dict[str, np.ndarray]]:
    """
    Simple single-agent JAX-native Q-learning trainer.

    This does not modify the existing training pipeline. It is meant as a
    prototype to compare behavior and performance before moving to a fully
    batched multi-agent implementation.
    """
    q = make_q_table(env, cfg, initial_q_value=initial_q_value)
    key = jax_random.PRNGKey(seed)

    rewards = []
    lengths = []
    completions = []

    for ep in range(num_episodes):
        q, metrics, key = run_episode_jax(
            q,
            env,
            cfg,
            fixed_starts,
            fixed_pickups,
            max_steps_per_episode,
            ep,
            key,
        )
        rewards.append(metrics["episode_reward"])
        lengths.append(metrics["episode_length"])
        completions.append(1.0 if metrics["episode_done"] else 0.0)

    history = {
        "rewards": np.array(rewards, dtype=float),
        "lengths": np.array(lengths, dtype=float),
        "completions": np.array(completions, dtype=float),
    }
    return q, history


