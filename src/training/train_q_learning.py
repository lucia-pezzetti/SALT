"""
Training function for tabular Q-learning when discrete=True.
"""
import jax
import jax.numpy as jnp
from jax import random as jax_random
from jax import vmap
import numpy as np
import wandb
from typing import Tuple, Optional, Dict
from pathlib import Path
import optax
import time
from collections import defaultdict

from taxi_env import TaxiState, TaxiEnv, init_env
from training.q_learning import (
    TabularQLearning,
    _jitted_step_with_discretization,
    _batched_select_action,
    _update_q_value_jit,
    _batched_update_q_values_vectorized,
)
from utils import offline_shortest_path_action


class PerformanceProfiler:
    """Profiler for tracking time spent in different parts of the code."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, list] = defaultdict(list)
        self.counts: Dict[str, int] = defaultdict(int)
        self.start_times: Dict[str, float] = {}
    
    def start(self, name: str):
        """Start timing an operation."""
        if self.enabled:
            self.start_times[name] = time.perf_counter()
    
    def end(self, name: str):
        """End timing an operation and record it."""
        if self.enabled and name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.timings[name].append(elapsed)
            self.counts[name] += 1
            del self.start_times[name]
    
    def time_block(self, name: str):
        """Context manager for timing a block of code."""
        return TimingContext(self, name)
    
    def get_summary(self) -> Dict[str, Dict]:
        """Get summary statistics for all timed operations."""
        summary = {}
        for name, times in self.timings.items():
            if times:
                summary[name] = {
                    'total': sum(times),
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': self.counts[name],
                }
        return summary
    
    def print_summary(self, num_episodes: int = None):
        """Print a formatted summary of timing statistics."""
        if not self.enabled:
            return
        
        summary = self.get_summary()
        if not summary:
            print("No profiling data collected.")
            return
        
        print("\n" + "="*80)
        print("PERFORMANCE PROFILING SUMMARY")
        print("="*80)
        
        # Sort by total time
        sorted_items = sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True)
        
        total_time = sum(s['total'] for s in summary.values())
        
        print(f"\n{'Operation':<40} {'Total (s)':<12} {'Mean (s)':<12} {'Count':<10} {'% of Total':<10}")
        print("-"*80)
        
        for name, stats in sorted_items:
            pct = (stats['total'] / total_time * 100) if total_time > 0 else 0
            print(f"{name:<40} {stats['total']:<12.4f} {stats['mean']:<12.6f} {stats['count']:<10} {pct:<10.2f}")
        
        print("-"*80)
        print(f"{'TOTAL':<40} {total_time:<12.4f}")
        
        if num_episodes:
            print(f"\nAverage time per episode: {total_time / num_episodes:.4f} seconds")
            print(f"Episodes per second: {num_episodes / total_time:.2f}")
        
        print("="*80 + "\n")


class TimingContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name
    
    def __enter__(self):
        self.profiler.start(self.name)
        return self
    
    def __exit__(self, *args):
        self.profiler.end(self.name)


def pretrain_q_learning_on_shortest_path(
    q_agent: TabularQLearning,
    env,
    fixed_starts,
    fixed_pickups,
    num_pretrain_episodes: int = 1000,
    max_steps_per_episode: int = 300,
    eval_starts=None,
    eval_pickups=None,
    seed: int = 42,
    log_fn=None,
    pretrain_learning_rate: Optional[float] = None,
):
    """
    Pretrain Q-learning agent using shortest path rollouts in discrete time.
    
    This function generates expert trajectories using shortest path actions
    and updates Q-values directly, providing a good initialization for Q-learning.
    
    Args:
        q_agent: Q-learning agent to pretrain
        env: TaxiEnv instance
        fixed_starts: Array of start node indices
        fixed_pickups: Array of pickup node indices
        num_pretrain_episodes: Number of pretraining episodes
        max_steps_per_episode: Maximum steps per episode
        eval_starts: Optional fixed starts for evaluation
        eval_pickups: Optional fixed pickups for evaluation
        seed: Random seed
        log_fn: Optional logging function(log_dict, step) for metrics
        
    Returns:
        Pretrained Q-learning agent (same object, modified in place)
    """
    key = jax_random.PRNGKey(seed)
    
    # Evaluation sets
    if eval_starts is None:
        eval_starts = fixed_starts[:min(5, len(fixed_starts))]
    if eval_pickups is None:
        eval_pickups = fixed_pickups[:min(5, len(fixed_pickups))]
    
    # Use separate learning rate for pretraining if provided, otherwise use agent's learning rate
    original_lr = q_agent.learning_rate
    if pretrain_learning_rate is not None:
        q_agent.learning_rate = pretrain_learning_rate
    
    print(f"\n{'='*60}")
    print(f"Starting Q-learning pretraining with shortest path rollouts")
    print(f"{'='*60}")
    print(f"Pretraining episodes: {num_pretrain_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"Discretization dt: {q_agent.dt}")
    print(f"Learning rate: {q_agent.learning_rate} {'(pretraining-specific)' if pretrain_learning_rate is not None else '(using agent LR)'}")
    print(f"Discount factor: {q_agent.gamma}")
    print()
    
    # Statistics
    pretrain_rewards = []
    pretrain_lengths = []
    pretrain_completions = []
    
    # Pretraining loop
    for episode in range(num_pretrain_episodes):
        # Sample random start and pickup
        key, key1, key2 = jax_random.split(key, 3)
        start = int(jax_random.choice(key1, fixed_starts))
        pickup = int(jax_random.choice(key2, fixed_pickups))
        
        # Initialize episode
        key, init_key = jax_random.split(key)
        state, _ = init_env(init_key, start, pickup, env.neighbor_mask_static)
        
        episode_reward = 0.0
        episode_length = 0
        episode_done = False
        
        # Run episode with shortest path actions
        for step in range(max_steps_per_episode):
            if state.done:
                episode_done = True
                break
            
            # Get expert shortest path action (use discrete travel times for consistency)
            expert_action = offline_shortest_path_action(
                state.current_node,
                state.pickup_node,
                env.adj_list,
                q_agent.discrete_travel_times,
                env.distances,
                state.neighbor_mask
            )
            expert_action = int(expert_action)
            
            # Take step with discretization using expert action
            key, action_key = jax_random.split(key)
            next_state, reward, done, info = q_agent.step_with_discretization(
                state, expert_action, action_key
            )
            
            # Update Q-value using expert action
            q_agent.update_q_value(state, expert_action, reward, next_state, done)
            
            # Update statistics
            episode_reward += float(reward)
            episode_length += 1
            
            # Move to next state
            state = next_state
            
            if done:
                episode_done = True
                break
        
        pretrain_rewards.append(episode_reward)
        pretrain_lengths.append(episode_length)
        pretrain_completions.append(1.0 if episode_done else 0.0)
        
        # Logging
        if (episode + 1) % 100 == 0 or episode == num_pretrain_episodes - 1:
            avg_reward = np.mean(pretrain_rewards[-100:])
            avg_length = np.mean(pretrain_lengths[-100:])
            completion_rate = float(np.mean(pretrain_completions[-100:]))  # Ensure float type
            stats = q_agent.get_statistics()
            
            log_dict = {
                'pretrain/episode': episode + 1,
                'pretrain/avg_reward': float(avg_reward),
                'pretrain/avg_length': float(avg_length),
                'pretrain/completion_rate': completion_rate,  # This should be ~0.98 (98%)
                'pretrain/q_table_size': int(stats['num_states']),
                'pretrain/avg_q_value': float(stats['avg_q_value']),
            }
            
            if log_fn is not None:
                log_fn(log_dict, episode + 1)
            
            print(f"Pretraining episode {episode + 1}/{num_pretrain_episodes}: "
                  f"Avg reward={avg_reward:.2f}, "
                  f"Avg length={avg_length:.1f}, "
                  f"Completion={completion_rate:.2%}, "
                  f"Q-table size={stats['num_states']}")
        
        # Evaluation
        if (episode + 1) % 500 == 0 or episode == num_pretrain_episodes - 1:
            # Use discrete evaluation for pretraining (same as training environment)
            eval_results = evaluate_q_agent(
                q_agent, env, eval_starts, eval_pickups, max_steps_per_episode, 
                seed=seed, use_continuous_eval=False
            )
            
            eval_log_dict = {
                'pretrain/eval_episode': episode + 1,
                'pretrain/eval_avg_reward': float(eval_results['avg_reward']),
                'pretrain/eval_avg_steps': float(eval_results['avg_steps']),
                'pretrain/eval_completion_rate': float(eval_results['completion_rate']),  # This is evaluation completion (may be 0%)
            }
            
            if log_fn is not None:
                log_fn(eval_log_dict, episode + 1)
            
            print(f"  Evaluation: Avg reward={eval_results['avg_reward']:.2f}, "
                  f"Avg steps={eval_results['avg_steps']:.1f}, "
                  f"Completion={eval_results['completion_rate']:.2%}")
    
    # Restore original learning rate
    q_agent.learning_rate = original_lr
    
    print(f"\n{'='*60}")
    print(f"Q-learning pretraining completed!")
    print(f"Final Q-table size: {q_agent.get_statistics()['num_states']}")
    print(f"Restored learning rate to: {q_agent.learning_rate}")
    print(f"{'='*60}\n")
    
    return q_agent


def train_q_learning(
    env,
    fixed_starts,
    fixed_pickups,
    num_episodes: int = 10000,
    num_agents: int = 1,
    max_steps_per_episode: int = 300,
    dt: float = 1.0,
    learning_rate: float = 0.1,
    discount_factor: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay_steps: int = 10000,
    eval_frequency: int = 100,
    eval_starts=None,
    eval_pickups=None,
    seed: int = 42,
    pretrain_enabled: bool = False,
    num_pretrain_episodes: int = 1000,
    pretrain_log_fn=None,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None,
    initial_q_value: float = 10.0,
    init_from_shortest_paths: bool = False,
    init_all_time_slices: bool = False,
    pretrain_learning_rate: Optional[float] = None,
    enable_profiling: bool = True,
    init_q_table_path: Optional[str] = None,
):
    """
    Train a tabular Q-learning agent.
    
    Args:
        env: TaxiEnv instance
        fixed_starts: Array of start node indices
        fixed_pickups: Array of pickup node indices
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        dt: Time discretization step (default 5.0)
        learning_rate: Q-learning learning rate
        discount_factor: Discount factor (gamma)
        epsilon_start: Initial epsilon for epsilon-greedy
        epsilon_end: Final epsilon for epsilon-greedy
        epsilon_decay_steps: Steps over which epsilon decays
        eval_frequency: Frequency of evaluation (in episodes)
        eval_starts: Optional fixed starts for evaluation
        eval_pickups: Optional fixed pickups for evaluation
        seed: Random seed
        pretrain_enabled: Whether to use shortest path pretraining
        num_pretrain_episodes: Number of pretraining episodes (if pretrain_enabled)
        pretrain_log_fn: Optional logging function for pretraining metrics
        initial_q_value: Initial Q-value for optimistic initialization (default 10.0)
        init_from_shortest_paths: Whether to initialize Q-table from shortest path travel times
        init_all_time_slices: If True, initialize for all time slices (up to 100). 
                              If False, only initialize for time=0.
        pretrain_learning_rate: Optional learning rate for pretraining (default: None = use agent's LR).
                               Recommended: 0.01-0.05 when using init_from_shortest_paths to avoid
                               overwriting good initialization values.
        
    Returns:
        Trained Q-learning agent
    """
    # Compute max_time_slices automatically from cycle_length / dt
    # Time wraps around at cycle_length, so we only need that many slices
    cycle_length = float(jnp.max(env.periods))  # Maximum period (should be cycle_length)
    computed_max_time_slices = int(cycle_length / dt)
    
    # Initialize Q-learning agent
    if load_path is not None and Path(load_path).exists():
        print(f"Loading Q-table from {load_path}")
        q_agent = TabularQLearning.load(env, load_path)
        # Use loaded agent's max_time_slices if available, otherwise use computed value
        max_time_slices = getattr(q_agent, 'max_time_slices', computed_max_time_slices)
        if max_time_slices != computed_max_time_slices:
            print(f"  Note: Using loaded max_time_slices={max_time_slices} (computed value would be {computed_max_time_slices})")
    else:
        max_time_slices = computed_max_time_slices
        q_agent = TabularQLearning(
            env=env,
            dt=dt,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            initial_q_value=initial_q_value,
            max_time_slices=max_time_slices,
        )
    
    # Print Q-table configuration
    print(f"\n{'='*60}")
    print(f"Q-table Configuration:")
    print(f"  Cycle length: {cycle_length:.1f} seconds")
    print(f"  Time discretization (dt): {dt:.1f} seconds")
    print(f"  Max time slices: {max_time_slices} (cycle_length / dt)")
    print(f"  Q-table size: {env.num_nodes} × {env.num_nodes} × {max_time_slices} × {env.max_deg}")
    print(f"{'='*60}\n")
    
    # Initialize Q-table from saved file if specified
    if init_q_table_path is not None and Path(init_q_table_path).exists():
        print(f"\n{'='*60}")
        print(f"Initializing Q-table from saved file: {init_q_table_path}")
        print(f"{'='*60}")
        stats_before = q_agent.get_statistics()
        print(f"Q-table BEFORE initialization from file:")
        print(f"  Num states: {stats_before['num_states']}")
        print(f"  Avg Q-value: {stats_before['avg_q_value']:.4f}")
        
        q_agent.load_q_table_from_file(init_q_table_path)
        
        stats_after = q_agent.get_statistics()
        print(f"Q-table AFTER initialization from file:")
        print(f"  Num states: {stats_after['num_states']}")
        print(f"  Avg Q-value: {stats_after['avg_q_value']:.4f}")
        print(f"{'='*60}\n")
    elif init_q_table_path is not None:
        print(f"Warning: Initialization Q-table path specified but file does not exist: {init_q_table_path}")
        print(f"Continuing with default initialization...\n")
    
    # Initialize profiler
    profiler = PerformanceProfiler(enabled=enable_profiling)
    
    # Convert to JAX arrays if needed
    fixed_starts = jnp.asarray(fixed_starts)
    fixed_pickups = jnp.asarray(fixed_pickups)
    
    # Initialize random key
    key = jax_random.PRNGKey(seed)
    
    # Evaluation sets
    if eval_starts is None:
        eval_starts = fixed_starts[:min(5, len(fixed_starts))]
    if eval_pickups is None:
        eval_pickups = fixed_pickups[:min(5, len(fixed_pickups))]
    
    # Initialize Q-table from shortest path travel times
    if init_from_shortest_paths:
        with profiler.time_block("q_table_initialization"):
            print(f"\n{'='*60}")
            print(f"Initializing Q-table from shortest path travel times")
            print(f"{'='*60}")
            stats_before = q_agent.get_statistics()
            print(f"Q-table BEFORE shortest path initialization:")
            print(f"  Num states: {stats_before['num_states']}")
            print(f"  Avg Q-value: {stats_before['avg_q_value']:.4f}")
            
            q_agent.initialize_q_values_from_shortest_paths(
                use_all_time_slices=init_all_time_slices,
                max_time_slices=max_time_slices,
            )
            
            stats_after = q_agent.get_statistics()
            print(f"Q-table AFTER shortest path initialization:")
            print(f"  Num states: {stats_after['num_states']}")
            print(f"  Avg Q-value: {stats_after['avg_q_value']:.4f}")
            print(f"  States added: {stats_after['num_states'] - stats_before['num_states']}")
            print(f"{'='*60}\n")
    
    # Pretraining with shortest path rollouts
    if pretrain_enabled:
        with profiler.time_block("pretraining"):
            # Log Q-table state before pretraining
            stats_before = q_agent.get_statistics()
            print(f"\n{'='*60}")
            print(f"Q-table BEFORE pretraining:")
            print(f"  Num states: {stats_before['num_states']}")
            print(f"  Avg Q-value: {stats_before['avg_q_value']:.4f}")
            print(f"  Min Q-value: {stats_before['min_q_value']:.4f}")
            print(f"  Max Q-value: {stats_before['max_q_value']:.4f}")
            print(f"{'='*60}\n")
            
            pretrain_q_learning_on_shortest_path(
                q_agent=q_agent,
                env=env,
                fixed_starts=fixed_starts,
                fixed_pickups=fixed_pickups,
                num_pretrain_episodes=num_pretrain_episodes,
                max_steps_per_episode=max_steps_per_episode,
                eval_starts=eval_starts,
                eval_pickups=eval_pickups,
                seed=seed,
                log_fn=pretrain_log_fn,
                pretrain_learning_rate=pretrain_learning_rate,
            )
            
            # Log Q-table state after pretraining
            stats_after = q_agent.get_statistics()
            print(f"\n{'='*60}")
            print(f"Q-table AFTER pretraining:")
            print(f"  Num states: {stats_after['num_states']}")
            print(f"  Avg Q-value: {stats_after['avg_q_value']:.4f}")
            print(f"  Min Q-value: {stats_after['min_q_value']:.4f}")
            print(f"  Max Q-value: {stats_after['max_q_value']:.4f}")
            print(f"  States added during pretraining: {stats_after['num_states'] - stats_before['num_states']}")
            print(f"{'='*60}\n")
            
            # Verify Q-table was updated
            if stats_after['num_states'] == stats_before['num_states'] and stats_before['num_states'] > 0:
                print("⚠️  WARNING: Q-table size did not change during pretraining!")
            elif stats_after['num_states'] > stats_before['num_states']:
                print(f"✅ Q-table successfully updated: {stats_before['num_states']} -> {stats_after['num_states']} states")
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    episode_completions = []
    
    if num_episodes == 0:
        print(f"\n{'='*60}")
        print(f"Evaluation-only mode: epochs=0")
        print(f"Q-table will be evaluated without training")
        print(f"{'='*60}\n")
    else:
        print(f"Starting Q-learning training with {num_episodes} episodes...")
        print(f"Discretization dt={dt}, learning_rate={learning_rate}, gamma={discount_factor}")
        print(f"Multi-agent training with {num_agents} agents per episode")
    
    # Debug print training start
    print(f"\n[TRAINING START] Episodes: {num_episodes} | Agents: {num_agents} | Max steps/ep: {max_steps_per_episode} | "
          f"LR: {learning_rate} | Gamma: {discount_factor} | Epsilon: {epsilon_start} -> {epsilon_end}")
    
    # Create batched initialization function
    batched_init = jax.vmap(
        lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static),
        in_axes=(0, 0, 0),
        out_axes=(0, 0)
    )
    
    # Create batched step function
    @jax.jit
    def batched_step(states, actions, keys, discrete_travel_times, dt, env):
        """Batched environment step."""
        # vmap over the step function - TaxiState will be automatically batched
        next_states, rewards, dones, infos = jax.vmap(
            lambda s, a, k: _jitted_step_with_discretization(env, s, a, discrete_travel_times, dt),
            in_axes=(0, 0, 0),
            out_axes=(0, 0, 0, 0)
        )(states, actions, keys)
        return next_states, rewards, dones, infos
    
    # Create JIT-compiled matching function
    @jax.jit
    def match_pickups_jit(
        starts: jnp.ndarray,
        pickups: jnp.ndarray,
        q_table: jnp.ndarray,
        do_random: jnp.bool_,
        key: jnp.ndarray,
        dt: float,
        max_time_slices: int,
        num_nodes: int,
        env: TaxiEnv,
    ) -> jnp.ndarray:
        """
        JIT-compiled epsilon-greedy matching.
        Uses cond to only compute optimal matching when needed.
        """
        def random_matching(pickups, key):
            return jax_random.permutation(key, pickups)
        
        def optimal_matching(starts, pickups, q_table, key, dt, max_time_slices, num_nodes, env):
            # Estimate returns directly from Q-table
            num_starts = starts.shape[0]
            num_pickups = pickups.shape[0]
            
            # Create all combinations
            starts_expanded = jnp.repeat(starts, num_pickups)
            pickups_expanded = jnp.tile(pickups, num_starts)
            
            # Estimate returns directly from Q-table (batched)
            from training.q_learning import _estimate_returns_batch_q_table_direct
            returns_flat = _estimate_returns_batch_q_table_direct(
                q_table,
                starts_expanded,
                pickups_expanded,
                num_nodes,
                env,
            )
            
            # Reshape to [num_starts, num_pickups]
            returns_matrix = returns_flat.reshape(num_starts, num_pickups)
            
            # Hungarian algorithm
            _, assignment = optax.assignment.hungarian_algorithm(-returns_matrix)
            return pickups[assignment]
        
        # Use cond to only compute optimal matching when needed
        matched_pickups = jax.lax.cond(
            do_random,
            lambda: random_matching(pickups, key),
            lambda: optimal_matching(starts, pickups, q_table, key, dt, max_time_slices, num_nodes, env),
        )
        return matched_pickups
    
    # Create JIT-compiled episode runner using scan (more efficient for fixed max_steps)
    def run_episode_batch(
        states,
        q_table,
        episode_step_base,
        epsilon_start,
        epsilon_end,
        epsilon_decay_steps,
        dt,
        max_time_slices,
        num_nodes,
        learning_rate,
        gamma,
        discrete_travel_times,
        env,
        max_steps,
        key,
    ):
        """
        Run a batched episode using scan - fully JIT-compiled.
        More efficient than while_loop when max_steps is fixed.
        Returns: (final_states, episode_rewards, episode_lengths, episode_dones, final_q_table, final_key)
        """
        num_agents = states.current_node.shape[0]
        
        # Initialize tracking arrays
        episode_rewards = jnp.zeros(num_agents, dtype=jnp.float32)
        episode_lengths = jnp.zeros(num_agents, dtype=jnp.int32)
        episode_dones = jnp.zeros(num_agents, dtype=jnp.bool_)
        
        def step_fn(carry, step_idx):
            (states, q_table, episode_rewards, episode_lengths, episode_dones, key) = carry
            
            # Mask for active (not done) agents
            # JAX/XLA will optimize away unnecessary computation for done agents
            active_mask = ~states.done
            
            # Select actions for all agents (batched)
            training_step = episode_step_base + step_idx
            key_new, action_key = jax_random.split(key)
            action_keys = jax_random.split(action_key, num_agents)
            
            # Use scalar training_step - JAX will broadcast efficiently
            training_steps = jnp.broadcast_to(
                jnp.int32(training_step), (num_agents,)
            )
            
            actions, _ = _batched_select_action(
                q_table,
                states,
                training_steps,
                epsilon_start,
                epsilon_end,
                epsilon_decay_steps,
                dt,
                max_time_slices,
                num_nodes,
                action_keys,
            )
            
            # Take step for all agents (batched)
            # Environment step handles done states efficiently (returns unchanged state via jnp.where)
            next_states, rewards, dones, infos = batched_step(
                states, actions, action_keys,
                discrete_travel_times, dt, env
            )
            
            # Update Q-values for all agents (vectorized - much faster than sequential)
            # Q-table update already masks done agents efficiently (keeps old values via jnp.where)
            q_table_new = _batched_update_q_values_vectorized(
                q_table,
                states.current_node,
                states.pickup_node,
                states.time,
                states.neighbor_mask,
                states.done,
                actions,
                rewards,
                next_states.current_node,
                next_states.pickup_node,
                next_states.time,
                next_states.neighbor_mask,
                next_states.done,
                dones,
                learning_rate,
                gamma,
                dt,
                max_time_slices,
                num_nodes,
            )
            
            # Update statistics (only for agents that are not done)
            episode_rewards_new = episode_rewards + jnp.where(active_mask, rewards, 0.0)
            episode_lengths_new = episode_lengths + jnp.where(active_mask, 1, 0)
            episode_dones_new = episode_dones | dones
            
            new_carry = (next_states, q_table_new, episode_rewards_new, episode_lengths_new, episode_dones_new, key_new)
            
            # Return carry and None (we don't need to collect intermediate outputs)
            return new_carry, None
        
        # Run episode with scan over step indices
        step_indices = jnp.arange(max_steps, dtype=jnp.int32)
        carry = (states, q_table, episode_rewards, episode_lengths, episode_dones, key)
        
        final_carry, _ = jax.lax.scan(step_fn, carry, step_indices)
        
        final_states, final_q_table, final_rewards, final_lengths, final_dones, final_key = final_carry
        
        return final_states, final_rewards, final_lengths, final_dones, final_q_table, final_key
    
    run_episode_batch = jax.jit(run_episode_batch, static_argnums=(13,))  # max_steps must be static for jnp.arange
    
    # JIT-compiled function to sample starts and pickups
    def sample_starts_pickups(key, fixed_starts, fixed_pickups, num_agents):
        """Sample starts and pickups for all agents in parallel."""
        key, key1, key2 = jax_random.split(key, 3)
        start_keys = jax_random.split(key1, num_agents)
        pickup_keys = jax_random.split(key2, num_agents)
        
        # Vectorized sampling
        def sample_one(key, choices):
            return jax_random.choice(key, choices)
        
        starts = jax.vmap(sample_one, in_axes=(0, None))(start_keys, fixed_starts)
        pickups = jax.vmap(sample_one, in_axes=(0, None))(pickup_keys, fixed_pickups)
        
        return starts, pickups, key
    
    sample_starts_pickups = jax.jit(sample_starts_pickups, static_argnums=(3,))  # num_agents is a static argument
    
    # Skip training loop if num_episodes=0 (evaluation-only mode)
    if num_episodes > 0:
        # Precompute total training steps once (optimization: avoid recomputing every episode)
        total_training_steps = num_episodes * max_steps_per_episode
        matching_epsilon_start = 1.0
        matching_epsilon_end = 0.1
        
        for episode in range(num_episodes):
            # Calculate training step at the start of episode
            training_step = episode * max_steps_per_episode
            # Debug print at start of episode (every 100000 steps to reduce verbosity)
            if training_step % 100000 == 0 or episode == 0:
                print(f"\n[EPISODE START] Episode {episode + 1}/{num_episodes} | Training step: {training_step}")
            
            # Sample num_agents starts and pickups (JIT-compiled)
            with profiler.time_block("sampling_starts_pickups"):
                starts, pickups, key = sample_starts_pickups(key, fixed_starts, fixed_pickups, num_agents)
                # Only block if we need values for debug prints
                if training_step % 100000 == 0 or episode == 0:
                    starts.block_until_ready()
                    pickups.block_until_ready()
            
            # Debug print sampled starts/pickups (every 100000 steps)
            if training_step % 100000 == 0 or episode == 0:
                # Avoid unnecessary conversions - only convert when needed for printing
                starts_list = starts.tolist() if hasattr(starts, 'tolist') else list(starts)
                pickups_list = pickups.tolist() if hasattr(pickups, 'tolist') else list(pickups)
                print(f"[SAMPLING] Starts: {starts_list} | Pickups: {pickups_list}")
            
            # Epsilon-greedy matching: random with prob epsilon, optimal with prob (1-epsilon)
            if num_agents > 1:
                with profiler.time_block("matching"):
                    # Linear decay over all training steps (convert to JAX array first for efficiency)
                    progress = jnp.minimum(jnp.float32(training_step) / jnp.float32(total_training_steps), 1.0)
                    matching_epsilon = matching_epsilon_start * (1.0 - progress) + matching_epsilon_end * progress
                    
                    # Decide: random matching (epsilon) or optimal matching (1-epsilon)
                    key, match_key = jax_random.split(key)
                    do_random_matching = jax_random.uniform(match_key) < matching_epsilon
                    
                    # JIT-compiled matching (only computes optimal matching when needed)
                    matched_pickups = match_pickups_jit(
                        starts,
                        pickups,
                        q_agent.q_table,
                        do_random_matching,
                        key,
                        q_agent.dt,
                        q_agent.max_time_slices,
                        q_agent.env.num_nodes,
                        q_agent.env,
                    )
                    # Only block if we need values for debug prints
                    if training_step % 100000 == 0 or episode == 0:
                        matched_pickups.block_until_ready()
                        do_random_matching.block_until_ready()
                    
                    # Debug print matching result (every 100000 steps)
                    if training_step % 100000 == 0 or episode == 0:
                        # Only convert to Python when actually printing (avoid unnecessary conversion)
                        matched_list = matched_pickups.tolist() if hasattr(matched_pickups, 'tolist') else list(matched_pickups)
                        do_random_val = bool(do_random_matching.item() if hasattr(do_random_matching, 'item') else do_random_matching)
                        print(f"[MATCHING] Epsilon: {matching_epsilon:.3f} | Random: {do_random_val} | Matched pickups: {matched_list}")
            else:
                matched_pickups = pickups
            
            # Initialize all agents in parallel
            with profiler.time_block("initialization"):
                key, init_key = jax_random.split(key)
                init_keys = jax_random.split(init_key, num_agents)
                states, _ = batched_init(init_keys, starts, matched_pickups)
            
            # Run episode for all agents in parallel (fully JIT-compiled with scan)
            with profiler.time_block("episode_execution"):
                episode_step_base = training_step
                # Debug print every 100000 steps
                if training_step % 100000 == 0 or episode == 0:
                    print(f"[EPISODE EXECUTION START] Episode {episode + 1} | Base step: {episode_step_base} | Max steps: {max_steps_per_episode}")
                
                states, episode_rewards_batch, episode_lengths_batch, episode_dones_batch, q_agent.q_table, key = run_episode_batch(
                    states,
                    q_agent.q_table,
                    episode_step_base,
                    q_agent.epsilon_start,
                    q_agent.epsilon_end,
                    q_agent.epsilon_decay_steps,
                    q_agent.dt,
                    q_agent.max_time_slices,
                    q_agent.env.num_nodes,
                    q_agent.learning_rate,
                    q_agent.gamma,
                    q_agent.discrete_travel_times,
                    q_agent.env,
                    max_steps_per_episode,
                    key,
                )
                # Only block when we need values (for debug prints or statistics aggregation)
                # Q-table only needs blocking for debug prints (it's updated in-place in JAX)
                if (episode + 1) % 100000 == 0 or episode == 0:
                    q_agent.q_table.block_until_ready()
                
                # Debug print episode completion (every 100000 steps)
                if training_step % 100000 == 0 or episode == 0:
                    episode_rewards_batch.block_until_ready()
                    episode_lengths_batch.block_until_ready()
                    episode_dones_batch.block_until_ready()
                    avg_reward = float(jnp.mean(episode_rewards_batch))
                    avg_length = float(jnp.mean(episode_lengths_batch))
                    completion = float(jnp.mean(episode_dones_batch))
                    print(f"[EPISODE EXECUTION END] Episode {episode + 1} | "
                          f"Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} | Completion: {completion:.2%}")
            
            # Aggregate statistics across agents (convert to Python only for logging)
            # Only block when we need the values (for logging every 1000 episodes)
            need_values = (episode + 1) % 1000 == 0 or (episode + 1) == num_episodes
            with profiler.time_block("statistics_aggregation"):
                if need_values:
                    # Block only when we need to log (converting to float will implicitly block, but explicit is clearer)
                    episode_rewards_batch.block_until_ready()
                    episode_lengths_batch.block_until_ready()
                    episode_dones_batch.block_until_ready()
                    avg_episode_reward = float(jnp.mean(episode_rewards_batch))
                    avg_episode_length = float(jnp.mean(episode_lengths_batch))
                    completion_rate = float(jnp.mean(episode_dones_batch))
                    episode_rewards.append(avg_episode_reward)
                    episode_lengths.append(avg_episode_length)
                    episode_completions.append(completion_rate)
                else:
                    # Don't compute or append when not needed (avoid blocking)
                    avg_episode_reward = 0.0  # Placeholder
                    avg_episode_length = 0.0
                    completion_rate = 0.0
            
            # Logging
            if (episode + 1) % 1000 == 0:
                with profiler.time_block("logging"):
                    avg_reward = np.mean(episode_rewards[-100:])
                    avg_length = np.mean(episode_lengths[-100:])
                    completion_rate = np.mean(episode_completions[-100:])
                    # Use the last training step from the episode
                    last_training_step = (episode + 1) * max_steps_per_episode
                    epsilon = q_agent.get_epsilon(last_training_step)
                    # Only compute expensive statistics every 10000 episodes (or at end)
                    compute_stats = (episode + 1) % 10000 == 0 or (episode + 1) == num_episodes
                    stats = q_agent.get_statistics() if compute_stats else {'num_states': 0, 'avg_q_value': 0.0, 'min_q_value': 0.0, 'max_q_value': 0.0}
                    
                    # print(f"Episode {episode + 1}/{num_episodes}: "
                    #       f"Avg reward={avg_reward:.2f}, "
                    #       f"Avg length={avg_length:.1f}, "
                    #       f"Completion={completion_rate:.2%}, "
                    #       f"Epsilon={epsilon:.3f}, "
                    #       f"Q-table size={stats['num_states']}")
                    
                    # Wandb logging
                    if wandb.run is not None:
                        wandb.log({
                            'training/episode': episode + 1,
                            'training/avg_reward': avg_reward,
                            'training/avg_length': avg_length,
                            'training/completion_rate': completion_rate,
                            'training/epsilon': epsilon,
                            'q_learning/q_table_size': stats['num_states'],
                            'q_learning/avg_q_value': stats['avg_q_value'],
                            'q_learning/max_q_value': stats['max_q_value'],
                            'q_learning/min_q_value': stats['min_q_value'],
                        })
                    
                    # Debug print for training metrics
                    # print(f"[TRAINING] Episode {episode + 1}/{num_episodes} | "
                    #       f"Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} | "
                    #       f"Completion: {completion_rate:.2%} | Epsilon: {epsilon:.3f} | "
                    #       f"Q-table size: {stats['num_states']} | Avg Q: {stats['avg_q_value']:.2f}")
            
            # Evaluation
            if (episode + 1) % eval_frequency == 0:
                with profiler.time_block("evaluation"):
                    if num_agents > 1 and eval_starts is not None and eval_pickups is not None:
                        # Multi-agent evaluation with matching
                        # Sample sets of starts and pickups, perform matching, then evaluate
                        eval_key, subkey = jax_random.split(key)
                        B = num_agents
                        num_eval_sets = 3  # Use 3 sets for periodic evaluation
                        
                        # Sample sets (optimized: use vmap instead of list comprehension)
                        eval_start_sets = []
                        eval_pickup_sets = []
                        # Vectorized sampling function
                        def sample_one_set(key, choices):
                            return jax_random.choice(key, choices)
                        sample_batch = jax.vmap(sample_one_set, in_axes=(0, None))
                        
                        for i in range(num_eval_sets):
                            key1, subkey = jax_random.split(subkey)
                            start_keys = jax_random.split(key1, B)
                            start_set = sample_batch(start_keys, env.fixed_starts)
                            eval_start_sets.append(start_set)
                            
                            key2, subkey = jax_random.split(subkey)
                            pickup_keys = jax_random.split(key2, B)
                            pickup_set = sample_batch(pickup_keys, env.fixed_pickups)
                            eval_pickup_sets.append(pickup_set)
                        
                        # Evaluate with matching for each set
                        all_rewards_cont = []
                        all_steps_cont = []
                        all_completions_cont = []
                        all_rewards_disc = []
                        all_steps_disc = []
                        all_completions_disc = []
                        
                        for start_set, pickup_set in zip(eval_start_sets, eval_pickup_sets):
                            # Perform Hungarian matching using Q-table
                            returns_matrix = q_agent.estimate_returns_for_matching(start_set, pickup_set)
                            _, assignment = optax.assignment.hungarian_algorithm(-returns_matrix)
                            matched_pickups = pickup_set[assignment]
                            
                            # Evaluate matched pairs (continuous)
                            for start, pickup in zip(start_set, matched_pickups):
                                episode_reward = 0.0
                                episode_steps = 0
                                episode_done = False
                                
                                # start and pickup are already integers from JAX arrays
                                state = init_env(jax_random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
                                
                                for step in range(max_steps_per_episode):
                                    if state.done:
                                        episode_done = True
                                        break
                                    
                                    # Discretize state time for Q-table lookup
                                    dt = q_agent.dt
                                    discretized_time = dt * round(state.time / dt)
                                    state_for_policy = TaxiState(
                                        current_node=state.current_node,
                                        pickup_node=state.pickup_node,
                                        done=state.done,
                                        step_count=state.step_count,
                                        neighbor_mask=state.neighbor_mask,
                                        time=discretized_time,
                                    )
                                    
                                    # Greedy action (optimized: use JAX instead of Python loops)
                                    curr, pickup, t_idx = q_agent._get_state_indices(state_for_policy)
                                    q_row = q_agent.q_table[curr, pickup, t_idx, :]  # [max_deg]
                                    valid_mask = jnp.array(state_for_policy.neighbor_mask, dtype=jnp.bool_)
                                    q_masked = jnp.where(valid_mask, q_row, -1e9)
                                    
                                    # Check if any valid actions exist (optimized: avoid double computation)
                                    has_valid = jnp.any(valid_mask)
                                    if not bool(has_valid.item() if hasattr(has_valid, 'item') else has_valid):
                                        break
                                    
                                    action = int(jnp.argmax(q_masked))
                                    
                                    # Continuous step
                                    next_state, reward, done, info = env.step(state, action)
                                    episode_reward += float(reward)
                                    episode_steps += 1
                                    state = next_state
                                    
                                    if done:
                                        episode_done = True
                                        break
                                
                                all_rewards_cont.append(episode_reward)
                                all_steps_cont.append(episode_steps)
                                all_completions_cont.append(1.0 if episode_done else 0.0)
                            
                            # Evaluate matched pairs (discrete)
                            for start, pickup in zip(start_set, matched_pickups):
                                episode_reward = 0.0
                                episode_steps = 0
                                episode_done = False
                                
                                # start and pickup are already integers from JAX arrays
                                state = init_env(jax_random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
                                
                                for step in range(max_steps_per_episode):
                                    if state.done:
                                        episode_done = True
                                        break
                                    
                                    # Greedy action (optimized: use JAX instead of Python loops)
                                    curr, pickup, t_idx = q_agent._get_state_indices(state)
                                    q_row = q_agent.q_table[curr, pickup, t_idx, :]  # [max_deg]
                                    valid_mask = jnp.array(state.neighbor_mask, dtype=jnp.bool_)
                                    q_masked = jnp.where(valid_mask, q_row, -1e9)
                                    
                                    # Check if any valid actions exist (optimized: avoid double computation)
                                    has_valid = jnp.any(valid_mask)
                                    if not bool(has_valid.item() if hasattr(has_valid, 'item') else has_valid):
                                        break
                                    
                                    action = int(jnp.argmax(q_masked))
                                    
                                    # Discrete step
                                    step_key = jax_random.PRNGKey(step)
                                    next_state, reward, done, info = q_agent.step_with_discretization(state, action, step_key)
                                    episode_reward += float(reward)
                                    episode_steps += 1
                                    state = next_state
                                    
                                    if done:
                                        episode_done = True
                                        break
                                
                                all_rewards_disc.append(episode_reward)
                                all_steps_disc.append(episode_steps)
                                all_completions_disc.append(1.0 if episode_done else 0.0)
                        
                        eval_results_continuous = {
                            'avg_reward': np.mean(all_rewards_cont),
                            'avg_steps': np.mean(all_steps_cont),
                            'completion_rate': np.mean(all_completions_cont),
                        }
                        eval_results_discrete = {
                            'avg_reward': np.mean(all_rewards_disc),
                            'avg_steps': np.mean(all_steps_disc),
                            'completion_rate': np.mean(all_completions_disc),
                        }
                    else:
                        # Single agent or no eval sets: use standard evaluation
                        # Continuous evaluation (default)
                        eval_results_continuous = evaluate_q_agent(
                            q_agent, env, eval_starts, eval_pickups, max_steps_per_episode,
                            use_continuous_eval=True
                        )
                        
                        # Discrete evaluation (same as training environment)
                        eval_results_discrete = evaluate_q_agent(
                            q_agent, env, eval_starts, eval_pickups, max_steps_per_episode,
                            use_continuous_eval=False
                        )
                    
                    # Debug print for evaluation metrics (every 100000 episodes)
                    if (episode + 1) % 100000 == 0:
                        print(f"[EVALUATION] Episode {episode + 1}/{num_episodes}")
                        print(f"  Continuous: Avg Reward: {eval_results_continuous['avg_reward']:.2f} | "
                              f"Avg Steps: {eval_results_continuous['avg_steps']:.1f} | "
                              f"Completion: {eval_results_continuous['completion_rate']:.2%}")
                        print(f"  Discrete:   Avg Reward: {eval_results_discrete['avg_reward']:.2f} | "
                              f"Avg Steps: {eval_results_discrete['avg_steps']:.1f} | "
                              f"Completion: {eval_results_discrete['completion_rate']:.2%}")
                    
                    # Use continuous evaluation results for logging (can be changed if needed)
                    eval_results = eval_results_continuous
                    
                    # Commented out wandb logging
                    # if wandb.run is not None:
                    #     wandb.log({
                    #         'evaluation/episode': episode + 1,
                    #         'evaluation/avg_reward': eval_results['avg_reward'],
                    #         'evaluation/avg_steps': eval_results['avg_steps'],
                    #         'evaluation/completion_rate': eval_results['completion_rate'],
                    #         'evaluation/discrete_avg_reward': eval_results_discrete['avg_reward'],
                    #         'evaluation/discrete_avg_steps': eval_results_discrete['avg_steps'],
                    #         'evaluation/discrete_completion_rate': eval_results_discrete['completion_rate'],
                    #     })
    else:
        # Evaluation-only mode: no training episodes
        print("Skipping training (epochs=0). Q-table will be evaluated as-is.\n")
    
    if num_episodes > 0:
        print("Q-learning training completed!")
    else:
        print("Evaluation-only mode completed!")
    if save_path is not None:
        print(f"Saving Q-table to {save_path}")
        q_agent.save(save_path)
    
    # Print performance profiling summary
    profiler.print_summary(num_episodes=num_episodes)
    
    return q_agent


def _discretize_state_time(state: TaxiState, dt: float) -> TaxiState:
    """
    Create a copy of state with discretized time for Q-table lookup.
    The discretized time is used to access the Q-table, but we keep
    the original continuous time in the actual state for environment steps.
    """
    discretized_time = dt * round(state.time / dt)
    return TaxiState(
        current_node=state.current_node,
        pickup_node=state.pickup_node,
        done=state.done,
        step_count=state.step_count,
        neighbor_mask=state.neighbor_mask,
        time=discretized_time,
    )


def evaluate_q_agent(
    q_agent: TabularQLearning,
    env,
    eval_starts,
    eval_pickups,
    max_steps: int = 300,
    seed: int = 42,
    use_continuous_eval: bool = True,
) -> dict:
    """
    Evaluate Q-learning agent on fixed start-pickup pairs.
    
    Two evaluation modes:
    1. Continuous evaluation (use_continuous_eval=True): 
       - Uses continuous environment steps for actual transitions and rewards
       - Discretizes state time for Q-table lookup (action selection)
       - This evaluates how well the policy trained in discrete space performs in continuous space
       
    2. Discrete evaluation (use_continuous_eval=False):
       - Uses discretized environment steps (q_agent.step_with_discretization)
       - Same environment as training
       - This evaluates performance in the same discrete environment used for training
    
    Args:
        q_agent: Trained Q-learning agent
        env: TaxiEnv instance
        eval_starts: Array of start node indices
        eval_pickups: Array of pickup node indices
        max_steps: Maximum steps per episode
        seed: Random seed
        use_continuous_eval: If True, use continuous environment; if False, use discrete environment
        
    Returns:
        Dictionary with evaluation metrics
    """
    key = jax_random.PRNGKey(seed)
    dt = q_agent.dt
    
    total_rewards = []
    total_steps = []
    completions = []
    
    # Convert to numpy arrays for easier handling
    eval_starts_arr = np.array(eval_starts)
    eval_pickups_arr = np.array(eval_pickups)
    
    # Check if arrays are paired (same length) or should create Cartesian product
    if len(eval_starts_arr) == len(eval_pickups_arr):
        # Arrays are paired: iterate over pairs
        for start, pickup in zip(eval_starts_arr, eval_pickups_arr):
            key, init_key = jax_random.split(key)
            state, _ = init_env(init_key, int(start), int(pickup), env.neighbor_mask_static)
            
            episode_reward = 0.0
            episode_steps = 0
            episode_done = False
            
            # Run episode with greedy policy (epsilon=0)
            for step in range(max_steps):
                if state.done:
                    episode_done = True
                    break
                
                # Greedy action (epsilon=0)
                key, action_key = jax_random.split(key)
                
                if use_continuous_eval:
                    # CONTINUOUS EVALUATION: Discretize state time for Q-table lookup
                    state_for_q_lookup = _discretize_state_time(state, dt)
                else:
                    # DISCRETE EVALUATION: Use state directly (already in discrete space)
                    state_for_q_lookup = state
                
                # Get Q-values for all valid actions
                valid_actions = []
                q_values = []
                for action in range(env.max_deg):
                    if state_for_q_lookup.neighbor_mask[action]:
                        valid_actions.append(action)
                        q_values.append(q_agent.get_q_value(state_for_q_lookup, action))
                
                if len(valid_actions) == 0:
                    break
                
                # Greedy action
                best_idx = np.argmax(q_values)
                action = valid_actions[best_idx]
                
                # Take step
                if use_continuous_eval:
                    # CONTINUOUS EVALUATION: Use continuous environment step
                    next_state, reward, done, info = env.step(state, action)
                else:
                    # DISCRETE EVALUATION: Use discretized environment step
                    next_state, reward, done, info = q_agent.step_with_discretization(
                        state, action, action_key
                    )
                
                # Accumulate reward
                episode_reward += float(reward)
                episode_steps += 1
                state = next_state
                
                if done:
                    episode_done = True
                    break
            
            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)
            completions.append(1.0 if episode_done else 0.0)
    else:
        # Different lengths: create Cartesian product (backward compatibility)
        for start in eval_starts_arr:
            for pickup in eval_pickups_arr:
                key, init_key = jax_random.split(key)
                state, _ = init_env(init_key, int(start), int(pickup), env.neighbor_mask_static)
                
                episode_reward = 0.0
                episode_steps = 0
                episode_done = False
                
                # Run episode with greedy policy (epsilon=0)
                for step in range(max_steps):
                    if state.done:
                        episode_done = True
                        break
                    
                    # Greedy action (epsilon=0)
                    key, action_key = jax_random.split(key)
                    
                    if use_continuous_eval:
                        # CONTINUOUS EVALUATION: Discretize state time for Q-table lookup
                        state_for_q_lookup = _discretize_state_time(state, dt)
                    else:
                        # DISCRETE EVALUATION: Use state directly (already in discrete space)
                        state_for_q_lookup = state
                    
                    # Get Q-values for all valid actions
                    valid_actions = []
                    q_values = []
                    for action in range(env.max_deg):
                        if state_for_q_lookup.neighbor_mask[action]:
                            valid_actions.append(action)
                            q_values.append(q_agent.get_q_value(state_for_q_lookup, action))
                    
                    if len(valid_actions) == 0:
                        break
                    
                    # Greedy action
                    best_idx = np.argmax(q_values)
                    action = valid_actions[best_idx]
                    
                    # Take step
                    if use_continuous_eval:
                        # CONTINUOUS EVALUATION: Use continuous environment step
                        next_state, reward, done, info = env.step(state, action)
                    else:
                        # DISCRETE EVALUATION: Use discretized environment step
                        next_state, reward, done, info = q_agent.step_with_discretization(
                            state, action, action_key
                        )
                    
                    # Accumulate reward
                    episode_reward += float(reward)
                    episode_steps += 1
                    state = next_state
                    
                    if done:
                        episode_done = True
                        break
                
                total_rewards.append(episode_reward)
                total_steps.append(episode_steps)
                completions.append(1.0 if episode_done else 0.0)
    
    return {
        'avg_reward': np.mean(total_rewards),
        'avg_steps': np.mean(total_steps),
        'completion_rate': np.mean(completions),
        'rewards': total_rewards,
        'steps': total_steps,
        'completions': completions,
    }

