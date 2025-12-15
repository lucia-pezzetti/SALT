"""
Training function for tabular Q-learning when discrete=True.
"""
import jax
import jax.numpy as jnp
from jax import random as jax_random
from jax import vmap
import numpy as np
import wandb
from typing import Tuple, Optional
from pathlib import Path

from taxi_env import TaxiState, init_env
from training.q_learning import TabularQLearning
from utils import offline_shortest_path_action


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
            
            # Get expert shortest path action
            expert_action = offline_shortest_path_action(
                state.current_node,
                state.pickup_node,
                env.adj_list,
                env.travel_times,
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
            eval_results = evaluate_q_agent(
                q_agent, env, eval_starts, eval_pickups, max_steps_per_episode, seed=seed
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
    max_steps_per_episode: int = 300,
    dt: float = 5.0,
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
    # Initialize Q-learning agent
    if load_path is not None and Path(load_path).exists():
        print(f"Loading Q-table from {load_path}")
        q_agent = TabularQLearning.load(env, load_path)
    else:
        q_agent = TabularQLearning(
            env=env,
            dt=dt,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            initial_q_value=initial_q_value,
        )
    
    # Initialize random key
    key = jax_random.PRNGKey(seed)
    
    # Evaluation sets
    if eval_starts is None:
        eval_starts = fixed_starts[:min(5, len(fixed_starts))]
    if eval_pickups is None:
        eval_pickups = fixed_pickups[:min(5, len(fixed_pickups))]
    
    # Initialize Q-table from shortest path travel times
    if init_from_shortest_paths:
        print(f"\n{'='*60}")
        print(f"Initializing Q-table from shortest path travel times")
        print(f"{'='*60}")
        stats_before = q_agent.get_statistics()
        print(f"Q-table BEFORE shortest path initialization:")
        print(f"  Num states: {stats_before['num_states']}")
        print(f"  Avg Q-value: {stats_before['avg_q_value']:.4f}")
        
        q_agent.initialize_q_values_from_shortest_paths(
            use_all_time_slices=init_all_time_slices,
            max_time_slices=100,
        )
        
        stats_after = q_agent.get_statistics()
        print(f"Q-table AFTER shortest path initialization:")
        print(f"  Num states: {stats_after['num_states']}")
        print(f"  Avg Q-value: {stats_after['avg_q_value']:.4f}")
        print(f"  States added: {stats_after['num_states'] - stats_before['num_states']}")
        print(f"{'='*60}\n")
    
    # Pretraining with shortest path rollouts
    if pretrain_enabled:
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
    
    print(f"Starting Q-learning training with {num_episodes} episodes...")
    print(f"Discretization dt={dt}, learning_rate={learning_rate}, gamma={discount_factor}")
    
    for episode in range(num_episodes):
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
        
        # Run episode
        for step in range(max_steps_per_episode):
            if state.done:
                episode_done = True
                break
            
            # Select action
            training_step = episode * max_steps_per_episode + step
            key, action_key = jax_random.split(key)
            action = q_agent.get_action(state, training_step, action_key)
            
            # Take step with discretization
            next_state, reward, done, info = q_agent.step_with_discretization(
                state, action, action_key
            )
            
            # Update Q-value
            q_agent.update_q_value(state, action, reward, next_state, done)
            
            # Update statistics
            episode_reward += float(reward)
            episode_length += 1
            
            # Move to next state
            state = next_state
            
            if done:
                episode_done = True
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_completions.append(1.0 if episode_done else 0.0)
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            completion_rate = np.mean(episode_completions[-100:])
            epsilon = q_agent.get_epsilon(training_step)
            stats = q_agent.get_statistics()
            
            # print(f"Episode {episode + 1}/{num_episodes}: "
            #       f"Avg reward={avg_reward:.2f}, "
            #       f"Avg length={avg_length:.1f}, "
            #       f"Completion={completion_rate:.2%}, "
            #       f"Epsilon={epsilon:.3f}, "
            #       f"Q-table size={stats['num_states']}")
            
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
        
        # Evaluation
        if (episode + 1) % eval_frequency == 0:
            eval_results = evaluate_q_agent(
                q_agent, env, eval_starts, eval_pickups, max_steps_per_episode
            )
            
            # print(f"Evaluation at episode {episode + 1}: "
            #       f"Avg reward={eval_results['avg_reward']:.2f}, "
            #       f"Avg steps={eval_results['avg_steps']:.1f}, "
            #       f"Completion={eval_results['completion_rate']:.2%}")
            
            if wandb.run is not None:
                wandb.log({
                    'evaluation/episode': episode + 1,
                    'evaluation/avg_reward': eval_results['avg_reward'],
                    'evaluation/avg_steps': eval_results['avg_steps'],
                    'evaluation/completion_rate': eval_results['completion_rate'],
                })
    
    print("Q-learning training completed!")
    if save_path is not None:
        print(f"Saving Q-table to {save_path}")
        q_agent.save(save_path)
    return q_agent


def evaluate_q_agent(
    q_agent: TabularQLearning,
    env,
    eval_starts,
    eval_pickups,
    max_steps: int = 300,
    seed: int = 42,
) -> dict:
    """
    Evaluate Q-learning agent on fixed start-pickup pairs.
    
    Args:
        q_agent: Trained Q-learning agent
        env: TaxiEnv instance
        eval_starts: Array of start node indices
        eval_pickups: Array of pickup node indices
        max_steps: Maximum steps per episode
        seed: Random seed
        
    Returns:
        Dictionary with evaluation metrics
    """
    key = jax_random.PRNGKey(seed)
    
    total_rewards = []
    total_steps = []
    completions = []
    
    # Evaluate on all combinations
    for start in eval_starts:
        for pickup in eval_pickups:
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
                # Get Q-values for all valid actions
                valid_actions = []
                q_values = []
                for action in range(env.max_deg):
                    if state.neighbor_mask[action]:
                        valid_actions.append(action)
                        q_values.append(q_agent.get_q_value(state, action))
                
                if len(valid_actions) == 0:
                    break
                
                # Greedy action
                best_idx = np.argmax(q_values)
                action = valid_actions[best_idx]
                
                # Take step
                next_state, reward, done, info = q_agent.step_with_discretization(
                    state, action, action_key
                )

                # print(f"Step {step}: Node={state.current_node}, Action={action}, Reward={reward}, Done={done}")
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

