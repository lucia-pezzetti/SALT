import json
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import wandb
from datetime import datetime

from taxi_env import TaxiState, init_env
from training.mcts import (
    get_init_fn, get_recurrent_fn, get_agent_loop, estimate_returns_batch
)
from utils import offline_shortest_path_action
from evaluation.plot_agent_paths import plot_rl_vs_shortest_path

from .context import RunContext


def run_mcts(args, ctx: RunContext) -> None:
    obs_fn_single, obs_fn_batch = ctx.obs_fn_single, ctx.obs_fn_batch

    if args.config is None:
        raise ValueError("Must pass --config path to your JSON for MCTS")
    with open(args.config, "r") as f:
        config = json.load(f)

    config['batch_size'] = args.num_agents
    config['num_steps'] = args.epochs * config['eval_frequency']
    config['num_simulations'] = min(2*ctx.max_length, 64)
    config['cycle_length'] = args.cycle_length


    # Initialize wandb for logging
    wandb.init(
        project=getattr(args, 'wandb_project', 'taxi-mcts'),
        name=f"mcts-{getattr(args, 'env_type', 'unknown')}-{args.epochs}epochs-{args.num_agents}agents-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "env_type": getattr(args, 'env_type', 'unknown'),
            "model": "mcts",
            "epochs": args.epochs,
            "num_agents": args.num_agents,
            "batch_size": args.num_agents,
            "num_steps": config['num_steps'],
            "gamma": ctx.env.gamma,
            "num_nodes": ctx.env.num_nodes,
            "max_deg": ctx.env.max_deg,
            "max_steps": ctx.env.max_steps,
            "pickup_bonus": ctx.env.pickup_bonus,
            "timeout_penalty": ctx.env.timeout_penalty,
            "mcts_config": config,
            "timestamp": datetime.now().isoformat(),
        }
    )

    init_fn = get_init_fn(ctx.env, config, obs_fn_single)
    key, env_states, V_apply, V_opt_state, V_opt_update, get_V_params, V_target_params = init_fn(jax.random.PRNGKey(0))

    def linear_epsilon_decay(initial_eps=0.9, final_eps=0.05, decay_steps=10000):
        def schedule(step):
            progress = jnp.clip(step / decay_steps, 0.0, 1.0)
            return initial_eps * (1.0 - progress) + final_eps * progress
        return schedule

    key, subkey = jax.random.split(key)
    epsilon_schedule = linear_epsilon_decay(initial_eps=0.9, final_eps=0.05, decay_steps=config['num_steps'])
    recurrent_fn = get_recurrent_fn(ctx.env, V_apply, obs_fn_batch, epsilon_schedule, curriculum_steps=config['num_steps']*0.8)
    agent_loop = get_agent_loop(ctx.env, config, obs_fn_batch, V_apply, recurrent_fn, V_opt_update, get_V_params, epsilon_schedule)

    state_dict = {
        'key': key,
        'env_states': env_states,
        'last_start': env_states.current_node,
        'V_opt_state': V_opt_state,
        'V_params': V_target_params,
        'V_target_params': V_target_params,
        'opt_t': 0,
        'avg_return': jnp.zeros(config['batch_size']),
        'episode_return': jnp.zeros(config['batch_size']),
        'num_episodes': jnp.zeros(config['batch_size']),
        'visit_counts': jnp.zeros(ctx.env.num_nodes, dtype=jnp.int32), 
        'cumulative_visits': jnp.zeros(ctx.env.num_nodes, dtype=jnp.int32),
        'loss': jnp.array(0.0),
        'episode_travel': jnp.zeros(config['batch_size']),
        'episode_wait': jnp.zeros(config['batch_size']),
        'episode_total_time': jnp.zeros(config['batch_size']),
        'avg_wait': jnp.zeros(config['batch_size']),
        'avg_travel': jnp.zeros(config['batch_size']),
        'avg_total_time': jnp.zeros(config['batch_size']),
        'completed_total_time': jnp.array(0.0),
        'completed_travel_time': jnp.array(0.0),
        'completed_wait_time': jnp.array(0.0),
        'completed_episodes': jnp.array(0.0),
    }

    num_eval_steps = config['num_steps'] // config['eval_frequency']
    avg_returns = np.zeros(num_eval_steps, dtype=np.float32)
    times = np.zeros(num_eval_steps, dtype=np.float32)

    print(f"Training will run for {num_eval_steps} evaluation steps...")

    for i in range(num_eval_steps):
        step_start_time = time.time()
        state_dict, metrics = agent_loop(state_dict)
        step_time = time.time() - step_start_time

        def sp_policy(state: TaxiState) -> int:
            curr = int(state.current_node)
            pickup = int(state.pickup_node)
            return int(offline_shortest_path_action(curr, pickup, ctx.env.adj_list, ctx.env.travel_times, ctx.env.distances, ctx.env.neighbor_mask_static[curr]))

        def evaluate_sp_policy(starts, pickups):
            sp_times = []
            starts_np = np.array(starts) if hasattr(starts, '__iter__') else np.array([starts])
            pickups_np = np.array(pickups) if hasattr(pickups, '__iter__') else np.array([pickups])
            for start, pickup in zip(starts_np, pickups_np):
                total_time = 0.0
                state = init_env(jax.random.PRNGKey(0), int(start), int(pickup), ctx.env.neighbor_mask_static)[0]
                step_count = 0
                while not state.done and step_count < ctx.env.max_steps:
                    action = sp_policy(state)
                    state, _, _, info = ctx.env.step(state, action)
                    total_time += float(info['travel'] + info['wait'])
                    step_count += 1
                sp_times.append(total_time)
            return np.array(sp_times)

        starts = np.array([int(x) for x in metrics['starts']])
        pickups = np.array([int(x) for x in metrics['pickups']])
        sp_times = evaluate_sp_policy(starts, pickups)
        avg_sp_total_time = np.mean(sp_times)

        performance_ratio = float(metrics['avg_total_time']) / float(avg_sp_total_time) if avg_sp_total_time > 0 else float('inf')

        log_metrics = {
            'training/step': i,
            'training/step_time': step_time,
            'training/loss': float(metrics['loss']),
            'training/avg_return': float(jnp.mean(metrics['avg_return'])),
            'training/avg_wait': float(metrics['avg_wait']),
            'training/avg_travel': float(metrics['avg_travel']),
            'training/avg_total_time': float(metrics['avg_total_time']),
            'training/avg_sp_total_time': float(avg_sp_total_time),
            'training/performance_ratio': performance_ratio,
            'training/opt_step': int(state_dict['opt_t']),
            'training/epsilon': float( (lambda t: max(0.05, 0.9 - 0.85 * (t / (config['num_steps'] or 1)))) (state_dict['opt_t']) ),
            'training/visit_freq_entropy': float(-jnp.sum(metrics['visit_freq'] * jnp.log(metrics['visit_freq'] + 1e-8))),
            'training/unique_nodes_visited': float(jnp.sum(metrics['visit_freq'] > 0)),
            'training/total_visits': float(jnp.sum(metrics['visit_freq'] * metrics['visit_freq'].sum())),
            'training/visit_diversity': float(jnp.sum(metrics['visit_freq'] > 0) / ctx.env.num_nodes),
            'training/num_episodes': float(jnp.sum(state_dict['num_episodes'])),
            'training/avg_episode_length': float(jnp.mean(state_dict['avg_total_time'])),
            'training/episode_completion_rate': float(jnp.sum(state_dict['num_episodes']) / (i + 1) / config['batch_size']),
            'training/improvement_over_sp': float(1.0 - performance_ratio),
            'training/relative_efficiency': float(avg_sp_total_time / metrics['avg_total_time']),
            'learning/loss': float(metrics['loss']),
            'learning/return_smooth': float(jnp.mean(metrics['avg_return'])),
            'learning/performance_smooth': performance_ratio,
            'learning/convergence_indicator': float(jnp.abs(performance_ratio - 1.0)),
            'learning/learning_rate': float(1.0 / (i + 1)),
            'learning/return_std': float(jnp.std(metrics['avg_return'])),
            'learning/time_std': float(jnp.std(metrics['avg_total_time'])),
            'learning/coefficient_of_variation': float(jnp.std(metrics['avg_total_time']) / (jnp.mean(metrics['avg_total_time']) + 1e-8)),
            'evaluation/value_difference': float(metrics.get('value_difference', 0.0)),
        }
        wandb.log(log_metrics, step=i)

        if i % 50 == 0 or i == num_eval_steps - 1:
            eval_metrics = {
                'evaluation/step': i,
                'evaluation/performance_ratio': performance_ratio,
                'evaluation/improvement_over_sp': float(1.0 - performance_ratio),
                'evaluation/avg_total_time': float(metrics['avg_total_time']),
                'evaluation/avg_sp_total_time': float(avg_sp_total_time),
                'evaluation/learning_progress': float(i / num_eval_steps),
                'evaluation/exploration_entropy': float(-jnp.sum(metrics['visit_freq'] * jnp.log(metrics['visit_freq'] + 1e-8))),
            }
            wandb.log(eval_metrics, step=i)

        if i % 100 == 0 or i == num_eval_steps - 1:
            print(
                f"Step {i}/{num_eval_steps} | Loss: {metrics['loss']:.4f} | "
                f"Performance Ratio: {performance_ratio:.3f} | Avg Time: {metrics['avg_total_time']:.2f}s | SP Time: {avg_sp_total_time:.2f}s | "
                f"Improvement: {(1.0 - performance_ratio)*100:.1f}%"
            )

        state_dict.update({
            'episode_return': jnp.zeros(config['batch_size']),
            'avg_travel': jnp.zeros(config['batch_size']),
            'avg_wait': jnp.zeros(config['batch_size']),
            'avg_return': jnp.zeros(config['batch_size']),
            'num_episodes': jnp.zeros(config['batch_size']),
            'episode_travel': jnp.zeros(config['batch_size']),
            'episode_wait': jnp.zeros(config['batch_size']),
            'visit_counts': jnp.zeros(ctx.env.num_nodes, dtype=jnp.int32),
            'loss': jnp.array(0.0),
        })

    final_V_params = state_dict['V_target_params']

    with open(args.params_dir + '.out', 'wb') as f:
        pickle.dump({'config': config, 'avg_returns': avg_returns, 'times': times}, f)
    with open(args.params_dir + '.params', 'wb') as f:
        pickle.dump({'V': final_V_params}, f)

    def greedy_V_policy(state: TaxiState) -> int:
        curr = state.current_node
        neighbors = ctx.env.adj_list[curr]
        mask = state.neighbor_mask
        best_a = None
        best_v = -jnp.inf
        for i, (n, valid) in enumerate(zip(neighbors, mask)):
            if not valid:
                continue
            next_state, reward, done, _ = ctx.env.step(state, i)
            if done:
                value = reward
            else:
                next_obs = obs_fn_single(next_state)
                next_v = V_apply(final_V_params, next_obs.astype(float))
                value = reward + ctx.env.gamma * next_v
            if value > best_v:
                best_v, best_a = value, i
        return int(best_a) if best_a is not None else 0

    def sp_policy(state: TaxiState) -> int:
        curr = int(state.current_node)
        pickup = int(state.pickup_node)
        return int(offline_shortest_path_action(curr, pickup, ctx.env.adj_list, ctx.env.travel_times, ctx.env.distances, ctx.env.neighbor_mask_static[curr]))

    def evaluate_policy(policy, starts, pickups, max_steps=100):
        times = []
        paths = []
        rewards = []
        for start, pickup in zip(starts, pickups):
            total_time = 0.0
            total_reward = 0.0
            step_count = 0
            state = init_env(jax.random.PRNGKey(0), start, pickup, ctx.env.neighbor_mask_static)[0]
            paths = [state.current_node]
            print(f"Start: {start}, Pickup: {pickup}, Initial node: {state.current_node}")
            while not state.done and step_count < max_steps:
                action = policy(state)
                state, reward, _, info = ctx.env.step(state, action)
                total_time += float(info['travel'] + info['wait'])
                step_count += 1
                print(f"Step {step_count}: Node {int(state.current_node)}, Action {action}, Travel: {info['travel']:.2f}, Wait: {info['wait']:.2f}")
                paths.append(state.current_node)
                total_reward += float(reward)
            times.append(total_time)
            rewards.append(total_reward)
        return np.array(times), np.array(paths), np.array(rewards)  # type: ignore

    # Generate all combinations of starts and pickups
    all_starts = np.array(ctx.env.fixed_starts)
    all_pickups = np.array(ctx.env.fixed_pickups)
    num_starts = len(all_starts)
    num_pickups = len(all_pickups)
    total_combinations = num_starts * num_pickups
    
    print(f"Evaluating all {total_combinations} combinations of {num_starts} starts and {num_pickups} pickups...")
    
    # Generate all combinations
    all_combinations = []
    for start in all_starts:
        for pickup in all_pickups:
            all_combinations.append((int(start), int(pickup)))
    
    # Evaluate all combinations
    all_v_times = []
    all_sp_times = []
    all_v_rewards = []
    all_sp_rewards = []
    
    for i, (start, pickup) in enumerate(all_combinations):
        eval_starts = jnp.array([start])
        eval_pickups = jnp.array([pickup])
        
        # Evaluate RL policy (greedy V policy)
        v_time, v_paths, v_rewards = evaluate_policy(greedy_V_policy, [start], [pickup])
        all_v_times.append(v_time[0])
        all_v_rewards.append(v_rewards[0])
        
        # Evaluate SP policy
        sp_time, sp_paths, sp_rewards = evaluate_policy(sp_policy, [start], [pickup])
        all_sp_times.append(sp_time[0])
        all_sp_rewards.append(sp_rewards[0])
        
        if (i + 1) % 4 == 0 or i == len(all_combinations) - 1:
            print(f"Progress: {i+1}/{len(all_combinations)} combinations evaluated...")
            print(f"  Start: {start}, Pickup: {pickup}")
            print(f"  RL Time: {v_time[0]:.2f}, SP Time: {sp_time[0]:.2f}")
            print(f"  RL Reward: {v_rewards[0]:.2f}, SP Reward: {sp_rewards[0]:.2f}")
        
        # Plot for each combination
        fig = plot_rl_vs_shortest_path(v_paths, sp_paths, ctx.G, ctx.node_to_idx, ctx.idx_to_node, eval_starts, eval_pickups)
        fig.savefig(f"rl_vs_sp_[{start}]_[{pickup}].png")
    
    # Print summary statistics
    all_v_times = np.array(all_v_times)
    all_sp_times = np.array(all_sp_times)
    all_v_rewards = np.array(all_v_rewards)
    all_sp_rewards = np.array(all_sp_rewards)
    
    print(f"\n=== Final Evaluation Summary ===")
    print(f"Total combinations evaluated: {len(all_combinations)}")
    print(f"Value Policy   avg time: {all_v_times.mean():.2f} (std: {all_v_times.std():.2f})")
    print(f"SP Policy      avg time: {all_sp_times.mean():.2f} (std: {all_sp_times.std():.2f})")
    print(f"Value Policy   avg reward: {all_v_rewards.mean():.2f} (std: {all_v_rewards.std():.2f})")
    print(f"SP Policy      avg reward: {all_sp_rewards.mean():.2f} (std: {all_sp_rewards.std():.2f})")
    print(f"Performance ratio (RL/SP): {all_v_times.mean() / all_sp_times.mean():.3f}")


