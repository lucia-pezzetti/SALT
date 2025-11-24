import numpy as np
import jax
import jax.numpy as jnp
import optax
import wandb

from datetime import datetime

from taxi_env import TaxiState, init_env
from training.train_q_learning import train_q_learning, evaluate_q_agent
from utils import offline_shortest_path_action
from evaluation.plot_agent_paths import plot_rl_vs_shortest_path

from .context import RunContext


def run_q_learning(args, ctx: RunContext) -> None:
    """Run tabular Q-learning training when discrete=True."""
    print("="*60)
    print("Running Tabular Q-Learning (Discrete Mode)")
    print("="*60)
    
    # Generate evaluation sets
    eval_starts_list = ctx.env.fixed_starts[:min(5, len(ctx.env.fixed_starts))].tolist() if hasattr(ctx.env.fixed_starts, 'tolist') else list(ctx.env.fixed_starts[:min(5, len(ctx.env.fixed_starts))])
    eval_pickups_list = ctx.env.fixed_pickups[:min(5, len(ctx.env.fixed_pickups))].tolist() if hasattr(ctx.env.fixed_pickups, 'tolist') else list(ctx.env.fixed_pickups[:min(5, len(ctx.env.fixed_pickups))])
    
    eval_starts_combined = []
    eval_pickups_combined = []
    for start in eval_starts_list:
        for pickup in eval_pickups_list:
            eval_starts_combined.append(int(start))
            eval_pickups_combined.append(int(pickup))
    
    eval_starts = jnp.array(eval_starts_combined, dtype=jnp.int32)
    eval_pickups = jnp.array(eval_pickups_combined, dtype=jnp.int32)
    
    print(f"Evaluating on {len(eval_starts)} combinations: {len(eval_starts_list)} starts × {len(eval_pickups_list)} pickups")
    
    # Initialize wandb
    wandb.init(
        project="ride-sharing-q-learning",
        name=f"qlearning-{args.env_type}-{args.epochs}episodes-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "env_type": args.env_type,
            "num_layers": args.num_layers,
            "layer_width": args.layer_width,
            "offset": args.offset,
            "cycle_length": args.cycle_length,
            "no_congestion": args.no_congestion,
            "place_name": args.place_name,
            "model": "q_learning",
            "discrete": True,
            "dt": 5.0,
            "num_episodes": args.epochs,
            "num_agents": args.num_agents,
            "gamma": args.gamma,
            "num_nodes": ctx.env.num_nodes,
            "max_deg": ctx.env.max_deg,
            "max_steps": ctx.env.max_steps,
            "pickup_bonus": ctx.env.pickup_bonus,
            "timeout_penalty": ctx.env.timeout_penalty,
            "eval_starts": eval_starts_list,
            "eval_pickups": eval_pickups_list,
            "eval_combinations": len(eval_starts),
            "timestamp": datetime.now().isoformat(),
            "pretrain_enabled": getattr(args, 'pretrain_enabled', False),
            "num_pretrain_episodes": getattr(args, 'num_pretrain_episodes', 1000),
            "epsilon_start": getattr(args, 'epsilon_start', 1.0),
            "epsilon_end": getattr(args, 'epsilon_end', 0.01),
            "epsilon_decay_fraction": getattr(args, 'epsilon_decay_fraction', 0.5),
        }
    )
    
    # Train Q-learning agent
    # Calculate epsilon decay steps based on total training steps, not just episodes
    # Get epsilon parameters from command-line arguments or use defaults
    epsilon_start = getattr(args, 'epsilon_start', 1.0)
    epsilon_end = getattr(args, 'epsilon_end', 0.01)
    epsilon_decay_fraction = getattr(args, 'epsilon_decay_fraction', 0.5)  # Fraction of training steps for decay
    
    total_training_steps = args.epochs * ctx.max_length
    epsilon_decay_steps = int(total_training_steps * epsilon_decay_fraction)
    
    print(f"\nEpsilon Decay Schedule:")
    print(f"  Total training steps: {total_training_steps:,}")
    print(f"  Epsilon decay steps: {epsilon_decay_steps:,}")
    print(f"  Epsilon will decay from {epsilon_start} to {epsilon_end} over {epsilon_decay_steps:,} steps")
    print(f"  (This is {epsilon_decay_steps / total_training_steps * 100:.1f}% of total training)\n")
    
    # Check if pretraining is enabled
    pretrain_enabled = getattr(args, 'pretrain_enabled', False)
    num_pretrain_episodes = getattr(args, 'num_pretrain_episodes', 1000)
    
    # Create logging function for pretraining metrics
    def pretrain_log_fn(log_dict, step):
        """Log pretraining metrics to wandb"""
        if wandb.run is not None:
            wandb.log(log_dict, step=step)
    
    q_agent = train_q_learning(
        env=ctx.env,
        fixed_starts=ctx.env.fixed_starts,
        fixed_pickups=ctx.env.fixed_pickups,
        num_episodes=args.epochs,
        max_steps_per_episode=ctx.max_length,
        dt=1.0,
        learning_rate=0.1,
        discount_factor=args.gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        eval_frequency=100,
        eval_starts=eval_starts,
        eval_pickups=eval_pickups,
        seed=args.seed,
        pretrain_enabled=pretrain_enabled,
        num_pretrain_episodes=num_pretrain_episodes,
        pretrain_log_fn=pretrain_log_fn if pretrain_enabled else None,
        save_path=args.q_table_path,
        load_path=args.q_table_path,
    )
    
    # Final evaluation
    final_eval = evaluate_q_agent(
        q_agent, ctx.env, eval_starts, eval_pickups, ctx.max_length, seed=args.seed
    )
    
    print(f"\nFinal Evaluation Results:")
    print(f"  Average Reward: {final_eval['avg_reward']:.2f}")
    print(f"  Average Steps: {final_eval['avg_steps']:.1f}")
    print(f"  Completion Rate: {final_eval['completion_rate']:.2%}")
    
    # Log final metrics
    wandb.log({
        "final/avg_reward": final_eval['avg_reward'],
        "final/avg_steps": final_eval['avg_steps'],
        "final/completion_rate": final_eval['completion_rate'],
    })
    
    # Shortest path baseline - CONTINUOUS (real-world comparison)
    def evaluate_shortest_path_baseline_continuous(starts, pickups):
        """Shortest path in continuous time - real-world comparison."""
        sp_times, sp_rewards, sp_steps, sp_completed = [], [], [], []
        for start, pickup in zip(starts, pickups):
            total_time = 0.0
            total_reward = 0.0
            step_count = 0
            state = init_env(jax.random.PRNGKey(0), start, pickup, ctx.env.neighbor_mask_static)[0]
            while not state.done and step_count < ctx.env.max_steps:
                action = offline_shortest_path_action(
                    state.current_node, state.pickup_node,
                    ctx.env.adj_list, ctx.env.travel_times, ctx.env.distances,
                    ctx.env.neighbor_mask_static[state.current_node]
                )
                state, reward, done, info = ctx.env.step(state, action)
                total_time += float(info['travel'] + info['wait'])
                total_reward += float(reward)
                step_count += 1
            sp_times.append(total_time)
            sp_rewards.append(total_reward)
            sp_steps.append(step_count)
            sp_completed.append(bool(state.done))
        return {
            'avg_time': np.mean(sp_times),
            'avg_reward': np.mean(sp_rewards),
            'avg_steps': np.mean(sp_steps),
            'completion_rate': np.mean(sp_completed),
            'times': sp_times,
            'rewards': sp_rewards,
            'steps': sp_steps,
        }
    
    # Shortest path baseline - DISCRETE (fair comparison with Q-learning)
    def evaluate_shortest_path_baseline_discrete(starts, pickups):
        """Shortest path in discrete time - fair comparison with Q-learning."""
        sp_times, sp_rewards, sp_steps, sp_completed = [], [], [], []
        for start, pickup in zip(starts, pickups):
            total_time = 0.0
            total_reward = 0.0
            step_count = 0
            state = init_env(jax.random.PRNGKey(0), start, pickup, ctx.env.neighbor_mask_static)[0]
            while not state.done and step_count < ctx.env.max_steps:
                action = offline_shortest_path_action(
                    state.current_node, state.pickup_node,
                    ctx.env.adj_list, ctx.env.travel_times, ctx.env.distances,
                    ctx.env.neighbor_mask_static[state.current_node]
                )
                # Use discretized step for fair comparison with Q-learning
                state, reward, done, info = q_agent.step_with_discretization(
                    state, action, jax.random.PRNGKey(step_count)
                )
                total_time += float(info['travel'] + info['wait'])
                total_reward += float(reward)
                step_count += 1
            sp_times.append(total_time)
            sp_rewards.append(total_reward)
            sp_steps.append(step_count)
            sp_completed.append(bool(state.done))
        return {
            'avg_time': np.mean(sp_times),
            'avg_reward': np.mean(sp_rewards),
            'avg_steps': np.mean(sp_steps),
            'completion_rate': np.mean(sp_completed),
            'times': sp_times,
            'rewards': sp_rewards,
            'steps': sp_steps,
        }
    
    sp_baseline_continuous = evaluate_shortest_path_baseline_continuous(eval_starts, eval_pickups)
    sp_baseline_discrete = evaluate_shortest_path_baseline_discrete(eval_starts, eval_pickups)
    
    print(f"\nShortest Path Baseline (Continuous - Real World):")
    print(f"  Average Reward: {sp_baseline_continuous['avg_reward']:.2f}")
    print(f"  Average Steps: {sp_baseline_continuous['avg_steps']:.1f}")
    print(f"  Completion Rate: {sp_baseline_continuous['completion_rate']:.2%}")
    
    print(f"\nShortest Path Baseline (Discrete - Fair Q-Learning Comparison):")
    print(f"  Average Reward: {sp_baseline_discrete['avg_reward']:.2f}")
    print(f"  Average Steps: {sp_baseline_discrete['avg_steps']:.1f}")
    print(f"  Completion Rate: {sp_baseline_discrete['completion_rate']:.2%}")
    
    wandb.log({
        "baseline/sp_continuous_avg_reward": sp_baseline_continuous['avg_reward'],
        "baseline/sp_continuous_avg_steps": sp_baseline_continuous['avg_steps'],
        "baseline/sp_continuous_completion_rate": sp_baseline_continuous['completion_rate'],
        "baseline/sp_discrete_avg_reward": sp_baseline_discrete['avg_reward'],
        "baseline/sp_discrete_avg_steps": sp_baseline_discrete['avg_steps'],
        "baseline/sp_discrete_completion_rate": sp_baseline_discrete['completion_rate'],
    })
    
    # Create Q-learning greedy policy
    def q_learning_policy(state: TaxiState) -> int:
        """Greedy Q-learning policy (epsilon=0)."""
        valid_actions = []
        q_values = []
        for action in range(ctx.env.max_deg):
            if state.neighbor_mask[action]:
                valid_actions.append(action)
                q_values.append(q_agent.get_q_value(state, action))
        
        if len(valid_actions) == 0:
            return 0  # Fallback
        
        best_idx = np.argmax(q_values)
        return valid_actions[best_idx]
    
    # Shortest path policy
    def sp_policy(state: TaxiState) -> int:
        return offline_shortest_path_action(
            state.current_node, state.pickup_node,
            ctx.env.adj_list, ctx.env.travel_times, ctx.env.distances,
            ctx.env.neighbor_mask_static[state.current_node]
        )
    
    def simulate_policy_time(policy_fn, start, pickup, base_seed, use_discrete):
        """Simulate a single start/pickup pair and return total travel time (seconds)."""
        key = jax.random.PRNGKey(base_seed)
        state = init_env(key, int(start), int(pickup), ctx.env.neighbor_mask_static)[0]
        total_time = 0.0
        steps = 0
        
        while (not bool(state.done)) and steps < ctx.max_length:
            action = policy_fn(state)
            if use_discrete:
                key, step_key = jax.random.split(key)
                next_state, reward, done, info = q_agent.step_with_discretization(
                    state, action, step_key
                )
            else:
                next_state, reward, done, info = ctx.env.step(state, action)
            
            total_time += float(info['travel'] + info['wait'])
            state = next_state
            steps += 1
            
            if done:
                break
        
        return total_time
    
    def build_cost_matrix(starts, pickups, policy_fn, base_seed, use_discrete):
        """Return a [len(starts), len(pickups)] matrix of total travel times."""
        starts_np = np.array(starts).astype(int)
        pickups_np = np.array(pickups).astype(int)
        cost = np.zeros((len(starts_np), len(pickups_np)), dtype=np.float32)
        
        for i, start in enumerate(starts_np):
            for j, pickup in enumerate(pickups_np):
                pair_seed = base_seed + i * 7919 + j * 104729
                cost[i, j] = simulate_policy_time(
                    policy_fn, start, pickup, pair_seed, use_discrete=use_discrete
                )
        
        return jnp.array(cost, dtype=jnp.float32)
    
    def match_pickups(starts, pickups, policy_fn, base_seed, use_discrete):
        """Assign pickups to starts using Hungarian matching on travel-time cost."""
        if len(starts) == 0:
            return pickups, jnp.zeros((0, 0), dtype=jnp.float32)
        
        cost_matrix = build_cost_matrix(starts, pickups, policy_fn, base_seed, use_discrete)
        _, assignment = optax.assignment.hungarian_algorithm(cost_matrix)
        matched_pickups = jnp.take(pickups, assignment, axis=0)
        return matched_pickups, cost_matrix
    
    # Evaluation loop with trajectory printing
    eval_iter = 20
    eval_loop_key = jax.random.PRNGKey(args.seed + 1000)
    
    print("\n" + "="*60)
    print("Starting trajectory evaluation and visualization...")
    print("="*60)
    
    for i in range(eval_iter):
        B = args.num_agents
        eval_loop_key, eval_key = jax.random.split(eval_loop_key)
        eval_keys = jax.random.split(eval_key, 2*B+1)
        start_keys, pickup_keys, base_key = eval_keys[:B], eval_keys[B:2*B], eval_keys[-1]
        eval_starts = jnp.array([jax.random.choice(k, ctx.env.fixed_starts) for k in start_keys])
        eval_pickups = jnp.array([jax.random.choice(k, ctx.env.fixed_pickups) for k in pickup_keys])
        base_seed = int(args.seed + i * 1000)
        
        matched_q_pickups, _ = match_pickups(
            eval_starts, eval_pickups, q_learning_policy, base_seed, use_discrete=True
        )
        matched_sp_pickups_cont, _ = match_pickups(
            eval_starts, eval_pickups, sp_policy, base_seed + 1, use_discrete=False
        )
        matched_sp_pickups_disc, _ = match_pickups(
            eval_starts, eval_pickups, sp_policy, base_seed + 2, use_discrete=True
        )
        
        def evaluate_policy_single(policy_fn, starts, pickups, use_discrete=False):
            """Evaluate policy and return times, paths, and rewards."""
            times, paths, rewards = [], [], []
            for start, pickup in zip(np.array(starts).tolist(), np.array(pickups).tolist()):
                total_time, total_reward = 0.0, 0.0
                step_count = 0
                state = init_env(jax.random.PRNGKey(0), start, pickup, ctx.env.neighbor_mask_static)[0]
                traj = [int(state.current_node)]
                
                while (not bool(state.done)) and step_count < 3*ctx.max_length:
                    action = policy_fn(state)
                    
                    if use_discrete:
                        # Use discretized step for Q-learning
                        next_state, reward, done, info = q_agent.step_with_discretization(
                            state, action, jax.random.PRNGKey(step_count)
                        )
                    else:
                        # Use normal step for shortest path
                        next_state, reward, done, info = ctx.env.step(state, action)
                    
                    total_time += float(info['travel'] + info['wait'])
                    total_reward += float(reward)
                    traj.append(int(next_state.current_node))
                    step_count += 1
                    state = next_state
                    
                    if done:
                        break
                
                times.append(total_time)
                rewards.append(total_reward)
                paths.append(traj)
            
            return np.array(times), paths, np.array(rewards)
        
        print(f"\nEvaluation iteration {i+1}/{eval_iter}")
        print(f"Evaluating Q-learning policy for starts: {eval_starts} and pickups: {matched_q_pickups}")
        q_times, q_paths, q_rewards = evaluate_policy_single(
            q_learning_policy, eval_starts, matched_q_pickups, use_discrete=True
        )
        
        print(f"Evaluating SP policy (continuous) for starts: {eval_starts} and pickups: {matched_sp_pickups_cont}")
        sp_times_continuous, sp_paths_continuous, sp_rewards_continuous = evaluate_policy_single(
            sp_policy, eval_starts, matched_sp_pickups_cont, use_discrete=False
        )
        
        print(f"Evaluating SP policy (discrete) for starts: {eval_starts} and pickups: {matched_sp_pickups_disc}")
        sp_times_discrete, sp_paths_discrete, sp_rewards_discrete = evaluate_policy_single(
            sp_policy, eval_starts, matched_sp_pickups_disc, use_discrete=True
        )
        
        # Print trajectories
        print(f"\nQ-Learning Trajectories:")
        for agent_idx, (start, pickup, path, time, reward) in enumerate(zip(eval_starts, eval_pickups, q_paths, q_times, q_rewards)):
            print(f"  Agent {agent_idx+1}: Start={int(start)}, Pickup={int(pickup)}")
            print(f"    Path: {' -> '.join(map(str, path))}")
            print(f"    Time: {time:.2f}s, Reward: {reward:.2f}, Steps: {len(path)-1}")
        
        print(f"\nShortest Path Trajectories (Continuous - Real World):")
        for agent_idx, (start, pickup, path, time, reward) in enumerate(zip(eval_starts, eval_pickups, sp_paths_continuous, sp_times_continuous, sp_rewards_continuous)):
            print(f"  Agent {agent_idx+1}: Start={int(start)}, Pickup={int(pickup)}")
            print(f"    Path: {' -> '.join(map(str, path))}")
            print(f"    Time: {time:.2f}s, Reward: {reward:.2f}, Steps: {len(path)-1}")
        
        print(f"\nShortest Path Trajectories (Discrete - Fair Comparison):")
        for agent_idx, (start, pickup, path, time, reward) in enumerate(zip(eval_starts, eval_pickups, sp_paths_discrete, sp_times_discrete, sp_rewards_discrete)):
            print(f"  Agent {agent_idx+1}: Start={int(start)}, Pickup={int(pickup)}")
            print(f"    Path: {' -> '.join(map(str, path))}")
            print(f"    Time: {time:.2f}s, Reward: {reward:.2f}, Steps: {len(path)-1}")
        
        # Plot comparison: Q-learning vs SP discrete (fair comparison)
        q_paths_list = [list(map(int, path)) for path in q_paths]
        sp_paths_discrete_list = [list(map(int, path)) for path in sp_paths_discrete]
        fig = plot_rl_vs_shortest_path(q_paths_list, sp_paths_discrete_list, ctx.G, ctx.node_to_idx, ctx.idx_to_node, eval_starts, eval_pickups)
        fig.savefig(f"qlearning_vs_sp_discrete_{np.array(eval_starts).tolist()}_{np.array(eval_pickups).tolist()}.png")
        
        print(f"\nQ-learning avg time: {np.mean(q_times):.2f}")
        print(f"SP (continuous) avg time: {np.mean(sp_times_continuous):.2f}")
        print(f"SP (discrete) avg time: {np.mean(sp_times_discrete):.2f}")
        print(f"Saved plot: qlearning_vs_sp_discrete_{np.array(eval_starts).tolist()}_{np.array(eval_pickups).tolist()}.png")
        
        # Log metrics for both comparisons
        final_eval_metrics = {
            # Q-learning metrics
            "final_eval/qlearning_avg_time": float(np.mean(q_times)),
            "final_eval/qlearning_times": q_times.tolist(),
            
            # Continuous SP (real-world comparison)
            "final_eval/sp_continuous_avg_time": float(np.mean(sp_times_continuous)),
            "final_eval/qlearning_vs_sp_continuous_ratio": float(np.mean(q_times) / np.mean(sp_times_continuous)),
            "final_eval/qlearning_vs_sp_continuous_improvement": float((np.mean(sp_times_continuous) - np.mean(q_times)) / np.mean(sp_times_continuous) * 100),
            "final_eval/sp_continuous_times": sp_times_continuous.tolist(),
            
            # Discrete SP (fair comparison)
            "final_eval/sp_discrete_avg_time": float(np.mean(sp_times_discrete)),
            "final_eval/qlearning_vs_sp_discrete_ratio": float(np.mean(q_times) / np.mean(sp_times_discrete)),
            "final_eval/qlearning_vs_sp_discrete_improvement": float((np.mean(sp_times_discrete) - np.mean(q_times)) / np.mean(sp_times_discrete) * 100),
            "final_eval/sp_discrete_times": sp_times_discrete.tolist(),
        }
        wandb.log(final_eval_metrics)
        
        # Comparison table: Q-learning vs SP discrete (fair comparison)
        eval_comparison_table_discrete = wandb.Table(
            columns=["Metric", "Q-Learning", "SP (Discrete)", "Improvement"], 
            data=[
                ["Average Time", float(np.mean(q_times)), float(np.mean(sp_times_discrete)), float((np.mean(sp_times_discrete) - np.mean(q_times)) / np.mean(sp_times_discrete) * 100)],
                ["Min Time", float(np.min(q_times)), float(np.min(sp_times_discrete)), float((np.min(sp_times_discrete) - np.min(q_times)) / np.min(sp_times_discrete) * 100)],
                ["Max Time", float(np.max(q_times)), float(np.max(sp_times_discrete)), float((np.max(sp_times_discrete) - np.max(q_times)) / np.max(sp_times_discrete) * 100)],
                ["Std Time", float(np.std(q_times)), float(np.std(sp_times_discrete)), 0.0],
            ]
        )
        wandb.log({"final_evaluation_comparison_discrete": eval_comparison_table_discrete})
        
        # Comparison table: Q-learning vs SP continuous (real-world comparison)
        eval_comparison_table_continuous = wandb.Table(
            columns=["Metric", "Q-Learning", "SP (Continuous)", "Improvement"], 
            data=[
                ["Average Time", float(np.mean(q_times)), float(np.mean(sp_times_continuous)), float((np.mean(sp_times_continuous) - np.mean(q_times)) / np.mean(sp_times_continuous) * 100)],
                ["Min Time", float(np.min(q_times)), float(np.min(sp_times_continuous)), float((np.min(sp_times_continuous) - np.min(q_times)) / np.min(sp_times_continuous) * 100)],
                ["Max Time", float(np.max(q_times)), float(np.max(sp_times_continuous)), float((np.max(sp_times_continuous) - np.max(q_times)) / np.max(sp_times_continuous) * 100)],
                ["Std Time", float(np.std(q_times)), float(np.std(sp_times_continuous)), 0.0],
            ]
        )
        wandb.log({"final_evaluation_comparison_continuous": eval_comparison_table_continuous})
    
    wandb.finish()
    print("\n" + "="*60)
    print("Q-learning training completed!")
    print("="*60)

