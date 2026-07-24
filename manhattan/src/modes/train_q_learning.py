import numpy as np
import jax
import jax.numpy as jnp
import optax
import wandb
import os

from datetime import datetime

from taxi_env import TaxiState, init_env
from training.train_q_learning import train_q_learning, evaluate_q_agent
from training.q_learning import TabularQLearning
from utils import offline_shortest_path_action, offline_shortest_path_action_discrete, offline_shortest_path_action_batch, offline_shortest_path_action_discrete_batch
# from evaluation.plot_agent_paths import plot_rl_vs_shortest_path  # Disabled: trajectory plots removed

from .context import RunContext


def run_q_learning(args, ctx: RunContext) -> None:
    """Run tabular Q-learning training when discrete=True."""
    print("="*60)
    print("Running Tabular Q-Learning (Discrete Mode)")
    print("="*60)
    
    # Generate evaluation sets for periodic evaluations during training: 3 sets of num_agents starts and num_agents pickups
    eval_starts_list = ctx.env.fixed_starts[:min(3, len(ctx.env.fixed_starts))].tolist() if hasattr(ctx.env.fixed_starts, 'tolist') else list(ctx.env.fixed_starts[:min(3, len(ctx.env.fixed_starts))])
    eval_pickups_list = ctx.env.fixed_pickups[:min(3, len(ctx.env.fixed_pickups))].tolist() if hasattr(ctx.env.fixed_pickups, 'tolist') else list(ctx.env.fixed_pickups[:min(3, len(ctx.env.fixed_pickups))])
    
    eval_starts_combined = []
    eval_pickups_combined = []
    for start in eval_starts_list:
        for pickup in eval_pickups_list:
            eval_starts_combined.append(int(start))
            eval_pickups_combined.append(int(pickup))
    
    eval_starts = jnp.array(eval_starts_combined, dtype=jnp.int32)
    eval_pickups = jnp.array(eval_pickups_combined, dtype=jnp.int32)
    
    print(f"Periodic evaluations during training will use {len(eval_starts)} combinations")
    print(f"Final evaluation will use {args.num_agents} agents with Hungarian matching (5×5 set combinations)")

    fixed_starts_trace = np.array(ctx.env.fixed_starts).astype(int).tolist()
    fixed_pickups_trace = np.array(ctx.env.fixed_pickups).astype(int).tolist()

    # Fast path: run only shortest-path baselines (continuous + discrete) and exit
    if getattr(args, "eval_only_sp", False):
        from jax import random as jax_random
        
        num_agents = args.num_agents
        num_iterations = 500
        
        print(f"\n=== Shortest Path Evaluation with {num_agents} agents ===")
        print(f"Running {num_iterations} iterations...")
        
        def evaluate_sp_continuous(env, starts, pickups, max_steps):
            sp_times, sp_rewards, sp_steps, sp_completed = [], [], [], []
            noise_key = jax_random.PRNGKey(42)
            for start, pickup in zip(starts, pickups):
                state = init_env(jax.random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
                total_time = 0.0
                total_reward = 0.0
                step_count = 0
                while not state.done and step_count < max_steps:
                    action = offline_shortest_path_action(
                        state.current_node, state.pickup_node,
                        env.adj_list, env.travel_times, env.distances,
                        env.neighbor_mask_static[state.current_node]
                    )
                    noise_key, nk = jax_random.split(noise_key)
                    state, reward, done, info = env.step(state, action, noise_key=nk)
                    total_time += float(info["travel"] + info["wait"])
                    total_reward += float(reward)
                    step_count += 1
                    if done:
                        break
                sp_times.append(total_time)
                sp_rewards.append(total_reward)
                sp_steps.append(step_count)
                sp_completed.append(bool(state.done))
            return {
                "avg_time": float(np.mean(sp_times)),
                "avg_reward": float(np.mean(sp_rewards)),
                "avg_steps": float(np.mean(sp_steps)),
                "completion_rate": float(np.mean(sp_completed)),
            }

        def evaluate_sp_discrete(env, starts, pickups, gamma, max_steps, dt):
            sp_times, sp_rewards, sp_steps, sp_completed = [], [], [], []
            # Use an untrained Q-agent solely for discretized stepping
            q_agent = TabularQLearning(
                env=env,
                dt=dt,
                learning_rate=0.1,
                discount_factor=gamma,
                epsilon_start=0.0,
                epsilon_end=0.0,
                epsilon_decay_steps=1,
                initial_q_value=0.0,
            )
            for idx, (start, pickup) in enumerate(zip(starts, pickups)):
                key = jax.random.PRNGKey(idx)
                state = init_env(key, start, pickup, env.neighbor_mask_static)[0]
                total_time = 0.0
                total_reward = 0.0
                step_count = 0
                while not state.done and step_count < max_steps:
                    action = offline_shortest_path_action_discrete(
                        state.current_node, state.pickup_node,
                        env.adj_list, q_agent.discrete_travel_times, env.distances,
                        env.neighbor_mask_static[state.current_node]
                    )
                    key, step_key, noise_key = jax.random.split(key, 3)
                    state, reward, done, info = q_agent.step_with_discretization(state, action, step_key, noise_key=noise_key)
                    total_time += float(info["travel"] + info["wait"])
                    total_reward += float(reward)
                    step_count += 1
                    if done:
                        break
                sp_times.append(total_time)
                sp_rewards.append(total_reward)
                sp_steps.append(step_count)
                sp_completed.append(bool(state.done))
            return {
                "avg_time": float(np.mean(sp_times)),
                "avg_reward": float(np.mean(sp_rewards)),
                "avg_steps": float(np.mean(sp_steps)),
                "completion_rate": float(np.mean(sp_completed)),
            }

        # Accumulate results across iterations
        all_cont_times, all_cont_rewards, all_cont_steps, all_cont_completions = [], [], [], []
        all_disc_times, all_disc_rewards, all_disc_steps, all_disc_completions = [], [], [], []
        
        dt = getattr(args, 'dt', 1.0)
        base_key = jax_random.PRNGKey(args.seed)
        
        for iteration in range(num_iterations):
            # Sample starts and pickups for this iteration
            key, subkey = jax_random.split(base_key)
            base_key = subkey
            all_keys = jax_random.split(subkey, 2 * num_agents)
            start_keys, pickup_keys = all_keys[:num_agents], all_keys[num_agents:]
            
            eval_starts = jnp.array([jax_random.choice(k, ctx.env.fixed_starts) for k in start_keys])
            eval_pickups = jnp.array([jax_random.choice(k, ctx.env.fixed_pickups) for k in pickup_keys])
            
            # Use Hungarian matching to assign pickups to starts
            cost_matrix = ctx.distances[eval_starts, :][:, eval_pickups]
            _, assignment = optax.assignment.hungarian_algorithm(cost_matrix)
            matched_pickups = eval_pickups[assignment]
            
            # Evaluate for this iteration
            sp_cont = evaluate_sp_continuous(ctx.env, eval_starts, matched_pickups, 2 * ctx.max_length)
            sp_disc = evaluate_sp_discrete(ctx.env, eval_starts, matched_pickups, args.gamma, 2 * ctx.max_length, dt)
            
            # Accumulate results (averages per iteration)
            all_cont_times.append(sp_cont['avg_time'])
            all_cont_rewards.append(sp_cont['avg_reward'])
            all_cont_steps.append(sp_cont['avg_steps'])
            all_cont_completions.append(sp_cont['completion_rate'])
            
            all_disc_times.append(sp_disc['avg_time'])
            all_disc_rewards.append(sp_disc['avg_reward'])
            all_disc_steps.append(sp_disc['avg_steps'])
            all_disc_completions.append(sp_disc['completion_rate'])
            
            if (iteration + 1) % 10 == 0:
                print(f"  Completed {iteration + 1}/{num_iterations} iterations...")
        
        # Compute final statistics across all iterations
        final_cont = {
            "avg_time": float(np.mean(all_cont_times)),
            "std_time": float(np.std(all_cont_times)),
            "avg_reward": float(np.mean(all_cont_rewards)),
            "std_reward": float(np.std(all_cont_rewards)),
            "avg_steps": float(np.mean(all_cont_steps)),
            "std_steps": float(np.std(all_cont_steps)),
            "completion_rate": float(np.mean(all_cont_completions)),
        }
        
        final_disc = {
            "avg_time": float(np.mean(all_disc_times)),
            "std_time": float(np.std(all_disc_times)),
            "avg_reward": float(np.mean(all_disc_rewards)),
            "std_reward": float(np.std(all_disc_rewards)),
            "avg_steps": float(np.mean(all_disc_steps)),
            "std_steps": float(np.std(all_disc_steps)),
            "completion_rate": float(np.mean(all_disc_completions)),
        }

        print("\n=== Shortest Path Baseline (Continuous - real time) ===")
        print(f"  Avg reward: {final_cont['avg_reward']:.2f} ± {final_cont['std_reward']:.2f}")
        print(f"  Avg steps: {final_cont['avg_steps']:.1f} ± {final_cont['std_steps']:.1f}")
        print(f"  Completion rate: {final_cont['completion_rate']:.2%}")
        print(f"  Avg time: {final_cont['avg_time']:.2f} ± {final_cont['std_time']:.2f}")

        print(f"\n=== Shortest Path Baseline (Discrete - dt={dt}, Q-learning fair) ===")
        print(f"  Avg reward: {final_disc['avg_reward']:.2f} ± {final_disc['std_reward']:.2f}")
        print(f"  Avg steps: {final_disc['avg_steps']:.1f} ± {final_disc['std_steps']:.1f}")
        print(f"  Completion rate: {final_disc['completion_rate']:.2%}")
        print(f"  Avg time: {final_disc['avg_time']:.2f} ± {final_disc['std_time']:.2f}")
        return
    
    # Initialize wandb
    wandb_project = os.environ.get(
        "WANDB_PROJECT",
        getattr(args, "wandb_project", "ride-sharing-q-learning"),
    )
    wandb.init(
        project=wandb_project,
        name=f"qlearning-{args.env_type}-{args.epochs}episodes-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "all_cli_args": dict(vars(args)),
            "wandb_project": wandb_project,
            "env_type": args.env_type,
            "num_layers": args.num_layers,
            "layer_width": args.layer_width,
            "offset": args.offset,
            "random_offsets": getattr(args, 'random_offsets', False),
            "traffic_offset_seed": 42,
            "cycle_length": args.cycle_length,
            "no_congestion": args.no_congestion,
            "noise": getattr(args, 'noise', False),
            "noise_level": getattr(args, 'noise_level', 0.0),
            "place_name": args.place_name,
            "zone_shp": getattr(args, 'zone_shp', None),
            "manhattan_area": getattr(args, 'manhattan_area', None),
            "model": "q_learning",
            "discrete": True,
            "dt": getattr(args, 'dt', 1.0),
            "num_episodes": args.epochs,
            "num_agents": args.num_agents,
            "gamma": args.gamma,
            "seed": args.seed,
            "num_nodes": ctx.env.num_nodes,
            "max_deg": ctx.env.max_deg,
            "max_steps": ctx.env.max_steps,
            "max_training_steps_per_episode": 2 * ctx.max_length,
            "pickup_bonus": ctx.env.pickup_bonus,
            "timeout_penalty": ctx.env.timeout_penalty,
            "sample_starts_from_three_fixed": getattr(args, 'sample_starts_from_three_fixed', False),
            "sample_pickups_from_three_fixed": getattr(args, 'sample_pickups_from_three_fixed', False),
            "three_fixed_selection_method": getattr(args, 'three_fixed_selection_method', None),
            "fixed_starts_count": len(fixed_starts_trace),
            "fixed_pickups_count": len(fixed_pickups_trace),
            "fixed_starts": fixed_starts_trace,
            "fixed_pickups_preview": fixed_pickups_trace[:50],
            "fixed_pickups_is_truncated": len(fixed_pickups_trace) > 50,
            "eval_starts": eval_starts_list,
            "eval_pickups": eval_pickups_list,
            "eval_combinations": len(eval_starts),
            "eval_frequency": getattr(args, 'eval_frequency', 100),
            "timestamp": datetime.now().isoformat(),
            "pretrain_enabled": getattr(args, 'pretrain_enabled', False),
            "num_pretrain_episodes": getattr(args, 'num_pretrain_episodes', 1000),
            "pretrain_learning_rate": getattr(args, 'pretrain_learning_rate', None),
            "init_from_shortest_paths": getattr(args, 'init_from_shortest_paths', False),
            "init_all_time_slices": getattr(args, 'init_all_time_slices', False),
            "init_q_table_path": getattr(args, 'init_q_table_path', None),
            "q_table_path": getattr(args, 'q_table_path', None),
            "q_table_dtype": getattr(args, 'q_table_dtype', 'float32'),
            "learning_rate": 0.1,
            "initial_q_value": 0.0,
            "no_round_trip": getattr(args, 'no_round_trip', False),
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
    
    total_training_steps = args.epochs * 2 * ctx.max_length
    epsilon_decay_steps = int(total_training_steps * epsilon_decay_fraction) if total_training_steps > 0 else 0
    
    if total_training_steps > 0:
        print(f"\nEpsilon Decay Schedule:")
        print(f"  Total training steps: {total_training_steps:,}")
        print(f"  Epsilon decay steps: {epsilon_decay_steps:,}")
        print(f"  Epsilon will decay from {epsilon_start} to {epsilon_end} over {epsilon_decay_steps:,} steps")
        print(f"  (This is {epsilon_decay_steps / total_training_steps * 100:.1f}% of total training)\n")
    else:
        print(f"\nEvaluation-only mode (epochs=0):")
        print(f"  Epsilon schedule not applicable (no training steps)\n")
    
    # Check if pretraining is enabled
    pretrain_enabled = getattr(args, 'pretrain_enabled', False)
    num_pretrain_episodes = getattr(args, 'num_pretrain_episodes', 1000)
    pretrain_learning_rate = getattr(args, 'pretrain_learning_rate', None)
    
    # Check if shortest path initialization is enabled
    init_from_shortest_paths = getattr(args, 'init_from_shortest_paths', False)
    init_all_time_slices = getattr(args, 'init_all_time_slices', False)
    
    # If initialization is used and pretraining is enabled, use a lower learning rate for pretraining
    # to avoid overwriting good initialization values
    if init_from_shortest_paths and pretrain_enabled and pretrain_learning_rate is None:
        pretrain_learning_rate = 0.01  # Lower learning rate to preserve initialization
        print(f"\n⚠️  Using lower pretraining learning rate ({pretrain_learning_rate}) to preserve initialization values")
        print(f"   (You can override with --pretrain_learning_rate)\n")
    
    # Create logging function for pretraining metrics
    def pretrain_log_fn(log_dict, step):
        """Log pretraining metrics to wandb"""
        if wandb.run is not None:
            # Ensure all values are Python native types for WandB
            log_dict_clean = {}
            for key, value in log_dict.items():
                if isinstance(value, (jnp.ndarray, np.ndarray)):
                    log_dict_clean[key] = float(value) if value.size == 1 else value.tolist()
                elif isinstance(value, (jnp.integer, np.integer)):
                    log_dict_clean[key] = int(value)
                elif isinstance(value, (jnp.floating, np.floating)):
                    log_dict_clean[key] = float(value)
                else:
                    log_dict_clean[key] = value
            wandb.log(log_dict_clean, step=int(step), commit=True)
    
    initial_q_value = 0.0
    
    q_agent = train_q_learning(
        env=ctx.env,
        fixed_starts=ctx.env.fixed_starts,
        fixed_pickups=ctx.env.fixed_pickups,
        num_episodes=args.epochs,
        num_agents=args.num_agents,
        max_steps_per_episode=2 * ctx.max_length,
        dt=getattr(args, 'dt', 1.0),
        learning_rate=0.1,
        discount_factor=args.gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        eval_frequency=getattr(args, 'eval_frequency', 100),
        eval_starts=eval_starts,
        eval_pickups=eval_pickups,
        seed=args.seed,
        pretrain_enabled=pretrain_enabled,
        num_pretrain_episodes=num_pretrain_episodes,
        pretrain_log_fn=pretrain_log_fn if pretrain_enabled else None,
        save_path=args.q_table_path,
        load_path=args.q_table_path,
        initial_q_value=initial_q_value,
        init_from_shortest_paths=init_from_shortest_paths,
        init_all_time_slices=init_all_time_slices,
        pretrain_learning_rate=pretrain_learning_rate,
        init_q_table_path=getattr(args, 'init_q_table_path', None),
        no_round_trip=getattr(args, 'no_round_trip', False),
        q_table_dtype=getattr(args, 'q_table_dtype', 'float32'),
    )
    
    # Final evaluation with matching for multiple agents
    print(f"\n{'='*60}")
    print("Final Evaluation (with Hungarian matching)")
    print(f"{'='*60}")
    
    B = args.num_agents
    num_start_sets = 5
    num_pickup_sets = 5
    
    # Sample 5 sets of B starts and 5 sets of B pickups
    final_eval_key = jax.random.PRNGKey(args.seed + 9999)
    all_start_sets = []
    all_pickup_sets = []
    
    for i in range(num_start_sets):
        key, final_eval_key = jax.random.split(final_eval_key)
        start_keys = jax.random.split(key, B)
        start_set = jnp.array([jax.random.choice(k, ctx.env.fixed_starts) for k in start_keys])
        all_start_sets.append(start_set)
    
    for i in range(num_pickup_sets):
        key, final_eval_key = jax.random.split(final_eval_key)
        pickup_keys = jax.random.split(key, B)
        pickup_set = jnp.array([jax.random.choice(k, ctx.env.fixed_pickups) for k in pickup_keys])
        all_pickup_sets.append(pickup_set)
    
    print(f"Sampled {num_start_sets} sets of {B} starts and {num_pickup_sets} sets of {B} pickups")
    print(f"Evaluating {num_start_sets} × {num_pickup_sets} = {num_start_sets * num_pickup_sets} combinations with Hungarian matching (using Q-table estimation, same as training)")
    
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
    
    # Helper functions for matching
    def simulate_policy_time(policy_fn, start, pickup, base_seed, use_discrete):
        """Simulate a single start/pickup pair and return total travel time (seconds)."""
        key = jax.random.PRNGKey(base_seed)
        state = init_env(key, int(start), int(pickup), ctx.env.neighbor_mask_static)[0]
        total_time = 0.0
        steps = 0
        
        while (not bool(state.done)) and steps < 2 * ctx.max_length:
            action = policy_fn(state)
            key, step_key, noise_key = jax.random.split(key, 3)
            if use_discrete:
                next_state, reward, done, info = q_agent.step_with_discretization(
                    state, action, step_key, noise_key=noise_key
                )
            else:
                next_state, reward, done, info = ctx.env.step(state, action, noise_key=noise_key)
            
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
        """Assign pickups to starts using Hungarian matching on Q-table estimated returns (same as training)."""
        if len(starts) == 0:
            return pickups, jnp.zeros((0, 0), dtype=jnp.float32)
        
        from training.q_learning import _estimate_returns_batch_q_table_direct
        
        starts_jax = jnp.array(starts, dtype=jnp.int32)
        pickups_jax = jnp.array(pickups, dtype=jnp.int32)
        num_starts = len(starts)
        num_pickups = len(pickups)
        
        # Create all combinations
        starts_expanded = jnp.repeat(starts_jax, num_pickups)
        pickups_expanded = jnp.tile(pickups_jax, num_starts)
        
        time_idx = jnp.int32(0)
        
        # Estimate returns directly from Q-table (batched, fast!)
        returns_flat = _estimate_returns_batch_q_table_direct(
            q_agent.q_table,
            starts_expanded,
            pickups_expanded,
            time_idx,
            ctx.env.num_nodes,
            q_agent.max_time_slices,
            ctx.env,
        )
        
        # Reshape to [num_starts, num_pickups] and convert to cost (negative return)
        returns_matrix = returns_flat.reshape(num_starts, num_pickups)
        cost_matrix = -returns_matrix  # Negative because Hungarian minimizes cost
        
        # Hungarian algorithm
        _, assignment = optax.assignment.hungarian_algorithm(cost_matrix)
        matched_pickups = jnp.take(pickups_jax, assignment, axis=0)
        return matched_pickups, cost_matrix
    
    # Collect all start-pickup pairs first (after matching)
    all_eval_starts = []
    all_eval_pickups = []
    all_eval_seeds = []
    
    for start_set_idx, start_set in enumerate(all_start_sets):
        for pickup_set_idx, pickup_set in enumerate(all_pickup_sets):
            base_seed = args.seed + start_set_idx * 1000 + pickup_set_idx * 100
            
            # Perform Hungarian matching for Q-learning
            matched_pickups, _ = match_pickups(
                start_set, pickup_set, q_learning_policy, base_seed, use_discrete=True
            )
            
            # Collect pairs for batched evaluation
            for start, pickup in zip(start_set, matched_pickups):
                all_eval_starts.append(int(start))
                all_eval_pickups.append(int(pickup))
                all_eval_seeds.append(base_seed)
    
    eval_starts_arr = jnp.array(all_eval_starts, dtype=jnp.int32)
    eval_pickups_arr = jnp.array(all_eval_pickups, dtype=jnp.int32)
    eval_seeds_arr = jnp.array(all_eval_seeds, dtype=jnp.uint32)
    
    # Create batched, JIT-compiled evaluation function
    @jax.jit
    def evaluate_batch_episodes(starts, pickups, seeds, q_table, dt, max_time_slices, num_nodes, env, max_steps):
        """Batched, JIT-compiled evaluation of multiple episodes in parallel."""
        num_episodes = starts.shape[0]
        
        # Initialize states for all episodes using seeds
        keys = jax.vmap(lambda s: jax.random.PRNGKey(s))(seeds)
        batch_init = jax.vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0))
        states, _ = batch_init(keys, starts, pickups)
        
        # Initialize tracking arrays
        episode_rewards = jnp.zeros(num_episodes, dtype=jnp.float32)
        episode_steps = jnp.zeros(num_episodes, dtype=jnp.int32)
        episode_dones = jnp.zeros(num_episodes, dtype=jnp.bool_)
        
        max_eval_steps = jnp.int32(max_steps)
        
        def cond_fn(carry):
            step, _, _, _, completed_mask = carry
            all_done = jnp.logical_or(jnp.all(completed_mask), step >= max_eval_steps)
            return jnp.logical_not(all_done)
        
        def body_fn(carry):
            step, states, rewards_acc, steps_acc, completed_mask = carry
            
            # Discretize state time for Q-table lookup (greedy policy, epsilon=0)
            discretized_times = dt * jnp.round(states.time / dt)
            
            # Get state indices for Q-table lookup
            curr = jnp.clip(states.current_node, 0, num_nodes - 1)
            pickup = jnp.clip(states.pickup_node, 0, num_nodes - 1)
            t_idx = jnp.clip(jnp.int32(discretized_times / dt), 0, max_time_slices - 1)
            
            # Get Q-values for all actions: [num_episodes, max_deg]
            q_values = q_table[curr, pickup, t_idx, :]  # [num_episodes, max_deg]
            
            # Mask invalid actions
            valid_mask = states.neighbor_mask  # [num_episodes, max_deg]
            q_masked = jnp.where(valid_mask, q_values, -jnp.inf)
            
            # Greedy action (epsilon=0)
            actions = jnp.argmax(q_masked, axis=-1)  # [num_episodes]
            
            # Take step in continuous environment
            batch_step = jax.vmap(env.step, in_axes=(0, 0))
            next_states, rewards, terminals, _ = batch_step(states, actions)
            
            # Update accumulators only for active episodes
            active = ~completed_mask
            rewards_acc = rewards_acc + rewards * active
            steps_acc = steps_acc + active.astype(jnp.int32)
            completed_mask = completed_mask | terminals
            
            return (
                step + jnp.int32(1),
                next_states,
                rewards_acc,
                steps_acc,
                completed_mask,
            )
        
        init_carry = (
            jnp.int32(0),
            states,
            episode_rewards,
            episode_steps,
            episode_dones,
        )
        
        _, final_states, total_rewards, total_steps, completed = jax.lax.while_loop(
            cond_fn, body_fn, init_carry
        )
        
        return total_rewards, total_steps, completed
    
    # Run batched evaluation
    print(f"Running batched evaluation for {len(all_eval_starts)} episodes...")
    all_rewards, all_steps, all_completions = evaluate_batch_episodes(
        eval_starts_arr,
        eval_pickups_arr,
        eval_seeds_arr,
        q_agent.q_table,
        q_agent.dt,
        q_agent.max_time_slices,
        ctx.env.num_nodes,
        ctx.env,
        2 * ctx.max_length
    )
    
    # Convert to numpy for final processing
    all_rewards = np.array(all_rewards)
    all_steps = np.array(all_steps)
    all_completions = np.array(all_completions)
    
    final_eval = {
        'avg_reward': np.mean(all_rewards),
        'avg_steps': np.mean(all_steps),
        'completion_rate': np.mean(all_completions),
        'rewards': all_rewards,
        'steps': all_steps,
        'completions': all_completions,
    }
    
    # Old learned-Q (assign-once) reporting: suppressed when the four-way
    # reassignment comparison is active (result #1 there supersedes it).
    if not getattr(args, "eval_reassignment_baselines", False):
        print(f"\nFinal Evaluation Results:")
        print(f"  Average Reward: {final_eval['avg_reward']:.2f}")
        print(f"  Average Steps: {final_eval['avg_steps']:.1f}")
        print(f"  Completion Rate: {final_eval['completion_rate']:.2%}")
        print(f"  Total evaluations: {len(all_rewards)} (25 set combinations × {B} agents)")

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
        noise_key = jax.random.PRNGKey(42)
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
                noise_key, nk = jax.random.split(noise_key)
                state, reward, done, info = ctx.env.step(state, action, noise_key=nk)
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
    
    # Shortest path baseline - DISCRETE
    def evaluate_shortest_path_baseline_discrete(starts, pickups):
        """Shortest path in discrete time - fair comparison with Q-learning."""
        sp_times, sp_rewards, sp_steps, sp_completed = [], [], [], []
        noise_key = jax.random.PRNGKey(99)
        for start, pickup in zip(starts, pickups):
            total_time = 0.0
            total_reward = 0.0
            step_count = 0
            state = init_env(jax.random.PRNGKey(0), start, pickup, ctx.env.neighbor_mask_static)[0]
            while not state.done and step_count < ctx.env.max_steps:
                action = offline_shortest_path_action_discrete(
                    state.current_node, state.pickup_node,
                    ctx.env.adj_list, q_agent.discrete_travel_times, ctx.env.distances,
                    ctx.env.neighbor_mask_static[state.current_node]
                )
                # Use discretized step
                noise_key, nk = jax.random.split(noise_key)
                state, reward, done, info = q_agent.step_with_discretization(
                    state, action, jax.random.PRNGKey(step_count), noise_key=nk
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
    
    # Shortest path policy (uses continuous travel times)
    def sp_policy(state: TaxiState) -> int:
        return offline_shortest_path_action(
            state.current_node, state.pickup_node,
            ctx.env.adj_list, ctx.env.travel_times, ctx.env.distances,
            ctx.env.neighbor_mask_static[state.current_node]
        )
    
    # Shortest path policy (uses discretized travel times with epsilon for zero-time edges)
    def sp_policy_discrete(state: TaxiState) -> int:
        return offline_shortest_path_action_discrete(
            state.current_node, state.pickup_node,
            ctx.env.adj_list, q_agent.discrete_travel_times, ctx.env.distances,
            ctx.env.neighbor_mask_static[state.current_node]
        )
    
    @jax.jit
    def build_cost_matrix_from_distances(starts, pickups, distances):
        """Fast: Build cost matrix using precomputed distance lookups."""
        # Use advanced indexing: distances[start[i], pickup[j]] for all i, j
        # This creates a [len(starts), len(pickups)] matrix
        starts_jax = jnp.array(starts, dtype=jnp.int32)
        pickups_jax = jnp.array(pickups, dtype=jnp.int32)
        cost_matrix = distances[jnp.ix_(starts_jax, pickups_jax)]  # Advanced indexing
        return cost_matrix
    
    def match_pickups(starts, pickups, policy_fn, base_seed, use_discrete):
        """
        Fast: Assign pickups to starts using Hungarian matching on precomputed distances.
        Uses offline distance matrix lookup instead of slow simulation.
        """
        if len(starts) == 0:
            return pickups, jnp.zeros((0, 0), dtype=jnp.float32)
        
        # Fast: Just use precomputed distance matrix (no simulation needed!)
        cost_matrix = build_cost_matrix_from_distances(starts, pickups, ctx.env.distances)
        _, assignment = optax.assignment.hungarian_algorithm(cost_matrix)
        matched_pickups = jnp.take(jnp.array(pickups, dtype=jnp.int32), assignment, axis=0)
        return matched_pickups, cost_matrix
    
    # Batch operations for evaluation (with per-step noise when noise_mask > 0)
    batch_step_continuous = jax.jit(jax.vmap(
        lambda s, a, nk: ctx.env.step(s, a, noise_key=nk),
        in_axes=(0, 0, 0),
    ))
    # Use JIT-compatible function directly (not the wrapper that calls float())
    from training.q_learning import _jitted_step_with_discretization
    batch_step_discrete_fn = jax.vmap(
        lambda s, a, nk: _jitted_step_with_discretization(ctx.env, s, a, q_agent.discrete_travel_times, q_agent.dt, noise_key=nk),
        in_axes=(0, 0, 0)
    )
    batch_step_discrete = jax.jit(batch_step_discrete_fn)
    batch_reset = jax.jit(jax.vmap(lambda k, s, p: init_env(k, s, p, ctx.env.neighbor_mask_static), in_axes=(0, 0, 0)))
    
    def evaluate_q_learning_batched_impl(q_table, starts, pickups, eval_key, dt, max_time_slices, num_nodes, max_steps, use_continuous_eval):
        """Evaluation of Q-learning policy with path tracking."""
        num_eval = starts.shape[0]
        eval_keys = jax.random.split(eval_key, num_eval)
        eval_states, _ = batch_reset(eval_keys, starts, pickups)
        
        total_times = jnp.zeros(num_eval, dtype=jnp.float32)
        total_rewards = jnp.zeros(num_eval, dtype=jnp.float32)
        completed = jnp.zeros(num_eval, dtype=bool)
        max_eval_steps = max_steps  # Now static, so can use directly
        
        # Track paths, step rewards, and step times (fixed-size buffers)
        path_buffer = jnp.zeros((num_eval, max_eval_steps + 1), dtype=jnp.int32)
        step_rewards_buffer = jnp.zeros((num_eval, max_eval_steps), dtype=jnp.float32)
        step_times_buffer = jnp.zeros((num_eval, max_eval_steps), dtype=jnp.float32)
        path_lengths = jnp.zeros(num_eval, dtype=jnp.int32)
        path_buffer = path_buffer.at[:, 0].set(eval_states.current_node)
        path_lengths = path_lengths + 1
        
        def cond_fn(carry):
            step, _, _, _, _, completed_mask, _, _, _, _, _ = carry
            max_steps_jax = jnp.int32(max_eval_steps)  # Convert static int to JAX int32 for comparison
            all_done = jnp.logical_or(jnp.all(completed_mask), step >= max_steps_jax)
            return jnp.logical_not(all_done)
        
        def body_fn(carry):
            step, states, times_acc, rewards_acc, completed_mask, path_buf, step_rewards_buf, step_times_buf, path_lens, step_keys, _ = carry
            
            # Discretize state time for Q-table lookup
            discretized_times = dt * jnp.round(states.time / dt)
            curr = jnp.clip(states.current_node, 0, num_nodes - 1)
            pickup = jnp.clip(states.pickup_node, 0, num_nodes - 1)
            t_idx = jnp.clip(jnp.int32(discretized_times / dt), 0, max_time_slices - 1)
            
            # Get Q-values and select greedy action
            q_values = q_table[curr, pickup, t_idx, :]  # [num_eval, max_deg]
            valid_mask = states.neighbor_mask
            q_masked = jnp.where(valid_mask, q_values, -jnp.inf)
            actions = jnp.argmax(q_masked, axis=-1)
            
            # Take step (continuous or discrete), with per-step noise keys
            step_keys, noise_subkey = jax.random.split(step_keys)
            noise_keys = jax.random.split(noise_subkey, num_eval)
            if use_continuous_eval:
                next_states, rewards, terminals, info = batch_step_continuous(states, actions, noise_keys)
            else:
                next_states, rewards, terminals, info = batch_step_discrete(states, actions, noise_keys)
            
            # Accumulate times and rewards
            travel_times = info['travel'] + info['wait']
            active = ~completed_mask
            times_acc = times_acc + travel_times * active
            rewards_acc = rewards_acc + rewards * active
            completed_mask = completed_mask | terminals
            
            # Track paths, step rewards, and step times
            path_idx = jnp.clip(path_lens, 0, max_eval_steps)
            step_idx = jnp.clip(path_lens - 1, 0, max_eval_steps - 1)
            path_buf = path_buf.at[jnp.arange(num_eval), path_idx].set(next_states.current_node)
            step_rewards_buf = step_rewards_buf.at[jnp.arange(num_eval), step_idx].set(rewards)
            step_times_buf = step_times_buf.at[jnp.arange(num_eval), step_idx].set(travel_times)
            path_lens = path_lens + active.astype(jnp.int32)
            
            return (
                step + jnp.int32(1),
                next_states,
                times_acc,
                rewards_acc,
                completed_mask,
                path_buf,
                step_rewards_buf,
                step_times_buf,
                path_lens,
                step_keys,
                None
            )
        
        init_keys = jax.random.split(eval_key, num_eval)
        init_carry = (
            jnp.int32(0),
            eval_states,
            total_times,
            total_rewards,
            completed,
            path_buffer,
            step_rewards_buffer,
            step_times_buffer,
            path_lengths,
            init_keys[0],  # Dummy key for discrete steps
            None
        )
        _, _, total_times, total_rewards, completed, paths, step_rewards, step_times, path_lengths, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, init_carry
        )
        
        return total_times, total_rewards, completed, paths, path_lengths, step_rewards, step_times
    
    evaluate_q_learning_batched = jax.jit(evaluate_q_learning_batched_impl, static_argnums=(7, 8))

    def evaluate_sp_policy_batched_impl(starts, pickups, eval_key, max_steps, use_discrete):
        """Evaluation of shortest path policy with path tracking."""
        num_eval = starts.shape[0]
        eval_keys = jax.random.split(eval_key, num_eval)
        eval_states, _ = batch_reset(eval_keys, starts, pickups)
        
        total_times = jnp.zeros(num_eval, dtype=jnp.float32)
        total_rewards = jnp.zeros(num_eval, dtype=jnp.float32)
        completed = jnp.zeros(num_eval, dtype=bool)
        max_eval_steps = max_steps
        
        path_buffer = jnp.zeros((num_eval, max_eval_steps + 1), dtype=jnp.int32)
        step_rewards_buffer = jnp.zeros((num_eval, max_eval_steps), dtype=jnp.float32)
        step_times_buffer = jnp.zeros((num_eval, max_eval_steps), dtype=jnp.float32)
        path_lengths = jnp.zeros(num_eval, dtype=jnp.int32)
        path_buffer = path_buffer.at[:, 0].set(eval_states.current_node)
        path_lengths = path_lengths + 1
        
        def cond_fn(carry):
            step, _, _, _, _, completed_mask, _, _, _, _, _ = carry
            max_steps_jax = jnp.int32(max_eval_steps)
            all_done = jnp.logical_or(jnp.all(completed_mask), step >= max_steps_jax)
            return jnp.logical_not(all_done)
        
        def body_fn(carry):
            step, states, times_acc, rewards_acc, completed_mask, path_buf, step_rewards_buf, step_times_buf, path_lens, step_keys, _ = carry
            
            if use_discrete:
                actions = offline_shortest_path_action_discrete_batch(
                    states.current_node,
                    states.pickup_node,
                    ctx.env.adj_list,
                    q_agent.discrete_travel_times,
                    ctx.env.distances,
                    ctx.env.neighbor_mask_static[states.current_node],
                    1e-6  # epsilon
                )
            else:
                actions = offline_shortest_path_action_batch(
                    states.current_node,
                    states.pickup_node,
                    ctx.env.adj_list,
                    ctx.env.travel_times,
                    ctx.env.distances,
                    ctx.env.neighbor_mask_static[states.current_node]
                )
            
            # Take step (continuous or discrete), with per-step noise keys
            step_keys, noise_subkey = jax.random.split(step_keys)
            noise_keys = jax.random.split(noise_subkey, num_eval)
            if use_discrete:
                next_states, rewards, terminals, info = batch_step_discrete(states, actions, noise_keys)
            else:
                next_states, rewards, terminals, info = batch_step_continuous(states, actions, noise_keys)
            
            # Accumulate times and rewards
            travel_times = info['travel'] + info['wait']
            active = ~completed_mask
            times_acc = times_acc + travel_times * active
            rewards_acc = rewards_acc + rewards * active
            completed_mask = completed_mask | terminals
            
            # Track paths, step rewards, and step times
            path_idx = jnp.clip(path_lens, 0, max_eval_steps)
            step_idx = jnp.clip(path_lens - 1, 0, max_eval_steps - 1)
            path_buf = path_buf.at[jnp.arange(num_eval), path_idx].set(next_states.current_node)
            step_rewards_buf = step_rewards_buf.at[jnp.arange(num_eval), step_idx].set(rewards)
            step_times_buf = step_times_buf.at[jnp.arange(num_eval), step_idx].set(travel_times)
            path_lens = path_lens + active.astype(jnp.int32)
            
            return (
                step + jnp.int32(1),
                next_states,
                times_acc,
                rewards_acc,
                completed_mask,
                path_buf,
                step_rewards_buf,
                step_times_buf,
                path_lens,
                step_keys,
                None
            )
        
        init_keys = jax.random.split(eval_key, num_eval)
        init_carry = (
            jnp.int32(0),
            eval_states,
            total_times,
            total_rewards,
            completed,
            path_buffer,
            step_rewards_buffer,
            step_times_buffer,
            path_lengths,
            init_keys[0],
            None
        )
        _, _, total_times, total_rewards, completed, paths, step_rewards, step_times, path_lengths, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, init_carry
        )
        
        return total_times, total_rewards, completed, paths, path_lengths, step_rewards, step_times
    
    # Wrap with jit, making max_steps and use_discrete static
    evaluate_sp_policy_batched = jax.jit(evaluate_sp_policy_batched_impl, static_argnums=(3, 4))

    def evaluate_sp_reassign_batched(starts, pickups, eval_key, max_steps, reassign_every, use_discrete):
        """Shortest-path routing with periodic JOINT reassignment (receding-horizon dispatch).

        Every `reassign_every` timesteps the agent->target matching is re-solved with the
        Hungarian algorithm on offline shortest-path distances between agents' CURRENT nodes
        and the fixed target pool `pickups`; between reassignments each agent follows the
        offline shortest path to its current target. Agents are stepped as a batch; the small
        B x B Hungarian solve runs on host every K steps (K << horizon, B == num_agents).

        A finished agent sits on its target (distance 0), so the Hungarian solution keeps it
        there -- served targets are thus implicitly "consumed" with no special-casing.
        Returns (total_times, total_rewards, completed) as numpy arrays, matching the columns
        used by the other baselines.
        """
        num_eval = int(starts.shape[0])
        eval_keys = jax.random.split(eval_key, num_eval)
        states, _ = batch_reset(eval_keys, starts, pickups)
        pool = jnp.asarray(pickups, dtype=jnp.int32)          # fixed candidate targets (B,)

        total_times = np.zeros(num_eval, dtype=np.float32)
        total_rewards = np.zeros(num_eval, dtype=np.float32)
        completed = np.zeros(num_eval, dtype=bool)
        key = eval_key

        for step in range(int(max_steps)):
            if bool(np.all(completed)):
                break
            # Periodic joint reassignment (step 0 already carries the initial matching).
            if reassign_every > 0 and step > 0 and (step % reassign_every == 0):
                cost = ctx.env.distances[states.current_node][:, pool]   # [B, B] offline SP distances
                _, assignment = optax.assignment.hungarian_algorithm(cost)
                states = states._replace(pickup_node=jnp.take(pool, assignment, axis=0))

            if use_discrete:
                actions = offline_shortest_path_action_discrete_batch(
                    states.current_node, states.pickup_node, ctx.env.adj_list,
                    q_agent.discrete_travel_times, ctx.env.distances,
                    ctx.env.neighbor_mask_static[states.current_node], 1e-6,
                )
            else:
                actions = offline_shortest_path_action_batch(
                    states.current_node, states.pickup_node, ctx.env.adj_list,
                    ctx.env.travel_times, ctx.env.distances,
                    ctx.env.neighbor_mask_static[states.current_node],
                )

            key, sub = jax.random.split(key)
            noise_keys = jax.random.split(sub, num_eval)
            if use_discrete:
                states, rewards, terminals, info = batch_step_discrete(states, actions, noise_keys)
            else:
                states, rewards, terminals, info = batch_step_continuous(states, actions, noise_keys)

            active = ~completed
            total_times += np.asarray(info['travel'] + info['wait'], dtype=np.float32) * active
            total_rewards += np.asarray(rewards, dtype=np.float32) * active
            completed = completed | np.asarray(terminals, dtype=bool)

        return total_times, total_rewards, completed

    # Per-agent-time version of the Q-return estimator (the built-in one broadcasts a
    # single scalar time_idx; mid-episode agents are at different times).
    from training.q_learning import _estimate_return_q_table_direct
    _estimate_returns_time = jax.vmap(
        _estimate_return_q_table_direct,
        in_axes=(None, 0, 0, 0, None, None, None),  # q_table, start, pickup, time_idx(batched), ...
        out_axes=0,
    )

    def evaluate_q_reassign_batched(starts, init_pickups, pool_pickups, eval_key,
                                    max_steps, use_continuous_eval):
        """SALT evaluation with OT assignment RE-SOLVED every timestep.

        Each step: estimate Q-returns (max_a Q) from every agent's CURRENT node and CURRENT
        (discretized) time to every target in `pool_pickups`, solve the Hungarian assignment
        on -returns, reassign each agent's target, then take one greedy learned-Q routing
        step. `init_pickups` is the t=0 OT matching (so reset targets are valid); subsequent
        steps re-solve. Batched over agents; the small B x B Hungarian runs on host each step.
        """
        num_eval = int(starts.shape[0])
        eval_keys = jax.random.split(eval_key, num_eval)
        states, _ = batch_reset(eval_keys, starts, init_pickups)
        pool = jnp.asarray(pool_pickups, dtype=jnp.int32)
        dt = q_agent.dt
        mts = q_agent.max_time_slices
        nn = ctx.env.num_nodes

        total_times = np.zeros(num_eval, dtype=np.float32)
        total_rewards = np.zeros(num_eval, dtype=np.float32)
        completed = np.zeros(num_eval, dtype=bool)
        key = eval_key

        for step in range(int(max_steps)):
            if bool(np.all(completed)):
                break
            # Re-solve OT every timestep (step 0 already carries the t=0 matching).
            if step > 0:
                cur = jnp.clip(states.current_node, 0, nn - 1)
                t_idx = jnp.clip(jnp.int32(jnp.round(states.time / dt)), 0, mts - 1)  # (B,)
                starts_exp = jnp.repeat(cur, num_eval)          # each agent x all targets
                pool_exp = jnp.tile(pool, num_eval)
                t_exp = jnp.repeat(t_idx, num_eval)
                returns = _estimate_returns_time(
                    q_agent.q_table, starts_exp, pool_exp, t_exp, nn, mts, ctx.env
                )
                cost = (-returns).reshape(num_eval, num_eval)   # [B agents, B targets]
                _, assignment = optax.assignment.hungarian_algorithm(cost)
                states = states._replace(pickup_node=jnp.take(pool, assignment, axis=0))

            # Greedy learned-Q routing step (epsilon=0), same lookup as evaluate_batch_episodes.
            curr = jnp.clip(states.current_node, 0, nn - 1)
            pk = jnp.clip(states.pickup_node, 0, nn - 1)
            ti = jnp.clip(jnp.int32(jnp.round(states.time / dt)), 0, mts - 1)
            q_vals = q_agent.q_table[curr, pk, ti, :]
            q_vals = jnp.where(states.neighbor_mask, q_vals, -jnp.inf)
            actions = jnp.argmax(q_vals, axis=-1)

            key, sub = jax.random.split(key)
            noise_keys = jax.random.split(sub, num_eval)
            if use_continuous_eval:
                states, rewards, terminals, info = batch_step_continuous(states, actions, noise_keys)
            else:
                states, rewards, terminals, info = batch_step_discrete(states, actions, noise_keys)

            active = ~completed
            total_times += np.asarray(info['travel'] + info['wait'], dtype=np.float32) * active
            total_rewards += np.asarray(rewards, dtype=np.float32) * active
            completed = completed | np.asarray(terminals, dtype=bool)

        return total_times, total_rewards, completed

    reassignment_periods = []
    if getattr(args, "eval_reassignment_baselines", False):
        reassignment_periods = [
            int(k) for k in str(getattr(args, "reassignment_periods", "5,10")).split(",") if k.strip()
        ]

    # Evaluation loop
    eval_iter = 50
    eval_loop_key = jax.random.PRNGKey(args.seed + 1000)
    max_eval_steps = 2 * ctx.max_length
    
    print("\n" + "="*60)
    print("Starting trajectory evaluation...")
    print(f"Running {eval_iter} iterations")
    print("="*60)
    
    for i in range(eval_iter):
        B = args.num_agents
        eval_loop_key, eval_key = jax.random.split(eval_loop_key)
        eval_keys = jax.random.split(eval_key, 2*B+1)
        start_keys, pickup_keys, base_key = eval_keys[:B], eval_keys[B:2*B], eval_keys[-1]
        eval_starts = jnp.array([jax.random.choice(k, ctx.env.fixed_starts) for k in start_keys])
        eval_pickups = jnp.array([jax.random.choice(k, ctx.env.fixed_pickups) for k in pickup_keys])
        base_seed = int(args.seed + i * 1000)
        
        # Use fast Q-table-based matching for Q-learning (same as training)
        from training.q_learning import _estimate_returns_batch_q_table_direct
        starts_jax = jnp.array(eval_starts, dtype=jnp.int32)
        pickups_jax = jnp.array(eval_pickups, dtype=jnp.int32)
        num_starts = len(eval_starts)
        num_pickups = len(eval_pickups)
        
        # Q-learning matching: use Q-table estimation (fast)
        # Always use time_idx=0 (episodes always start at t=0)
        time_idx = jnp.int32(0)
        
        starts_expanded = jnp.repeat(starts_jax, num_pickups)
        pickups_expanded = jnp.tile(pickups_jax, num_starts)
        returns_flat = _estimate_returns_batch_q_table_direct(
            q_agent.q_table, starts_expanded, pickups_expanded,
            time_idx, ctx.env.num_nodes, q_agent.max_time_slices, ctx.env
        )
        returns_matrix = returns_flat.reshape(num_starts, num_pickups)
        cost_matrix_q = -returns_matrix
        _, assignment_q = optax.assignment.hungarian_algorithm(cost_matrix_q)
        matched_q_pickups = jnp.take(pickups_jax, assignment_q, axis=0)

        # Assignment-ablation baselines (R3-W1): reuse the SAME learned Q routing
        # policy but swap out SALT's optimal-transport assignment. This isolates
        # how much of SALT's benefit comes from the OT layer vs. the routing.
        if getattr(args, "eval_assignment_baselines", False):
            # Independent RNG derived from base_seed so the Q/SP eval streams
            # below are byte-identical whether or not baselines are enabled.
            baseline_key = jax.random.PRNGKey(base_seed + 555)
            # (1) Random assignment: a random permutation of the pickups.
            rand_perm = jax.random.permutation(baseline_key, num_pickups)
            matched_rand_pickups = jnp.take(pickups_jax, rand_perm, axis=0)
            # (2) Myopic nominal-shortest-path assignment: Hungarian on the static
            #     precomputed distance matrix (ignores congestion / time). This is
            #     the cheap "nominal shortest-path assignment" reassessed at t=0.
            dist_cost = build_cost_matrix_from_distances(
                starts_jax, pickups_jax, ctx.env.distances
            )
            _, assignment_myopic = optax.assignment.hungarian_algorithm(dist_cost)
            matched_myopic_pickups = jnp.take(pickups_jax, assignment_myopic, axis=0)

        # Shortest path matching
        matched_sp_pickups_cont, _ = match_pickups(
            eval_starts, eval_pickups, sp_policy, base_seed + 1, use_discrete=False
        )
        matched_sp_pickups_disc, _ = match_pickups(
            eval_starts, eval_pickups, sp_policy_discrete, base_seed + 2, use_discrete=True
        )
        
        eval_loop_key, q_cont_key = jax.random.split(eval_loop_key)
        eval_loop_key, q_disc_key = jax.random.split(eval_loop_key)
        eval_loop_key, sp_cont_key = jax.random.split(eval_loop_key)
        eval_loop_key, sp_disc_key = jax.random.split(eval_loop_key)
        
        print(f"\n Evaluation iteration {i+1}/{eval_iter}")
        
        # Q-learning continuous evaluation
        q_times_continuous, q_rewards_continuous, q_completed_continuous, q_paths_continuous, q_path_lens_continuous, q_step_rewards_continuous, q_step_times_continuous = evaluate_q_learning_batched(
            q_agent.q_table, eval_starts, matched_q_pickups, q_cont_key,
            q_agent.dt, q_agent.max_time_slices, ctx.env.num_nodes, max_eval_steps, use_continuous_eval=True
        )
        
        # Q-learning discrete evaluation
        q_times_discrete, q_rewards_discrete, q_completed_discrete, q_paths_discrete, q_path_lens_discrete, q_step_rewards_discrete, q_step_times_discrete = evaluate_q_learning_batched(
            q_agent.q_table, eval_starts, matched_q_pickups, q_disc_key,
            q_agent.dt, q_agent.max_time_slices, ctx.env.num_nodes, max_eval_steps, use_continuous_eval=False
        )
        
        # SP continuous evaluation
        sp_times_continuous, sp_rewards_continuous, sp_completed_continuous, sp_paths_continuous, sp_path_lens_continuous, sp_step_rewards_continuous, sp_step_times_continuous = evaluate_sp_policy_batched(
            eval_starts, matched_sp_pickups_cont, sp_cont_key, max_eval_steps, use_discrete=False
        )
        
        # SP discrete evaluation
        sp_times_discrete, sp_rewards_discrete, sp_completed_discrete, sp_paths_discrete, sp_path_lens_discrete, sp_step_rewards_discrete, sp_step_times_discrete = evaluate_sp_policy_batched(
            eval_starts, matched_sp_pickups_disc, sp_disc_key, max_eval_steps, use_discrete=True
        )

        # Assignment-ablation baselines: SAME learned routing, different assignment.
        if getattr(args, "eval_assignment_baselines", False):
            rand_eval_key = jax.random.PRNGKey(base_seed + 777)
            myopic_eval_key = jax.random.PRNGKey(base_seed + 888)
            rand_times_continuous = np.array(evaluate_q_learning_batched(
                q_agent.q_table, eval_starts, matched_rand_pickups, rand_eval_key,
                q_agent.dt, q_agent.max_time_slices, ctx.env.num_nodes, max_eval_steps,
                use_continuous_eval=True,
            )[0])
            myopic_times_continuous = np.array(evaluate_q_learning_batched(
                q_agent.q_table, eval_starts, matched_myopic_pickups, myopic_eval_key,
                q_agent.dt, q_agent.max_time_slices, ctx.env.num_nodes, max_eval_steps,
                use_continuous_eval=True,
            )[0])

        # Convert to numpy and extract variable-length paths/step data
        q_times_continuous = np.array(q_times_continuous)
        q_times_discrete = np.array(q_times_discrete)
        sp_times_continuous = np.array(sp_times_continuous)
        sp_times_discrete = np.array(sp_times_discrete)
        q_rewards_continuous = np.array(q_rewards_continuous)
        q_rewards_discrete = np.array(q_rewards_discrete)
        sp_rewards_continuous = np.array(sp_rewards_continuous)
        sp_rewards_discrete = np.array(sp_rewards_discrete)
        
        # Extract paths and step-level data (convert from padded arrays to lists)
        def extract_paths_and_steps(paths_array, path_lens_array, step_rewards_array, step_times_array):
            paths_list = []
            step_rewards_list = []
            step_times_list = []
            for j in range(paths_array.shape[0]):
                path_len = int(path_lens_array[j])
                paths_list.append([int(x) for x in paths_array[j, :path_len]])
                step_rewards_list.append([float(x) for x in step_rewards_array[j, :path_len-1] if path_len > 1])
                step_times_list.append([float(x) for x in step_times_array[j, :path_len-1] if path_len > 1])
            return paths_list, step_rewards_list, step_times_list
        
        q_paths_continuous, q_step_rewards_continuous, q_step_times_continuous = extract_paths_and_steps(
            q_paths_continuous, q_path_lens_continuous, q_step_rewards_continuous, q_step_times_continuous
        )
        q_paths_discrete, q_step_rewards_discrete, q_step_times_discrete = extract_paths_and_steps(
            q_paths_discrete, q_path_lens_discrete, q_step_rewards_discrete, q_step_times_discrete
        )
        sp_paths_continuous, sp_step_rewards_continuous, sp_step_times_continuous = extract_paths_and_steps(
            sp_paths_continuous, sp_path_lens_continuous, sp_step_rewards_continuous, sp_step_times_continuous
        )
        sp_paths_discrete, sp_step_rewards_discrete, sp_step_times_discrete = extract_paths_and_steps(
            sp_paths_discrete, sp_path_lens_discrete, sp_step_rewards_discrete, sp_step_times_discrete
        )
        
        # (Per-agent trajectory printing removed.)

        # Old per-iteration Q/SP time dump: suppressed under the four-way comparison
        # (which prints its own clean summary and logs the four results to wandb).
        if not getattr(args, "eval_reassignment_baselines", False):
            print(f"\nQ-learning avg time (continuous): {q_times_continuous.tolist()}")
            print(f"Q-learning avg time (discrete): {q_times_discrete.tolist()}")
            print(f"SP (continuous) avg time: {sp_times_continuous.tolist()}")
            print(f"SP (discrete) avg time: {sp_times_discrete.tolist()}")

            # Keep scalar means as additional context (new labels to avoid ambiguity).
            print(f"Q-learning mean over agents (continuous): {np.mean(q_times_continuous):.2f}")
            print(f"Q-learning mean over agents (discrete): {np.mean(q_times_discrete):.2f}")
            print(f"SP mean over agents (continuous): {np.mean(sp_times_continuous):.2f}")
            print(f"SP mean over agents (discrete): {np.mean(sp_times_discrete):.2f}")

        if getattr(args, "eval_assignment_baselines", False):
            print(
                "\n--- Assignment ablations (same learned routing, no OT layer) ---"
            )
            print(f"Random-assignment avg time (continuous): {rand_times_continuous.tolist()}")
            print(f"Myopic-nominal-SP-assignment avg time (continuous): {myopic_times_continuous.tolist()}")
            print(f"Random-assignment mean over agents (continuous): {np.mean(rand_times_continuous):.2f}")
            print(f"Myopic-nominal-SP-assignment mean over agents (continuous): {np.mean(myopic_times_continuous):.2f}")
            print(
                "SALT (OT + learned routing) mean over agents (continuous): "
                f"{np.mean(q_times_continuous):.2f}"
            )
            if wandb.run is not None:
                wandb.log({
                    f"final_eval/iter{i}/random_assignment_mean_time_continuous": float(np.mean(rand_times_continuous)),
                    f"final_eval/iter{i}/myopic_nominal_sp_assignment_mean_time_continuous": float(np.mean(myopic_times_continuous)),
                    f"final_eval/iter{i}/salt_ot_mean_time_continuous": float(np.mean(q_times_continuous)),
                })

        # === Four-way final comparison (enabled by --eval_reassignment_baselines) ===
        #   (1) SALT: learned-Q routing + OT assignment RE-SOLVED every timestep
        #   (2) shortest-path, static (assign once at t=0)     [reuse sp_* computed above]
        #   (3,4) shortest-path + reassignment every K (from --reassignment_periods)
        if getattr(args, "eval_reassignment_baselines", False):
            salt_t_cont, _, salt_c_cont = evaluate_q_reassign_batched(
                eval_starts, matched_q_pickups, eval_pickups,
                jax.random.PRNGKey(base_seed + 2500), max_eval_steps, use_continuous_eval=True,
            )
            salt_t_disc, _, salt_c_disc = evaluate_q_reassign_batched(
                eval_starts, matched_q_pickups, eval_pickups,
                jax.random.PRNGKey(base_seed + 2600), max_eval_steps, use_continuous_eval=False,
            )
            print("\n--- Final four-way comparison (mean time over agents) ---")
            print(f"  (1) SALT (learned Q + OT every timestep): cont={np.mean(salt_t_cont):.2f}  disc={np.mean(salt_t_disc):.2f}")
            print(f"  (2) Shortest-path static (assign@t=0):    cont={np.mean(sp_times_continuous):.2f}  disc={np.mean(sp_times_discrete):.2f}")
            log_payload = {
                f"final_eval/iter{i}/salt_reassign_every_step_mean_time_continuous": float(np.mean(salt_t_cont)),
                f"final_eval/iter{i}/salt_reassign_every_step_mean_time_discrete": float(np.mean(salt_t_disc)),
                f"final_eval/iter{i}/salt_reassign_every_step_completion_continuous": float(np.mean(salt_c_cont)),
                f"final_eval/iter{i}/salt_reassign_every_step_completion_discrete": float(np.mean(salt_c_disc)),
                f"final_eval/iter{i}/sp_static_mean_time_continuous": float(np.mean(sp_times_continuous)),
                f"final_eval/iter{i}/sp_static_mean_time_discrete": float(np.mean(sp_times_discrete)),
                f"final_eval/iter{i}/sp_static_completion_continuous": float(np.mean(sp_completed_continuous)),
                f"final_eval/iter{i}/sp_static_completion_discrete": float(np.mean(sp_completed_discrete)),
            }
            for _k in reassignment_periods:
                re_t_cont, _, re_c_cont = evaluate_sp_reassign_batched(
                    eval_starts, matched_sp_pickups_cont, jax.random.PRNGKey(base_seed + 3000 + _k),
                    max_eval_steps, _k, use_discrete=False,
                )
                re_t_disc, _, re_c_disc = evaluate_sp_reassign_batched(
                    eval_starts, matched_sp_pickups_disc, jax.random.PRNGKey(base_seed + 4000 + _k),
                    max_eval_steps, _k, use_discrete=True,
                )
                print(f"  (K={_k}) Shortest-path + reassign every {_k}:   cont={np.mean(re_t_cont):.2f}  disc={np.mean(re_t_disc):.2f}")
                log_payload.update({
                    f"final_eval/iter{i}/sp_reassign{_k}_mean_time_continuous": float(np.mean(re_t_cont)),
                    f"final_eval/iter{i}/sp_reassign{_k}_mean_time_discrete": float(np.mean(re_t_disc)),
                    f"final_eval/iter{i}/sp_reassign{_k}_completion_continuous": float(np.mean(re_c_cont)),
                    f"final_eval/iter{i}/sp_reassign{_k}_completion_discrete": float(np.mean(re_c_disc)),
                })
            if wandb.run is not None:
                wandb.log(log_payload)

    wandb.finish()
    print("\n" + "="*60)
    print("Q-learning training completed!")
    print("="*60)
