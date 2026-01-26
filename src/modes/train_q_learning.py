import numpy as np
import jax
import jax.numpy as jnp
import optax
import wandb

from datetime import datetime

from taxi_env import TaxiState, init_env
from training.train_q_learning import train_q_learning, evaluate_q_agent
from training.q_learning import TabularQLearning
from utils import offline_shortest_path_action, offline_shortest_path_action_discrete
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

    # Fast path: run only shortest-path baselines (continuous + discrete) and exit
    if getattr(args, "eval_only_sp", False):
        from jax import random as jax_random
        
        num_agents = args.num_agents
        num_iterations = 500
        
        print(f"\n=== Shortest Path Evaluation with {num_agents} agents ===")
        print(f"Running {num_iterations} iterations...")
        
        def evaluate_sp_continuous(env, starts, pickups, max_steps):
            sp_times, sp_rewards, sp_steps, sp_completed = [], [], [], []
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
                    state, reward, done, info = env.step(state, action)
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
                    key, step_key = jax.random.split(key)
                    state, reward, done, info = q_agent.step_with_discretization(state, action, step_key)
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
            "dt": getattr(args, 'dt', 1.0),
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
            "init_from_shortest_paths": getattr(args, 'init_from_shortest_paths', False),
            "init_all_time_slices": getattr(args, 'init_all_time_slices', False),
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
    
    # Use optimistic initialization with positive large value to favor exploration
    initial_q_value = 10.0  # Positive large value for optimistic initialization
    
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
        eval_frequency=100,
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
    
    # Create Q-learning greedy policy (needed for matching)
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
    
    # Helper functions for matching (define before use)
    def simulate_policy_time(policy_fn, start, pickup, base_seed, use_discrete):
        """Simulate a single start/pickup pair and return total travel time (seconds)."""
        key = jax.random.PRNGKey(base_seed)
        state = init_env(key, int(start), int(pickup), ctx.env.neighbor_mask_static)[0]
        total_time = 0.0
        steps = 0
        
        while (not bool(state.done)) and steps < 2 * ctx.max_length:
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
        """Assign pickups to starts using Hungarian matching on Q-table estimated returns (same as training)."""
        if len(starts) == 0:
            return pickups, jnp.zeros((0, 0), dtype=jnp.float32)
        
        # Use the same fast Q-table-based estimation as during training (not slow simulations)
        from training.q_learning import _estimate_returns_batch_q_table_direct
        
        starts_jax = jnp.array(starts, dtype=jnp.int32)
        pickups_jax = jnp.array(pickups, dtype=jnp.int32)
        num_starts = len(starts)
        num_pickups = len(pickups)
        
        # Create all combinations (same as training)
        starts_expanded = jnp.repeat(starts_jax, num_pickups)
        pickups_expanded = jnp.tile(pickups_jax, num_starts)
        
        # Estimate returns directly from Q-table (batched, fast!)
        returns_flat = _estimate_returns_batch_q_table_direct(
            q_agent.q_table,
            starts_expanded,
            pickups_expanded,
            ctx.env.num_nodes,
            ctx.env,
        )
        
        # Reshape to [num_starts, num_pickups] and convert to cost (negative return)
        returns_matrix = returns_flat.reshape(num_starts, num_pickups)
        cost_matrix = -returns_matrix  # Negative because Hungarian minimizes cost
        
        # Hungarian algorithm (same as training)
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
    
    # Convert to JAX arrays for batched evaluation
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
    
    print(f"\nFinal Evaluation Results:")
    print(f"  Average Reward: {final_eval['avg_reward']:.2f}")
    print(f"  Average Steps: {final_eval['avg_steps']:.1f}")
    print(f"  Completion Rate: {final_eval['completion_rate']:.2%}")
    print(f"  Total evaluations: {len(all_rewards)} (25 set combinations × {B} agents)")
    
    # # Log final metrics
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
                action = offline_shortest_path_action_discrete(
                    state.current_node, state.pickup_node,
                    ctx.env.adj_list, q_agent.discrete_travel_times, ctx.env.distances,
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
    
    # Shortest path baselines removed - final evaluation now uses matching approach
    # Baselines are still computed in periodic evaluations during training
    
    # Baseline logging commented out - variables no longer exist (replaced with matching approach)
    # wandb.log({
    #     "baseline/sp_continuous_avg_reward": sp_baseline_continuous['avg_reward'],
    #     "baseline/sp_continuous_avg_steps": sp_baseline_continuous['avg_steps'],
    #     "baseline/sp_continuous_completion_rate": sp_baseline_continuous['completion_rate'],
    #     "baseline/sp_discrete_avg_reward": sp_baseline_discrete['avg_reward'],
    #     "baseline/sp_discrete_avg_steps": sp_baseline_discrete['avg_steps'],
    #     "baseline/sp_discrete_completion_rate": sp_baseline_discrete['completion_rate'],
    # })
    
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
    
    def simulate_policy_time(policy_fn, start, pickup, base_seed, use_discrete):
        """Simulate a single start/pickup pair and return total travel time (seconds)."""
        key = jax.random.PRNGKey(base_seed)
        state = init_env(key, int(start), int(pickup), ctx.env.neighbor_mask_static)[0]
        total_time = 0.0
        steps = 0
        
        while (not bool(state.done)) and steps < 2 * ctx.max_length:
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
    eval_iter = 500
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
        
        # Use fast Q-table-based matching for Q-learning (same as training)
        from training.q_learning import _estimate_returns_batch_q_table_direct
        starts_jax = jnp.array(eval_starts, dtype=jnp.int32)
        pickups_jax = jnp.array(eval_pickups, dtype=jnp.int32)
        num_starts = len(eval_starts)
        num_pickups = len(eval_pickups)
        
        # Q-learning matching: use Q-table estimation (fast)
        starts_expanded = jnp.repeat(starts_jax, num_pickups)
        pickups_expanded = jnp.tile(pickups_jax, num_starts)
        returns_flat = _estimate_returns_batch_q_table_direct(
            q_agent.q_table, starts_expanded, pickups_expanded,
            ctx.env.num_nodes, ctx.env
        )
        returns_matrix = returns_flat.reshape(num_starts, num_pickups)
        cost_matrix_q = -returns_matrix
        _, assignment_q = optax.assignment.hungarian_algorithm(cost_matrix_q)
        matched_q_pickups = jnp.take(pickups_jax, assignment_q, axis=0)
        
        # Shortest path matching: use simulation-based (original implementation)
        matched_sp_pickups_cont, _ = match_pickups(
            eval_starts, eval_pickups, sp_policy, base_seed + 1, use_discrete=False
        )
        matched_sp_pickups_disc, _ = match_pickups(
            eval_starts, eval_pickups, sp_policy_discrete, base_seed + 2, use_discrete=True
        )
        
        def discretize_state_time(state: TaxiState, dt: float) -> TaxiState:
            """Create a copy of state with discretized time for Q-table lookup."""
            discretized_time = dt * round(state.time / dt)
            return TaxiState(
                current_node=state.current_node,
                pickup_node=state.pickup_node,
                done=state.done,
                step_count=state.step_count,
                neighbor_mask=state.neighbor_mask,
                time=discretized_time,
            )
        
        def evaluate_policy_single(policy_fn, starts, pickups, use_discrete=False, use_continuous_eval=False):
            """
            Evaluate policy and return times, paths, rewards, step rewards, and step times.
            
            Args:
                policy_fn: Policy function that takes a state and returns an action
                starts: List of start nodes
                pickups: List of pickup nodes
                use_discrete: If True, discretize state time for Q-table lookup (for Q-learning)
                use_continuous_eval: If True, use continuous env.step() even when use_discrete=True
                                     (for Q-learning: train in discrete, eval in continuous)
            """
            times, paths, rewards, step_rewards_list, step_times_list = [], [], [], [], []
            dt = q_agent.dt if hasattr(q_agent, 'dt') else 1.0
            
            for start, pickup in zip(np.array(starts).tolist(), np.array(pickups).tolist()):
                total_time, total_reward = 0.0, 0.0
                step_count = 0
                state = init_env(jax.random.PRNGKey(0), start, pickup, ctx.env.neighbor_mask_static)[0]
                traj = [int(state.current_node)]
                step_rewards = []
                step_times = []
                
                while (not bool(state.done)) and step_count < 2*ctx.max_length:
                    # For Q-learning with continuous evaluation:
                    # Discretize state time for Q-table lookup (action selection)
                    # but use continuous environment for actual step
                    if use_discrete and use_continuous_eval:
                        state_for_policy = discretize_state_time(state, dt)
                        action = policy_fn(state_for_policy)
                        # Use continuous environment step
                        next_state, reward, done, info = ctx.env.step(state, action)
                    elif use_discrete:
                        # Original behavior: use discretized step for Q-learning
                        state_for_policy = discretize_state_time(state, dt)
                        action = policy_fn(state_for_policy)
                        next_state, reward, done, info = q_agent.step_with_discretization(
                            state, action, jax.random.PRNGKey(step_count)
                        )
                    else:
                        # Use normal step for shortest path
                        action = policy_fn(state)
                        next_state, reward, done, info = ctx.env.step(state, action)
                    
                    step_time = float(info['travel'] + info['wait'])
                    total_time += step_time
                    step_reward = float(reward)
                    total_reward += step_reward
                    step_rewards.append(step_reward)
                    step_times.append(step_time)
                    traj.append(int(next_state.current_node))
                    step_count += 1
                    state = next_state
                    
                    if done:
                        break
                
                times.append(total_time)
                rewards.append(total_reward)
                paths.append(traj)
                step_rewards_list.append(step_rewards)
                step_times_list.append(step_times)
            
            return np.array(times), paths, np.array(rewards), step_rewards_list, step_times_list
        
        print(f"\nEvaluation iteration {i+1}/{eval_iter}")
        print(f"Evaluating Q-learning policy (continuous) for starts: {eval_starts} and pickups: {matched_q_pickups}")
        # Use continuous evaluation: discretize state for Q-table lookup, but use continuous env.step()
        q_times_continuous, q_paths_continuous, q_rewards_continuous, q_step_rewards_continuous, q_step_times_continuous = evaluate_policy_single(
            q_learning_policy, eval_starts, matched_q_pickups, use_discrete=True, use_continuous_eval=True
        )
        
        print(f"Evaluating Q-learning policy (discrete) for starts: {eval_starts} and pickups: {matched_q_pickups}")
        # Use discrete evaluation: same environment as training
        q_times_discrete, q_paths_discrete, q_rewards_discrete, q_step_rewards_discrete, q_step_times_discrete = evaluate_policy_single(
            q_learning_policy, eval_starts, matched_q_pickups, use_discrete=True, use_continuous_eval=False
        )
        
        print(f"Evaluating SP policy (continuous) for starts: {eval_starts} and pickups: {matched_sp_pickups_cont}")
        sp_times_continuous, sp_paths_continuous, sp_rewards_continuous, sp_step_rewards_continuous, sp_step_times_continuous = evaluate_policy_single(
            sp_policy, eval_starts, matched_sp_pickups_cont, use_discrete=False
        )
        
        print(f"Evaluating SP policy (discrete) for starts: {eval_starts} and pickups: {matched_sp_pickups_disc}")
        sp_times_discrete, sp_paths_discrete, sp_rewards_discrete, sp_step_rewards_discrete, sp_step_times_discrete = evaluate_policy_single(
            sp_policy_discrete, eval_starts, matched_sp_pickups_disc, use_discrete=True
        )
        
        # Print trajectories with step rewards and step times
        print(f"\nQ-Learning Trajectories (Continuous - Real World):")
        for agent_idx, (start, pickup, path, time, reward, step_rewards, step_times) in enumerate(
            zip(eval_starts, matched_q_pickups, q_paths_continuous, q_times_continuous, q_rewards_continuous, q_step_rewards_continuous, q_step_times_continuous)
        ):
            print(f"  Agent {agent_idx+1}: Start={int(start)}, Pickup={int(pickup)}")
            print(f"    Path: {' -> '.join(map(str, path))}")
            print(f"    Time: {time:.2f}s, Total Reward: {reward:.2f}, Steps: {len(path)-1}")
            print(f"    Rewards per step: {', '.join([f'Step {i+1}: {r:.2f}' for i, r in enumerate(step_rewards)])}")
            print(f"    Times per step: {', '.join([f'Step {i+1}: {t:.2f}s' for i, t in enumerate(step_times)])}")
        
        print(f"\nQ-Learning Trajectories (Discrete - Training Environment):")
        for agent_idx, (start, pickup, path, time, reward, step_rewards, step_times) in enumerate(
            zip(eval_starts, matched_q_pickups, q_paths_discrete, q_times_discrete, q_rewards_discrete, q_step_rewards_discrete, q_step_times_discrete)
        ):
            print(f"  Agent {agent_idx+1}: Start={int(start)}, Pickup={int(pickup)}")
            print(f"    Path: {' -> '.join(map(str, path))}")
            print(f"    Time: {time:.2f}s, Total Reward: {reward:.2f}, Steps: {len(path)-1}")
            print(f"    Rewards per step: {', '.join([f'Step {i+1}: {r:.2f}' for i, r in enumerate(step_rewards)])}")
            print(f"    Times per step: {', '.join([f'Step {i+1}: {t:.2f}s' for i, t in enumerate(step_times)])}")
        
        print(f"\nShortest Path Trajectories (Continuous - Real World):")
        for agent_idx, (start, pickup, path, time, reward, step_rewards, step_times) in enumerate(
            zip(eval_starts, matched_sp_pickups_cont, sp_paths_continuous, sp_times_continuous, sp_rewards_continuous, sp_step_rewards_continuous, sp_step_times_continuous)
        ):
            print(f"  Agent {agent_idx+1}: Start={int(start)}, Pickup={int(pickup)}")
            print(f"    Path: {' -> '.join(map(str, path))}")
            print(f"    Time: {time:.2f}s, Total Reward: {reward:.2f}, Steps: {len(path)-1}")
            print(f"    Rewards per step: {', '.join([f'Step {i+1}: {r:.2f}' for i, r in enumerate(step_rewards)])}")
            print(f"    Times per step: {', '.join([f'Step {i+1}: {t:.2f}s' for i, t in enumerate(step_times)])}")
        
        print(f"\nShortest Path Trajectories (Discrete - Fair Comparison):")
        for agent_idx, (start, pickup, path, time, reward, step_rewards, step_times) in enumerate(
            zip(eval_starts, matched_sp_pickups_disc, sp_paths_discrete, sp_times_discrete, sp_rewards_discrete, sp_step_rewards_discrete, sp_step_times_discrete)
        ):
            print(f"  Agent {agent_idx+1}: Start={int(start)}, Pickup={int(pickup)}")
            print(f"    Path: {' -> '.join(map(str, path))}")
            print(f"    Time: {time:.2f}s, Total Reward: {reward:.2f}, Steps: {len(path)-1}")
            print(f"    Rewards per step: {', '.join([f'Step {i+1}: {r:.2f}' for i, r in enumerate(step_rewards)])}")
            print(f"    Times per step: {', '.join([f'Step {i+1}: {t:.2f}s' for i, t in enumerate(step_times)])}")
        
        # Plot comparison: Q-learning vs SP discrete (fair comparison) - DISABLED
        # q_paths_list = [list(map(int, path)) for path in q_paths]
        # sp_paths_discrete_list = [list(map(int, path)) for path in sp_paths_discrete]
        # fig = plot_rl_vs_shortest_path(
        #     q_paths_list,
        #     sp_paths_discrete_list,
        #     ctx.G,
        #     ctx.node_to_idx,
        #     ctx.idx_to_node,
        #     eval_starts,
        #     matched_q_pickups,
        # )
        # matched_pickups_list = np.array(matched_q_pickups).tolist()
        # fig.savefig(f"qlearning_vs_sp_discrete_{np.array(eval_starts).tolist()}_{matched_pickups_list}.png")
        
        print(f"\nQ-learning avg time (continuous): {np.mean(q_times_continuous):.2f}")
        print(f"Q-learning avg time (discrete): {np.mean(q_times_discrete):.2f}")
        print(f"SP (continuous) avg time: {np.mean(sp_times_continuous):.2f}")
        print(f"SP (discrete) avg time: {np.mean(sp_times_discrete):.2f}")
        
        # Log metrics for both comparisons
    #     final_eval_metrics = {
    #         # Q-learning metrics
    #         "final_eval/qlearning_avg_time": float(np.mean(q_times)),
    #         "final_eval/qlearning_times": q_times.tolist(),
            
    #         # Continuous SP (real-world comparison)
    #         "final_eval/sp_continuous_avg_time": float(np.mean(sp_times_continuous)),
    #         "final_eval/qlearning_vs_sp_continuous_ratio": float(np.mean(q_times) / np.mean(sp_times_continuous)),
    #         "final_eval/qlearning_vs_sp_continuous_improvement": float((np.mean(sp_times_continuous) - np.mean(q_times)) / np.mean(sp_times_continuous) * 100),
    #         "final_eval/sp_continuous_times": sp_times_continuous.tolist(),
            
    #         # Discrete SP (fair comparison)
    #         "final_eval/sp_discrete_avg_time": float(np.mean(sp_times_discrete)),
    #         "final_eval/qlearning_vs_sp_discrete_ratio": float(np.mean(q_times) / np.mean(sp_times_discrete)),
    #         "final_eval/qlearning_vs_sp_discrete_improvement": float((np.mean(sp_times_discrete) - np.mean(q_times)) / np.mean(sp_times_discrete) * 100),
    #         "final_eval/sp_discrete_times": sp_times_discrete.tolist(),
    #     }
    #     wandb.log(final_eval_metrics)
        
    #     # Comparison table: Q-learning vs SP discrete (fair comparison)
    #     eval_comparison_table_discrete = wandb.Table(
    #         columns=["Metric", "Q-Learning", "SP (Discrete)", "Improvement"], 
    #         data=[
    #             ["Average Time", float(np.mean(q_times)), float(np.mean(sp_times_discrete)), float((np.mean(sp_times_discrete) - np.mean(q_times)) / np.mean(sp_times_discrete) * 100)],
    #             ["Min Time", float(np.min(q_times)), float(np.min(sp_times_discrete)), float((np.min(sp_times_discrete) - np.min(q_times)) / np.min(sp_times_discrete) * 100)],
    #             ["Max Time", float(np.max(q_times)), float(np.max(sp_times_discrete)), float((np.max(sp_times_discrete) - np.max(q_times)) / np.max(sp_times_discrete) * 100)],
    #             ["Std Time", float(np.std(q_times)), float(np.std(sp_times_discrete)), 0.0],
    #         ]
    #     )
    #     wandb.log({"final_evaluation_comparison_discrete": eval_comparison_table_discrete})
        
    #     # Comparison table: Q-learning vs SP continuous (real-world comparison)
    #     eval_comparison_table_continuous = wandb.Table(
    #         columns=["Metric", "Q-Learning", "SP (Continuous)", "Improvement"], 
    #         data=[
    #             ["Average Time", float(np.mean(q_times)), float(np.mean(sp_times_continuous)), float((np.mean(sp_times_continuous) - np.mean(q_times)) / np.mean(sp_times_continuous) * 100)],
    #             ["Min Time", float(np.min(q_times)), float(np.min(sp_times_continuous)), float((np.min(sp_times_continuous) - np.min(q_times)) / np.min(sp_times_continuous) * 100)],
    #             ["Max Time", float(np.max(q_times)), float(np.max(sp_times_continuous)), float((np.max(sp_times_continuous) - np.max(q_times)) / np.max(sp_times_continuous) * 100)],
    #             ["Std Time", float(np.std(q_times)), float(np.std(sp_times_continuous)), 0.0],
    #         ]
    #     )
    #     wandb.log({"final_evaluation_comparison_continuous": eval_comparison_table_continuous})
    
    wandb.finish()
    print("\n" + "="*60)
    print("Q-learning training completed!")
    print("="*60)

