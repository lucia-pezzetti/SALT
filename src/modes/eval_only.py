import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jax_random
import optax

from taxi_env import TaxiState, init_env
from models.q_network import QNetworkSimple, QNetworkUntied
from utils import offline_shortest_path_action
from training.mcts import estimate_returns_batch
from evaluation.evaluation import evaluate_all_combinations, create_traveling_times_plot

from .context import RunContext


def run_eval_only(args, ctx: RunContext) -> None:
    if args.load_params is None:
        raise ValueError("--load_params must be specified when using --eval_only")

    print("=== Evaluation-Only Mode ===")
    print(f"Loading parameters from: {args.load_params}")

    with open(args.load_params, 'rb') as f:
        saved_data = pickle.load(f)

    # Detect type
    if isinstance(saved_data, dict):
        if 'V' in saved_data:
            saved_params = saved_data['V']
            model_type = 'pi'
        else:
            saved_params = saved_data
            model_type = 'dqn'
    else:
        saved_params = saved_data
        model_type = 'dqn'

    print(f"Loaded {model_type} parameters")

    # DQN evaluation
    if model_type == 'dqn':
        if args.use_untied:
            model = QNetworkUntied(hidden_dim=128, num_actions=ctx.env.max_deg)
        else:
            model = QNetworkSimple(hidden_dim=128, num_actions=ctx.env.max_deg)

        def make_agent_policy(model, params, obs_fn_batch):
            def agent_policy(state: TaxiState) -> int:
                batch_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)
                sf = obs_fn_batch(batch_state)
                mask = batch_state.neighbor_mask
                q_vals = model.apply(params, sf, mask)
                return int(jnp.argmax(q_vals, axis=-1)[0])
            return agent_policy

        agent_policy = make_agent_policy(model, saved_params, ctx.obs_fn_batch)

        def sp_policy(state: TaxiState) -> int:
            curr = int(state.current_node)
            pickup = int(state.pickup_node)
            return int(offline_shortest_path_action(curr, pickup, ctx.env.adj_list, ctx.env.travel_times, ctx.env.distances, ctx.env.neighbor_mask_static[curr]))

        if args.plot_traveling_times:
            print("Evaluating ALL possible start-pickup combinations for comprehensive plotting...")
            all_starts = ctx.env.fixed_starts
            all_pickups = ctx.env.fixed_pickups
            max_combinations = 10000
            rl_times_array, sp_times_array, starts_array, pickups_array = evaluate_all_combinations(
                agent_policy, sp_policy, all_starts, all_pickups, max_combinations=max_combinations, env=ctx.env, init_env=init_env
            )
            plot_filename = f"traveling_times_comparison_{args.env_type}_{args.num_agents}agents_all_combinations_dqn.pdf"
            create_traveling_times_plot(
                rl_times_array, sp_times_array, starts_array, pickups_array, save_path=plot_filename
            )
        else:
            print("Running DQN evaluation...")
            key = jax.random.PRNGKey(0)
            for run in range(5):
                B = args.num_agents
                if args.fixed_eval:
                    eval_starts = jnp.array(args.fixed_starts)
                    eval_pickups = jnp.array(args.fixed_pickups)
                    print(f"Using fixed evaluation: starts={eval_starts}, pickups={eval_pickups}")
                else:
                    key, eval_key = jax.random.split(key)
                    eval_keys = jax.random.split(eval_key, 2*B)
                    start_keys, pickup_keys = eval_keys[:B], eval_keys[B:]
                    eval_starts = jnp.array([jax.random.choice(k, ctx.fixed_starts_idx) for k in start_keys])
                    eval_pickups = jnp.array([jax.random.choice(k, ctx.fixed_pickups_idx) for k in pickup_keys])

                dqn_times = []
                total_steps_all_agents = 0
                for agent_idx, (start, pickup) in enumerate(zip(eval_starts, eval_pickups)):
                    total_time = 0.0
                    step_count = 0
                    trajectory = []
                    state = init_env(jax.random.PRNGKey(0), start, pickup, ctx.env.neighbor_mask_static)[0]
                    trajectory.append(f"Start: {int(start)}")
                    while not state.done:
                        action = agent_policy(state)
                        state, _, _, info = ctx.env.step(state, action)
                        total_time += float(info['travel'] + info['wait'])
                        step_count += 1
                        trajectory.append(f"Step {step_count}: Node {int(state.current_node)}, Action {action}, Travel: {info['travel']:.2f}, Wait: {info['wait']:.2f}")
                    trajectory.append(f"End: Node {int(state.current_node)}, Total time: {total_time:.2f}, Total steps: {step_count}")
                    total_steps_all_agents += step_count
                    print(f"Run {run+1}, Agent {agent_idx+1} trajectory:")
                    for step in trajectory:
                        print(f"  {step}")
                    print()
                    dqn_times.append(total_time)

                sp_times = []
                for _, (start, pickup) in enumerate(zip(eval_starts, eval_pickups)):
                    total_time = 0.0
                    state = init_env(jax.random.PRNGKey(0), start, pickup, ctx.env.neighbor_mask_static)[0]
                    while not state.done:
                        action = sp_policy(state)
                        state, _, _, info = ctx.env.step(state, action)
                        total_time += float(info['travel'] + info['wait'])
                    sp_times.append(total_time)

                print(f"Run {run+1}: DQN avg time: {np.mean(dqn_times):.2f}, SP avg time: {np.mean(sp_times):.2f}, Total steps across all agents: {total_steps_all_agents}")
                print("="*50)

    # Policy-improvement evaluation
    elif model_type == 'pi':
        print("Policy improvement model detected - setting up value function...")

        config_file = args.config if hasattr(args, 'config') else "config.json"
        if not __import__('os').path.exists(config_file):
            print(f"ERROR: Config file {config_file} not found. Cannot recreate the exact model architecture.")
            print("Please ensure the config file exists and matches the training configuration.")
            raise SystemExit(1)
        else:
            import json
            with open(config_file, "r") as f:
                config = json.load(f)

        print(f"Using config: {config}")

        import haiku as hk

        class GraphAwareVFunction(hk.Module):
            def __init__(self, config, name=None):
                super().__init__(name=name)
                self.hidden_dim = config['num_hidden_units']
                self.num_layers = config['num_hidden_layers']
                self.activation = jax.nn.relu
                self.cycle_length = config.get('cycle_length', 200)

            def __call__(self, obs, adj_list=None, road_types=None):
                current_pos = obs[..., 0:2]
                pickup_pos = obs[..., 2:4]
                relative_pos = obs[..., 4:6]
                distance = obs[..., 6:7]
                angle = obs[..., 7:8]
                time = obs[..., 8:9]
                time = time / self.cycle_length
                distance = distance / 1.0
                angle = angle / jnp.pi
                features = jnp.concatenate([
                    current_pos, pickup_pos, relative_pos, distance, angle, time
                ], axis=-1)
                features = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(features)
                x = features
                for _ in range(self.num_layers):
                    residual = x if x.shape[-1] == self.hidden_dim else None
                    x = hk.Linear(self.hidden_dim)(x)
                    x = self.activation(x)
                    x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
                    if residual is not None:
                        x = x + residual
                output = hk.Linear(1)(x).squeeze(-1)
                return jnp.clip(output, -100.0, 100.0)

        V_net = hk.without_apply_rng(hk.transform(lambda obs: GraphAwareVFunction(config)(obs)))
        dummy_state, _ = init_env(jax.random.PRNGKey(0), ctx.fixed_starts_idx[0], ctx.fixed_pickups_idx[0], ctx.env.neighbor_mask_static)
        dummy_obs = ctx.obs_fn_single(dummy_state)
        dummy_key = jax.random.PRNGKey(0)
        _ = V_net.init(dummy_key, dummy_obs)

        V_params = saved_params
        V_apply = V_net.apply

        print("Value function created successfully!")
        print(f"Model architecture: {config['num_hidden_layers']} layers, {config['num_hidden_units']} hidden units")
        print(f"Parameter shapes: {jax.tree_util.tree_map(lambda x: x.shape, V_params)}")

        test_obs = dummy_obs
        test_value = V_apply(V_params, test_obs)
        print(f"Test value function output: {test_value}")

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
                    next_obs = ctx.obs_fn_single(next_state)
                    next_v = V_apply(V_params, next_obs.astype(float))
                    value = reward + ctx.env.gamma * next_v
                if value > best_v:
                    best_v, best_a = value, i
            return int(best_a) if best_a is not None else 0

        def sp_policy(state: TaxiState) -> int:
            curr = int(state.current_node)
            pickup = int(state.pickup_node)
            return int(offline_shortest_path_action(curr, pickup, ctx.env.adj_list, ctx.env.travel_times, ctx.env.distances, ctx.env.neighbor_mask_static[curr]))

        def evaluate_policy(policy, starts, pickups, max_steps=100, print_trajectories=False, run_id=0):
            times = []
            total_steps_all_agents = 0
            for agent_idx, (start, pickup) in enumerate(zip(starts, pickups)):
                total_time = 0.0
                step_count = 0
                trajectory = []
                state = init_env(jax.random.PRNGKey(0), start, pickup, ctx.env.neighbor_mask_static)[0]
                trajectory.append(f"Start: {int(start)}")
                while not state.done and step_count < max_steps:
                    action = policy(state)
                    state, _, _, info = ctx.env.step(state, action)
                    total_time += float(info['travel'] + info['wait'])
                    step_count += 1
                    trajectory.append(f"Step {step_count}: Node {int(state.current_node)}, Action {action}, Travel: {info['travel']:.2f}, Wait: {info['wait']:.2f}")
                trajectory.append(f"End: Node {int(state.current_node)}, Total time: {total_time:.2f}, Total steps: {step_count}")
                total_steps_all_agents += step_count
                if print_trajectories:
                    print(f"Run {run_id+1}, Agent {agent_idx+1} trajectory:")
                    for step in trajectory:
                        print(f"  {step}")
                    print()
                times.append(total_time)
            if print_trajectories:
                print(f"Run {run_id+1}: Total steps across all agents: {total_steps_all_agents}")
                print("="*50)
            return np.array(times)

        if args.plot_traveling_times:
            print("Evaluating ALL possible start-pickup combinations for comprehensive plotting...")
            all_starts = ctx.env.fixed_starts
            all_pickups = ctx.env.fixed_pickups
            max_combinations = 100000
            rl_times_array, sp_times_array, starts_array, pickups_array = evaluate_all_combinations(
                greedy_V_policy, sp_policy, all_starts, all_pickups, max_combinations=max_combinations, env=ctx.env, init_env=init_env
            )
            plot_filename = f"traveling_times_comparison_{args.env_type}_{args.num_agents}agents_all_combinations.pdf"
            create_traveling_times_plot(
                rl_times_array, sp_times_array, starts_array, pickups_array, save_path=plot_filename
            )
        else:
            key = jax_random.PRNGKey(0)
            for run_id in range(20):
                B = args.num_agents
                if args.fixed_eval:
                    eval_starts = jnp.array(args.fixed_starts)
                    rl_pickups = jnp.array(args.fixed_pickups)
                    sp_pickups = jnp.array(args.fixed_pickups)
                    print(f"Using fixed evaluation: starts={eval_starts}, pickups={rl_pickups}")
                    key, eval_key = jax.random.split(key)
                    init_keys = jax.random.split(eval_key, B**2)
                    print_trajectories = True
                else:
                    key, eval_key = jax.random.split(key)
                    eval_keys = jax.random.split(eval_key, 2*B+1)
                    start_keys, pickup_keys, base_key = eval_keys[:B], eval_keys[B:2*B], eval_keys[-1]
                    init_keys  = jax.random.split(base_key, B**2)
                    eval_starts = jnp.array([jax.random.choice(k, ctx.env.fixed_starts) for k in start_keys])
                    eval_pickups = jnp.array([jax.random.choice(k, ctx.env.fixed_pickups) for k in pickup_keys])
                    batch_init = jax.vmap(lambda k, s, p: init_env(k, s, p, ctx.env.neighbor_mask_static), in_axes=(0, 0, 0))
                    returns_matrix = estimate_returns_batch(
                        init_keys, V_apply, ctx.obs_fn_batch, batch_init,
                        V_params,
                        eval_starts, eval_pickups
                    )
                    _, rl_assignment = optax.assignment.hungarian_algorithm(-returns_matrix)
                    rl_pickups = eval_pickups[rl_assignment]
                    print_trajectories = run_id < 3
                    D = ctx.distances[eval_starts, :][:, eval_pickups]
                    _, sp_col = optax.assignment.hungarian_algorithm(D)
                    sp_pickups = eval_pickups[sp_col]

                v_time = evaluate_policy(greedy_V_policy, eval_starts, rl_pickups, print_trajectories=print_trajectories, run_id=run_id)
                mixed_time = evaluate_policy(greedy_V_policy, eval_starts, sp_pickups, print_trajectories=print_trajectories, run_id=run_id)
                sp_time = evaluate_policy(sp_policy, eval_starts, sp_pickups, print_trajectories=print_trajectories, run_id=run_id)
                print(f"Value Policy: starts - {eval_starts}, pickups - {rl_pickups}")
                print(f"SP Policy: starts - {eval_starts}, pickups - {sp_pickups}")
                print(f"Value Policy   avg time: {v_time.mean():.2f}")
                print(f"Mixed Policy   avg time: {mixed_time.mean():.2f}")
                print(f"SP Policy      avg time: {sp_time.mean():.2f}")


