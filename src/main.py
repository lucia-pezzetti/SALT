import networkx as nx
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jax_random, vmap
import os
import pickle
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading

# Enable JAX optimizations for GPU
jax.config.update('jax_enable_x64', False)  # Use float32 for better GPU performance
jax.config.update('jax_compilation_cache_dir', None)  # Will be set by environment
from numpy._core.fromnumeric import argsort
import wandb
import json

from taxi_env_utils import build_adj_and_time_matrix, make_obs_fn, load_or_compute_distance_matrix_parallel, load_or_build_graph
from taxi_env import TaxiEnv, init_env, TaxiState
from training.dqn_trainer import train
from models.q_network import QNetwork, QNetworkSimple, QNetworkUntied
from utils import estimate_returns_jit, EstimateReturnsState, smart_greedy_next_hop
from brute_force import brute_force_shortest_path, build_cost_matrix
from training.policy_improvement import get_init_fn, get_recurrent_fn, get_agent_loop, estimate_returns_batch
from training.ppo import get_ppo_init_fn, get_ppo_agent_loop
from evaluation.evaluation import evaluate_all_combinations, create_traveling_times_plot
from evaluation.plot_agent_paths import plot_agent_path_from_trajectory, plot_rl_vs_shortest_path

import argparse
from functools import partial
import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
import optax
import pickle
import os
from datetime import datetime

# Thread-safe caching
_cache_lock = threading.Lock()


def optimize_jax_config():
    """Set optimal JAX configuration"""
    # Use environment variable for cache dir if set, otherwise create one
    cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', f"/tmp/jax_cache_{os.getpid()}")
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', cache_dir)
    
    # Enable XLA optimizations
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_enable_compilation_cache', True)
    
    # Set memory preallocation (respect environment variables if set)
    if 'XLA_PYTHON_CLIENT_PREALLOCATE' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    
    # Get platform info
    platforms = os.environ.get('JAX_PLATFORMS', 'auto')
    
    # print(f"JAX optimized: cache_dir={cache_dir}, float32=True, compilation_cache=True, platforms={platforms}")

parser = argparse.ArgumentParser(description="Highly Optimized Ride-sharing Simulator")
parser.add_argument("--env_type", type=str, choices=["manhattan", "simple"], default="manhattan", help="Type of environment to use")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the customised grid environment")
parser.add_argument("--layer_width", type=int, default=3, help="Width of each layer in the customised grid environment")
parser.add_argument("--offset", type=float, default=0.0, help="Offset for the customised grid environment")
parser.add_argument("--cycle_length", type=int, default=200, help="Cycle length for the customised grid environment")
parser.add_argument("--no_congestion", type=bool, default=False, help="different types of roads have different congestion levels")
parser.add_argument("--place_name", type=str, default="Manhattan, New York City, New York, USA", help="Place name for the graph (used for Manhattan)")
parser.add_argument("--zone_shp", type=str, default="../data/processed/taxi_zones.shp", help="Path to the shapefile for zones (used for Manhattan)")
parser.add_argument("--num_agents", type=int, default=1, help="Number of agents in the environment")
parser.add_argument("--base_time", type=float, default=1.0, help="Base travel time for grid environment")
parser.add_argument("--max_steps", type=int, default=300, help="Maximum number of steps per episode")
parser.add_argument("--pickup_bonus", type=float, default=50.0, help="Bonus for picking up a passenger")
parser.add_argument("--timeout_penalty", type=float, default=-5.0, help="Penalty for timeout")
parser.add_argument("--n_expert_samples", type=int, default=5000, help="Number of expert samples for pretraining")
parser.add_argument("--hidden_dims", nargs='+', type=int, default=[512, 512], help="Hidden dimensions for the neural network")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for training")
parser.add_argument("--epochs", type=int, default=100_000, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--num_steps", type=int, default=128, help="Number of steps for training")
parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor for training")
parser.add_argument("--epsilon_start", type=float, default=0.5, help="Initial epsilon for epsilon-greedy policy")
parser.add_argument("--epsilon_end", type=float, default=0.1, help="Final epsilon for epsilon-greedy policy")
parser.add_argument("--sp_bias_beta", type=float, default=2.0, help="Shortest-path bias strength for exploration (higher = more SP bias)")
parser.add_argument("--params_dir", type=str, default=None, help="Output file for pretrained parameters")
parser.add_argument("--pretrain_ckpt", type=str, default="pretrained_params.pkl", help="Checkpoint file for pretrained parameters")
parser.add_argument("--model", type=str, default="ppo", choices=["dqn", "pi", "ppo"])
parser.add_argument("--config", "-c", type=str, default="config.json", help="Path to configuration file")
parser.add_argument("--wandb_project", type=str, default="taxi-pi", help="WandB project name")
parser.add_argument("--use_untied", type=bool, default=True, help="untied heads for Q-network")
parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for caching graph and distance data")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel distance computation")
parser.add_argument("--eval_only", action="store_true", help="Skip training and only run evaluation with saved parameters")
parser.add_argument("--load_params", type=str, default=None, help="Path to saved parameters file for evaluation-only mode")
parser.add_argument("--fixed_eval", action="store_true", help="Use fixed starts and pickups for evaluation instead of random sampling")
parser.add_argument("--fixed_starts", nargs='+', type=int, default=None, help="Fixed start node indices for evaluation (e.g., --fixed_starts 1 2 3)")
parser.add_argument("--fixed_pickups", nargs='+', type=int, default=None, help="Fixed pickup node indices for evaluation (e.g., --fixed_pickups 4 5 6)")
parser.add_argument("--plot_traveling_times", action="store_true", help="Create a plot comparing traveling times between RL and shortest path for each initial-destination pair")

args = parser.parse_args()

# Validate fixed evaluation parameters
if args.fixed_eval:
    if args.fixed_starts is None or args.fixed_pickups is None:
        raise ValueError("When using --fixed_eval, both --fixed_starts and --fixed_pickups must be provided")
    if len(args.fixed_starts) != len(args.fixed_pickups):
        raise ValueError("--fixed_starts and --fixed_pickups must have the same length")
    # print(f"Fixed evaluation mode: starts={args.fixed_starts}, pickups={args.fixed_pickups}")

# Optimize JAX configuration
optimize_jax_config()

# Create cache directory
os.makedirs(args.cache_dir, exist_ok=True)

if args.params_dir is None:
    args.params_dir = (
        f"{args.env_type}"
        f"_layers{args.num_layers}"
        f"width{args.layer_width}"
        f"_offset{args.offset:g}"
        f"_cycle{args.cycle_length}"
        f"_epochs{args.epochs}"
        f"_agents{args.num_agents}"
        f"_model{args.model}"
        ".pkl"
    )

# --- Load or build the environment with caching ---
if args.env_type == "manhattan":
    graph_cache_file = os.path.join(args.cache_dir, "manhattan_graph_4zones_1nodeperzone.pkl")
else:
    graph_cache_file = os.path.join(args.cache_dir, f"simple_graph_{args.num_layers}layers_{args.offset}offset.pkl")
G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params = load_or_build_graph(args, graph_cache_file)

print(f"Graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
print(f"starts: {fixed_starts_idx}, pickups: {fixed_pickups_idx}")

# --- Build graph structures ---
start_time = time.time()
adj_list, travel_times, neighbor_mask_static = build_adj_and_time_matrix(
    G, node_to_idx=node_to_idx
)

# Place graph structures on GPU with optimal memory layout
adj_list = jax.device_put(jnp.array(adj_list, dtype=jnp.int32))
travel_times = jax.device_put(jnp.array(travel_times, dtype=jnp.float32))
neighbor_mask_static = jax.device_put(jnp.array(neighbor_mask_static, dtype=bool))

# --- Precompute shortest-path distance matrix with parallel processing ---
if args.env_type == "manhattan":
    distance_cache_file = os.path.join(args.cache_dir, "manhattan_distances_4zones_1nodeperzone.pkl")
else:
    distance_cache_file = os.path.join(args.cache_dir, f"simple_distances_{args.num_layers}layers_{args.offset}offset.pkl")
dist_mat, hop_dist_mat, max_length, paths_dict = load_or_compute_distance_matrix_parallel(
    G, node_to_idx, distance_cache_file, args.num_workers
)

# Place distance matrices on GPU for faster access
distances = jax.device_put(jnp.array(dist_mat, dtype=jnp.float32))
hop_distances = jax.device_put(jnp.array(hop_dist_mat, dtype=jnp.float32))
# print(f"Max shortest path: {max_length}")

# --- Create environment ---
# print("Creating environment...")
env = TaxiEnv(
    adj_list=adj_list,
    travel_times=travel_times,
    neighbor_mask_static=neighbor_mask_static,
    fixed_starts=fixed_starts_idx,
    fixed_pickups=fixed_pickups_idx,
    distances=distances,
    hop_distances=hop_distances,
    max_steps=args.max_steps,
    traffic_params=traffic_params,
    paths_dict=paths_dict,
    pickup_bonus=args.pickup_bonus,
    timeout_penalty=args.timeout_penalty,
    gamma=args.gamma,   
)

# --- Observation function ---
obs_fn_single, obs_fn_batch = make_obs_fn(env, G, node_to_idx)

# --- Initialize batched environment states for metrics logging and evaluation ---
num_envs = args.num_agents
key = jax_random.PRNGKey(1)
key1, key2 = jax_random.split(key)
all_keys = jax_random.split(key2, 2 * num_envs)
start_keys, pickup_keys = all_keys[:num_envs], all_keys[num_envs:]
key3, eval_key = jax_random.split(key1)

start_idxs = jax.device_put(jnp.stack([jax_random.choice(k, fixed_starts_idx) for k in start_keys]))
pickup_idxs = jax.device_put(jnp.stack([jax_random.choice(k, fixed_pickups_idx) for k in pickup_keys]))

# print(f"Start idxs: {start_idxs}")
# print(f"Pickup idxs: {pickup_idxs}")

@jax.jit
def init_env_batch(keys, starts, pickups, neighbor_mask_static):
    # now vmap over *three* varying axes
    states, info = jax.vmap(
        init_env,
        in_axes=(0, 0, 0, None),
    )(keys, starts, pickups, neighbor_mask_static)
    return states

keys = jax_random.split(key3, num_envs)
batched_states = init_env_batch(
    keys,
    start_idxs.astype(jnp.int32),
    pickup_idxs.astype(jnp.int32),
    neighbor_mask_static,
)

# Pre-compute this once outside the training loop
estimate_state = EstimateReturnsState.create(rollout_steps=10, gamma=args.gamma)

# Ensure all fixed arrays are on GPU with consistent dtypes
fixed_starts_idx = jax.device_put(jnp.array(fixed_starts_idx, dtype=jnp.int32))
fixed_pickups_idx = jax.device_put(jnp.array(fixed_pickups_idx, dtype=jnp.int32))

# Handle evaluation-only mode
if args.eval_only:
    if args.load_params is None:
        raise ValueError("--load_params must be specified when using --eval_only")
    
    print("=== Evaluation-Only Mode ===")
    print(f"Loading parameters from: {args.load_params}")
    
    # Load saved parameters
    with open(args.load_params, 'rb') as f:
        saved_data = pickle.load(f)
    
    # Handle different parameter file formats
    if isinstance(saved_data, dict):
        if 'V' in saved_data:
            # Policy improvement format
            saved_params = saved_data['V']
            model_type = 'pi'
        else:
            # DQN format
            saved_params = saved_data
            model_type = 'dqn'
    else:
        # Direct DQN parameters
        saved_params = saved_data
        model_type = 'dqn'
    
    print(f"Loaded {model_type} parameters")
    
    # Skip training and go directly to evaluation
    if model_type == 'dqn':
        # DQN evaluation
        if args.use_untied:
            model = QNetworkUntied(hidden_dim=128, num_actions=env.max_deg)
        else:
            model = QNetworkSimple(hidden_dim=128, num_actions=env.max_deg)
        
        def make_agent_policy(model, params, obs_fn_batch):
            def agent_policy(state: TaxiState) -> int:
                batch_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)
                sf = obs_fn_batch(batch_state)
                mask = batch_state.neighbor_mask
                q_vals = model.apply(params, sf, mask)
                return int(jnp.argmax(q_vals, axis=-1)[0])
            return agent_policy
        
        agent_policy = make_agent_policy(model, saved_params, obs_fn_batch)
        
        # Define shortest path policy using smart greedy approach
        def sp_policy(state: TaxiState) -> int:
            """Smart greedy shortest path policy using travel time distances"""
            curr = int(state.current_node)
            pickup = int(state.pickup_node)
            
            # Use smart greedy approach for optimal pathfinding
            return int(smart_greedy_next_hop(curr, pickup, env.adj_list, env.travel_times, env.distances, env.neighbor_mask_static[curr]))
        
        # Initialize data collection for plotting if flag is enabled
        if args.plot_traveling_times:
            print("Evaluating ALL possible start-pickup combinations for comprehensive plotting...")
            # Use all possible start and pickup nodes
            all_starts = env.fixed_starts
            all_pickups = env.fixed_pickups
            
            # Set a reasonable limit to avoid extremely long evaluation times
            max_combinations = 10000  # Adjust this based on your needs
            
            # Evaluate all combinations
            rl_times_array, sp_times_array, starts_array, pickups_array = evaluate_all_combinations(
                agent_policy, sp_policy, all_starts, all_pickups, max_combinations=max_combinations, env=env, init_env=init_env
            )
            
            # Create filename for saving the plot
            plot_filename = f"traveling_times_comparison_{args.env_type}_{args.num_agents}agents_all_combinations_dqn.pdf"
            
            # Generate the plot
            create_traveling_times_plot(
                rl_times_array, 
                sp_times_array, 
                starts_array, 
                pickups_array, 
                save_path=plot_filename
            )
        else:
            # Run evaluation
            print("Running DQN evaluation...")
            for run in range(5):  # Run 5 evaluation episodes
                B = args.num_agents
                
                if args.fixed_eval:
                    # Use fixed starts and pickups
                    eval_starts = jnp.array(args.fixed_starts)
                    eval_pickups = jnp.array(args.fixed_pickups)
                    print(f"Using fixed evaluation: starts={eval_starts}, pickups={eval_pickups}")
                else:
                    # Use random sampling
                    key, eval_key = jax.random.split(key)
                    eval_keys = jax.random.split(eval_key, 2*B)
                    start_keys, pickup_keys = eval_keys[:B], eval_keys[B:]
                    eval_starts = jnp.array([jax.random.choice(k, fixed_starts_idx) for k in start_keys])
                    eval_pickups = jnp.array([jax.random.choice(k, fixed_pickups_idx) for k in pickup_keys])
                
                # Evaluate DQN policy
                dqn_times = []
                total_steps_all_agents = 0
                for agent_idx, (start, pickup) in enumerate(zip(eval_starts, eval_pickups)):
                    total_time = 0.0
                    step_count = 0
                    trajectory = []
                    state = init_env(jax.random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
                    trajectory.append(f"Start: {int(start)}")
                    
                    while not state.done:
                        action = agent_policy(state)
                        state, _, _, info = env.step(state, action)
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
                
                # Evaluate shortest path policy for comparison
                sp_times = []
                for agent_idx, (start, pickup) in enumerate(zip(eval_starts, eval_pickups)):
                    total_time = 0.0
                    state = init_env(jax.random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
                    
                    while not state.done:
                        action = sp_policy(state)
                        state, _, _, info = env.step(state, action)
                        total_time += float(info['travel'] + info['wait'])
                    
                    sp_times.append(total_time)
                
                print(f"Run {run+1}: DQN avg time: {np.mean(dqn_times):.2f}, SP avg time: {np.mean(sp_times):.2f}, Total steps across all agents: {total_steps_all_agents}")
                print("="*50)
    
    elif model_type == 'pi':
        print("Policy improvement model detected - setting up value function...")
        
        # We need a config file to recreate the V_apply function
        config_file = args.config if hasattr(args, 'config') else "config.json"
        if not os.path.exists(config_file):
            print(f"ERROR: Config file {config_file} not found. Cannot recreate the exact model architecture.")
            print("Please ensure the config file exists and matches the training configuration.")
            exit(1)
        else:
            import json
            with open(config_file, "r") as f:
                config = json.load(f)
        
        print(f"Using config: {config}")
        
        # Create the value function using the EXACT same architecture as training
        import haiku as hk
        
        # Use the EXACT same GraphAwareVFunction as in training
        class GraphAwareVFunction(hk.Module):
            def __init__(self, config, name=None):
                super().__init__(name=name)
                self.hidden_dim = config['num_hidden_units']  # This should be 256 from config
                self.num_layers = config['num_hidden_layers']  # This should be 3 from config
                self.activation = jax.nn.relu  # From config activation
                self.cycle_length = config.get('cycle_length', 200)  # From config
                
            def __call__(self, obs, adj_list=None, road_types=None):
                """
                Enhanced value function for lat/lon-based observations.
                Input: obs [..., 9] = [current_pos(2), pickup_pos(2), relative_pos(2), distance(1), angle(1), time(1)]
                """
                # Extract features from the 9-dimensional observation
                current_pos = obs[..., 0:2]      # [..., 2] - current position (lat/lon)
                pickup_pos = obs[..., 2:4]       # [..., 2] - pickup position (lat/lon)
                relative_pos = obs[..., 4:6]     # [..., 2] - direction vector
                distance = obs[..., 6:7]         # [..., 1] - distance to pickup
                angle = obs[..., 7:8]            # [..., 1] - direction angle
                time = obs[..., 8:9]    
                
                # Normalize time feature
                time = time / self.cycle_length  # Normalize by typical cycle length
                
                # Normalize distance
                distance = distance / 1.0  # Could be made configurable
                
                # Normalize angle to [-1, 1] range
                angle = angle / jnp.pi  # Convert from [-π, π] to [-1, 1]

                # Combine all features
                features = jnp.concatenate([
                    current_pos,    # [..., 2] - current position
                    pickup_pos,     # [..., 2] - pickup position
                    relative_pos,   # [..., 2] - direction vector
                    distance,       # [..., 1] - distance
                    angle,          # [..., 1] - angle
                    time            # [..., 1] - time
                ], axis=-1)  # Total: [..., 9]
                
                # Layer normalization for stability
                features = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(features)

                # Process through MLP
                x = features
                for i in range(self.num_layers):
                    residual = x if x.shape[-1] == self.hidden_dim else None
                    x = hk.Linear(
                        self.hidden_dim,
                        w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                        b_init=hk.initializers.Constant(0.0)
                    )(x)
                    x = self.activation(x)
                    # Layer norm after activation
                    x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
                    if residual is not None:
                        x = x + residual  # Residual connection
                        
                # Final layer with small initialization for stability
                output = hk.Linear(
                    1,
                    w_init=hk.initializers.VarianceScaling(0.1, "fan_in", "truncated_normal"),
                    b_init=hk.initializers.Constant(0.0)
                )(x).squeeze(-1)
                
                # Clip output
                return jnp.clip(output, -100.0, 100.0)
        
        # Create V_apply function
        V_net = hk.without_apply_rng(hk.transform(lambda obs: GraphAwareVFunction(config)(obs)))
        
        # Create a dummy observation to initialize the network
        dummy_state, _ = init_env(jax.random.PRNGKey(0), fixed_starts_idx[0], fixed_pickups_idx[0], env.neighbor_mask_static)
        dummy_obs = obs_fn_single(dummy_state)
        
        # Initialize with dummy params (we'll replace with saved params)
        dummy_key = jax.random.PRNGKey(0)
        dummy_params = V_net.init(dummy_key, dummy_obs)
        
        # Replace with saved parameters
        V_params = saved_params
        V_apply = V_net.apply
        
        print("Value function created successfully!")
        print(f"Model architecture: {config['num_hidden_layers']} layers, {config['num_hidden_units']} hidden units")
        print(f"Parameter shapes: {jax.tree_util.tree_map(lambda x: x.shape, V_params)}")
        
        # Test the value function with a dummy input
        test_obs = dummy_obs
        test_value = V_apply(V_params, test_obs)
        print(f"Test value function output: {test_value}")
        
        # Create policy using the value function
        def greedy_V_policy(state: TaxiState) -> int:
            """Use the learned value function to select actions greedily"""
            curr = state.current_node
            neighbors = env.adj_list[curr]
            mask = state.neighbor_mask
            
            best_a = None
            best_v = -jnp.inf
            
            for i, (n, valid) in enumerate(zip(neighbors, mask)):
                if not valid:
                    continue
                
                # Create next state without stepping (simulate transition)
                next_state, reward, done, info = env.step(state, i)
            
                if done:
                    # If this action leads to completion, use the immediate reward
                    value = reward
                else:
                    next_obs = obs_fn_single(next_state)
                    next_v = V_apply(V_params, next_obs.astype(float))
                    value = reward + env.gamma * next_v
                
                if value > best_v:
                    best_v, best_a = value, i
            
            return int(best_a) if best_a is not None else 0

        def sp_policy(state: TaxiState) -> int:
            """Smart greedy shortest path policy using travel time distances"""
            curr = int(state.current_node)
            pickup = int(state.pickup_node)
            
            # Use smart greedy approach for optimal pathfinding
            return int(smart_greedy_next_hop(curr, pickup, env.adj_list, env.travel_times, env.distances, env.neighbor_mask_static[curr]))

        def evaluate_policy(policy, starts, pickups, max_steps=100, print_trajectories=False, run_id=0):
            """Evaluate a policy on a given initial state"""
            times = []
            total_steps_all_agents = 0
            for agent_idx, (start, pickup) in enumerate(zip(starts, pickups)):
                total_time = 0.0
                step_count = 0
                trajectory = []
                # Initialize a fresh single‐env state
                state = init_env(jax.random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
                trajectory.append(f"Start: {int(start)}")
                
                while not state.done and step_count < max_steps:
                    action = policy(state)
                    state, _, _, info = env.step(state, action)
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
            
            return onp.array(times)


        # print("Evaluating learned policy...")
        
        # Initialize data collection for plotting if flag is enabled
        if args.plot_traveling_times:
            print("Evaluating ALL possible start-pickup combinations for comprehensive plotting...")
            # Use all possible start and pickup nodes
            all_starts = env.fixed_starts
            all_pickups = env.fixed_pickups
            
            # Set a reasonable limit to avoid extremely long evaluation times
            max_combinations = 100000  # Adjust this based on your needs
            
            # Evaluate all combinations
            rl_times_array, sp_times_array, starts_array, pickups_array = evaluate_all_combinations(
                greedy_V_policy, sp_policy, all_starts, all_pickups, max_combinations=max_combinations, env=env, init_env=init_env
            )
            
            # Create filename for saving the plot
            plot_filename = f"traveling_times_comparison_{args.env_type}_{args.num_agents}agents_all_combinations.pdf"
            
            # Generate the plot
            create_traveling_times_plot(
                rl_times_array, 
                sp_times_array, 
                starts_array, 
                pickups_array, 
                save_path=plot_filename
            )
        else:
            # Original evaluation loop for non-plotting mode
            for run_id in range(20):
                B = args.num_agents  # number of agents
                
                if args.fixed_eval:
                    # Use fixed starts and pickups
                    eval_starts = jnp.array(args.fixed_starts)
                    rl_pickups = jnp.array(args.fixed_pickups)
                    sp_pickups = jnp.array(args.fixed_pickups)
                    print(f"Using fixed evaluation: starts={eval_starts}, pickups={rl_pickups}")
                    # For fixed evaluation, we still need init_keys for the returns matrix computation
                    key, eval_key = jax.random.split(key)
                    init_keys = jax.random.split(eval_key, B**2)
                    print_trajectories = True
                else:
                    # Use random sampling
                    key, eval_key = jax.random.split(key)
                    
                    # Sample evaluation start and pickup points
                    eval_keys = jax.random.split(eval_key, 2*B+1)
                    start_keys, pickup_keys, base_key = eval_keys[:B], eval_keys[B:2*B], eval_keys[-1]
                    init_keys  = jax.random.split(base_key, B**2)
                    eval_starts = jnp.array([jax.random.choice(k, env.fixed_starts) for k in start_keys])
                    eval_pickups = jnp.array([jax.random.choice(k, env.fixed_pickups) for k in pickup_keys])
                
                    batch_init = jax.vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0))
                    
                    returns_matrix = estimate_returns_batch(
                        init_keys, V_apply, obs_fn_batch, batch_init,
                        V_params,
                        eval_starts, eval_pickups
                    )

                    # print(f"Opposite of returns matrix: {-returns_matrix}")
                
                    # Solve optimal transport problem
                    _, rl_assignment = optax.assignment.hungarian_algorithm(-returns_matrix)
                    rl_pickups = eval_pickups[rl_assignment]

                    # print(f"Value Policy: starts - {eval_starts}, pickups - {rl_pickups}")

                    # Print trajectories for first few runs
                    print_trajectories = run_id < 3
                    # Keep distance computation on GPU
                    D = distances[eval_starts, :][:, eval_pickups]  # Direct JAX indexing
                    # print(f"Distance matrix: {D}")
                    _, sp_col = optax.assignment.hungarian_algorithm(D)
                    sp_pickups = eval_pickups[sp_col]
                    # print(f"SP Policy: starts - {eval_starts}, pickups - {sp_pickups}")

                v_time = evaluate_policy(greedy_V_policy, eval_starts, rl_pickups, print_trajectories=print_trajectories, run_id=run_id)
                mixed_time = evaluate_policy(greedy_V_policy, eval_starts, sp_pickups, print_trajectories=print_trajectories, run_id=run_id)
                sp_time = evaluate_policy(sp_policy, eval_starts, sp_pickups, print_trajectories=print_trajectories, run_id=run_id)

                print(f"Value Policy: starts - {eval_starts}, pickups - {rl_pickups}")
                print(f"SP Policy: starts - {eval_starts}, pickups - {sp_pickups}")

                print(f"Value Policy   avg time: {v_time.mean():.2f}")
                print(f"Mixed Policy   avg time: {mixed_time.mean():.2f}")
                print(f"SP Policy      avg time: {sp_time.mean():.2f}")
        
        # # Run evaluation
        # print("Running PI evaluation...")
        # for run in range(5):  # Run 5 evaluation episodes
        #     B = args.num_agents
        #     key, eval_key = jax.random.split(key)
        #     eval_keys = jax.random.split(eval_key, 2*B)
        #     start_keys, pickup_keys = eval_keys[:B], eval_keys[B:]
        #     eval_starts = jnp.array([jax.random.choice(k, fixed_starts_idx) for k in start_keys])
        #     eval_pickups = jnp.array([jax.random.choice(k, fixed_pickups_idx) for k in pickup_keys])
            
        #     # Simple evaluation
        #     times = []
        #     for start, pickup in zip(eval_starts, eval_pickups):
        #         total_time = 0.0
        #         state = init_env(jax.random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
        #         while not state.done:
        #             action = pi_policy(state)
        #             state, _, _, info = env.step(state, action)
        #             total_time += float(info['travel'] + info['wait'])
        #         times.append(total_time)
            
        #     print(f"Run {run+1}: PI avg time: {np.mean(times):.2f}")
    
    print("=== Evaluation Complete ===")
    exit(0)

# WandB will be initialized after PPO config is loaded

# --- DQN ---
if args.model == "dqn":
    # print("Starting DQN training...")
    params = train(
        env               = env,
        init_state_fn     = lambda: batched_states,
        obs_fn_batch      = obs_fn_batch,
        key               = eval_key,
        fixed_starts      = fixed_starts_idx,
        fixed_pickups     = fixed_pickups_idx,
        estimate_state    = estimate_state,
        num_steps         = args.num_steps,
        epochs            = args.epochs,
        batch_size        = args.batch_size,
        lr                = args.lr,
        gamma             = args.gamma,
        epsilon_start     = args.epsilon_start,
        epsilon_end       = args.epsilon_end,
        logger            = None,  # Using wandb directly instead
        use_untied        = args.use_untied,
        sp_bias_beta      = args.sp_bias_beta,  # Shortest-path bias strength
    )

    # save the trained parameters
    with open(args.params_dir,'wb') as f:
        pickle.dump(params, f)

    def make_agent_policy(model, params, obs_fn_batch):
        """
        Wraps a Flax model into a single-state greedy policy using the batched obs_fn_batch.
        Correctly builds a batch of size 1 for inference.
        """
        def agent_policy(state: TaxiState) -> int:
            # Create a batched TaxiState of size 1 using tree_map
            batch_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)
            
            sf = obs_fn_batch(batch_state)
            mask = batch_state.neighbor_mask       # Already [1, max_deg]
            q_vals = model.apply(params, sf, mask)
            return int(jnp.argmax(q_vals, axis=-1)[0])
        return agent_policy
    
    if args.use_untied:
        model = QNetworkUntied(hidden_dim=128, num_actions=env.max_deg)
    else:
        model = QNetworkSimple(hidden_dim=128, num_actions=env.max_deg)
    agent_policy = make_agent_policy(model, params, obs_fn_batch)

    def init_state_fn(key, starts, pickups):
        state, _ = init_env(key, starts, pickups, env.neighbor_mask_static)
        return state

    # shortest-path policy using smart greedy approach
    def sp_policy(state: TaxiState) -> int:
        """Smart greedy shortest path policy using travel time distances"""
        curr = int(state.current_node)
        pickup = int(state.pickup_node)
        
        # Use smart greedy approach for optimal pathfinding
        return int(smart_greedy_next_hop(curr, pickup, env.adj_list, env.travel_times, env.distances, env.neighbor_mask_static[curr]))

    # Rollout helper
    def eval_matching(starts, pickups, policy) -> onp.ndarray:
        times = []
        for s, p in zip(onp.array(starts), onp.array(pickups)):
            # initialize a fresh single‐env state
            state = init_env(jax.random.PRNGKey(0), int(s), int(p), env.neighbor_mask_static)[0]
            total = 0.0
            while not bool(state.done):
                a = policy(state)
                state, _, _, info = env.step(state, a)
                total += float(info['travel'] + info['wait'])
            times.append(total)
        return onp.array(times)

    # print("Evaluating agent policy against SP baseline...")
    for _ in range(20):
        B = args.num_agents  # number of simultaneous taxi–passenger pairs

        # Sample one fixed batch of start & pickup indices
        eval_key, subkey = jax.random.split(eval_key)
        all_keys    = jax.random.split(subkey, 2 * B)
        start_keys, pickup_keys = all_keys[:B], all_keys[B:]
        starts  = jnp.array([jax.random.choice(k, fixed_starts_idx)  for k in start_keys])
        pickups = jnp.array([jax.random.choice(k, fixed_pickups_idx) for k in pickup_keys])

        batched_init = jax.vmap(
            partial(init_env, neighbor_mask_static=neighbor_mask_static),
            in_axes=(0, 0, 0),
            out_axes=(0, 0)
        )

        # Compute the RL‐matching via estimated returns (using your helper)
        def init_env_fn(starts: jnp.ndarray, pickups: jnp.ndarray):
                # make one new RNG‐key per trajectory
                rng_keys = jax.random.split(jax.random.PRNGKey(0), starts.shape[0])
                return batched_init(rng_keys, starts, pickups)

        R = estimate_returns_jit(
            env, params, model, obs_fn_batch,
            init_env_fn,
            starts, pickups,
            estimate_state=estimate_state,
            rollout_steps=10
        )  # → shape [B, B]
        R = jnp.array(R, dtype=jnp.float32)

        # solve the optimal transport problem
        _, rl_col = optax.assignment.hungarian_algorithm(-R)
        rl_pickups = pickups[rl_col]

        # SP‐matching via pure SP distances (keep on GPU)
        D = distances[starts, :][:, pickups]  # Direct JAX indexing, no host transfer
        _, sp_col = optax.assignment.hungarian_algorithm(D)
        sp_pickups = pickups[sp_col]

        rl_on_rl_times = eval_matching(starts,   rl_pickups, agent_policy)
        sp_on_sp_times = eval_matching(starts,   sp_pickups, sp_policy)

        # print("=== Matching & Policy Evaluation ===")
        print(f"RL matching + RL policy avg time: {rl_on_rl_times.mean():.2f}")
        print(f"SP matching + SP policy avg time: {sp_on_sp_times.mean():.2f}")

# --- Policy Improvement ---
elif args.model == "pi":
    obs_fn_single, obs_fn_batch = make_obs_fn(env, G, node_to_idx)
    # init
    if args.config is None:
        raise ValueError("Must pass --config path to your JSON for policy improvement")
    with open(args.config, "r") as f:
        config = json.load(f)

    # batch_size = num_agents
    config['batch_size'] = args.num_agents
    config['num_steps'] = args.epochs * config['eval_frequency']
    # Cap simulations to prevent excessive tree search on large graphs
    config['num_simulations'] = min(2*max_length, 64)  # Reduced from 128 to 64 for better performance
    # print(f"Num simulations in tree search: {config['num_simulations']} (capped from {2*max_length})")

    init_fn = get_init_fn(env, config, obs_fn_single)
    key, env_states, V_apply, V_opt_state, V_opt_update, get_V_params, V_target_params = init_fn(key)

    def linear_epsilon_decay(initial_eps=0.9, final_eps=0.1, decay_steps=10000):
        """Linear decay from initial_eps to final_eps over decay_steps"""
        def schedule(step):
            progress = jnp.clip(step / decay_steps, 0.0, 1.0)
            return initial_eps * (1.0 - progress) + final_eps * progress
        return schedule
    
    # build rec fn & agent loop
    key, subkey = jax_random.split(key)
    epsilon_schedule = linear_epsilon_decay(initial_eps=0.9, final_eps=0.05, decay_steps= config['num_steps'])
    recurrent_fn = get_recurrent_fn(env, V_apply, obs_fn_batch, epsilon_schedule, curriculum_steps=config['num_steps']*0.8)
    agent_loop = get_agent_loop(env, config, obs_fn_batch, obs_fn_single, V_apply, recurrent_fn, V_opt_update, get_V_params, epsilon_schedule)

    # initialize stats
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
        'visit_counts': jnp.zeros(env.num_nodes, dtype=jnp.int32), 
        'cumulative_visits': jnp.zeros(env.num_nodes, dtype=jnp.int32),
        # 'loss': jnp.array(0.0),  # Added loss tracking
        'policy_loss': jnp.array(0.0),  # Policy loss tracking
        'value_loss': jnp.array(0.0),   # Value loss tracking
        'total_loss': jnp.array(0.0),   # Total loss tracking
        'episode_travel': jnp.zeros(config['batch_size']),
        'episode_wait': jnp.zeros(config['batch_size']),
        'episode_total_time': jnp.zeros(config['batch_size']),
        'avg_wait': jnp.zeros(config['batch_size']),
        'avg_travel': jnp.zeros(config['batch_size']),
        'avg_total_time': jnp.zeros(config['batch_size']),
        # value_differences now logged directly to wandb
    }

    # Pre-allocate arrays for better memory management
    num_eval_steps = config['num_steps'] // config['eval_frequency']
    avg_returns = np.zeros(num_eval_steps, dtype=np.float32)
    times = np.zeros(num_eval_steps, dtype=np.float32)
    
    # print(f"Starting policy improvement training for {config['num_steps']} steps...")
    print(f"Training will run for {num_eval_steps} evaluation steps...")
    
    for i in range(num_eval_steps):
        step_start_time = time.time()
        state_dict, metrics = agent_loop(state_dict)
        step_time = time.time() - step_start_time
        
        # Evaluate shortest path policy for baseline comparison (outside JIT)
        def sp_policy(state: TaxiState) -> int:
            """Smart greedy shortest path policy using travel time distances"""
            curr = int(state.current_node)
            pickup = int(state.pickup_node)
            
            # Use smart greedy approach for optimal pathfinding
            return int(smart_greedy_next_hop(curr, pickup, env.adj_list, env.travel_times, env.distances, env.neighbor_mask_static[curr]))

        def evaluate_sp_policy(starts, pickups):
            """Evaluate shortest path policy on given start/pickup pairs"""
            sp_times = []
            # Convert JAX arrays to numpy arrays to avoid tree leaves issues
            starts_np = np.array(starts) if hasattr(starts, '__iter__') else np.array([starts])
            pickups_np = np.array(pickups) if hasattr(pickups, '__iter__') else np.array([pickups])
            
            for start, pickup in zip(starts_np, pickups_np):
                total_time = 0.0
                # Convert to Python int to avoid JAX tree issues
                start_int = int(start)
                pickup_int = int(pickup)
                state = init_env(jax.random.PRNGKey(0), start_int, pickup_int, env.neighbor_mask_static)[0]
                step_count = 0
                while not state.done and step_count < env.max_steps:
                    action = sp_policy(state)
                    state, _, _, info = env.step(state, action)
                    total_time += float(info['travel'] + info['wait'])
                    step_count += 1
                
                # Debug: print if episode didn't complete
                if not state.done:
                    print(f"Warning: SP episode didn't complete after {env.max_steps} steps")
                    print(f"Start: {start_int}, Pickup: {pickup_int}, Final node: {state.current_node}")
                
                sp_times.append(total_time)
            return np.array(sp_times)

        # Value differences are now logged directly to wandb in the metrics

        # Evaluate SP policy on current starts/pickups
        # Convert JAX arrays to numpy arrays to avoid tree leaves issues
        starts = np.array([int(x) for x in metrics['starts']])
        pickups = np.array([int(x) for x in metrics['pickups']])
        sp_times = evaluate_sp_policy(starts, pickups)
        avg_sp_total_time = np.mean(sp_times)
        
        # Debug: print values for first few steps
        # if i < 3:
        #     print(f"Step {i}: Learned avg_total_time: {metrics['avg_total_time']:.2f}")
        #     print(f"Step {i}: SP times: {sp_times}")
        #     print(f"Step {i}: SP avg_total_time: {avg_sp_total_time:.2f}")
        
        # Calculate performance ratio (learned policy / shortest path)
        # Values < 1.0 indicate learned policy is better than SP baseline
        # Add safety check to avoid division by zero
        if avg_sp_total_time > 0:
            performance_ratio = float(metrics['avg_total_time']) / float(avg_sp_total_time)
        else:
            performance_ratio = float('inf')  # Indicate SP evaluation failed
            # print(f"starts: {starts}, pickups: {pickups}")
            # print(f"Warning: SP evaluation returned zero time at step {i}")
        
        # Log comprehensive metrics for learning curve analysis
        log_metrics = {
            # Core performance metrics
            'training/step': i,
            'training/step_time': step_time,
            # 'training/loss': float(metrics['loss']),
            'training/policy_loss': float(metrics['policy_loss']),
            'training/value_loss': float(metrics['value_loss']),
            'training/total_loss': float(metrics['total_loss']),
            'training/avg_return': float(jnp.mean(metrics['avg_return'])),
            'training/avg_wait': float(metrics['avg_wait']),
            'training/avg_travel': float(metrics['avg_travel']),
            'training/avg_total_time': float(metrics['avg_total_time']),
            'training/avg_sp_total_time': float(avg_sp_total_time),
            'training/performance_ratio': performance_ratio,
            
            # Learning progress metrics
            'training/opt_step': int(state_dict['opt_t']),
            'training/epsilon': float(epsilon_schedule(state_dict['opt_t'])) if epsilon_schedule else 0.1,
            
            # Exploration metrics
            'training/visit_freq_entropy': float(-jnp.sum(metrics['visit_freq'] * jnp.log(metrics['visit_freq'] + 1e-8))),
            'training/unique_nodes_visited': float(jnp.sum(metrics['visit_freq'] > 0)),
            'training/total_visits': float(jnp.sum(metrics['visit_freq'] * metrics['visit_freq'].sum())),
            'training/visit_diversity': float(jnp.sum(metrics['visit_freq'] > 0) / env.num_nodes),
            
            # Episode statistics
            'training/num_episodes': float(jnp.sum(state_dict['num_episodes'])),
            'training/avg_episode_length': float(jnp.mean(state_dict['avg_total_time'])),
            'training/episode_completion_rate': float(jnp.sum(state_dict['num_episodes']) / (i + 1) / config['batch_size']),
            
            # Value function quality metrics (commented out - not available in this scope)
            # 'training/value_target_mean': float(jnp.mean(conservative_targets)),
            # 'training/value_target_std': float(jnp.std(conservative_targets)),
            # 'training/search_value_mean': float(jnp.mean(search_val)),
            # 'training/sp_value_mean': float(jnp.mean(V)),
            # 'training/value_improvement': float(jnp.mean(search_val - V)),
            
            # Efficiency metrics
            'training/improvement_over_sp': float(1.0 - performance_ratio),  # Positive = better than SP
            'training/relative_efficiency': float(avg_sp_total_time / metrics['avg_total_time']),
        }
        
        # Log additional state metrics (convert to float to avoid device-host transfers)
        if 'episode_return' in state_dict:
            log_metrics['training/episode_return'] = float(jnp.mean(state_dict['episode_return']))
        if 'num_episodes' in state_dict:
            log_metrics['training/num_episodes'] = float(jnp.sum(state_dict['num_episodes']))
        if 'visit_counts' in state_dict:
            log_metrics['training/total_visits'] = float(jnp.sum(state_dict['visit_counts']))
            log_metrics['training/unique_nodes_visited'] = float(jnp.sum(state_dict['visit_counts'] > 0))
        
        # Add learning curve specific metrics
        log_metrics.update({
            # Learning stability metrics
            # 'learning/loss_smooth': float(metrics['loss']),
            'learning/loss_smooth': float(metrics['total_loss']),  # For smoothing in plots
            'learning/return_smooth': float(jnp.mean(metrics['avg_return'])),  # For smoothing in plots
            'learning/performance_smooth': performance_ratio,  # For smoothing in plots
            
            # Convergence indicators
            'learning/convergence_indicator': float(jnp.abs(performance_ratio - 1.0)),  # Distance from optimal
            'learning/learning_rate': float(1.0 / (i + 1)),  # Effective learning rate
            
            # Statistical measures for learning curves
            'learning/return_std': float(jnp.std(metrics['avg_return'])),
            'learning/time_std': float(jnp.std(metrics['avg_total_time'])),
            'learning/coefficient_of_variation': float(jnp.std(metrics['avg_total_time']) / (jnp.mean(metrics['avg_total_time']) + 1e-8)),
            
            # Value difference metrics (V_policy - SP_policy performance)
            'evaluation/value_difference': float(metrics.get('value_difference', 0.0)),  # Current value difference
        })
        
        wandb.log(log_metrics, step=i)
        
        # Periodic detailed evaluation for learning curve analysis
        if i % 50 == 0 or i == num_eval_steps - 1:
            # Log detailed evaluation metrics
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
        
        # Print progress every 100 steps
        if i % 100 == 0 or i == num_eval_steps - 1:
            # print(f"Step {i}/{num_eval_steps} | Loss: {metrics['loss']:.4f} | "
            print(f"Step {i}/{num_eval_steps} | Policy Loss: {metrics['policy_loss']:.4f} | Value Loss: {metrics['value_loss']:.4f} | "
                  f"Performance Ratio: {performance_ratio:.3f} | "
                  f"Avg Time: {metrics['avg_total_time']:.2f}s | "
                  f"SP Time: {avg_sp_total_time:.2f}s | "
                  f"Improvement: {(1.0 - performance_ratio)*100:.1f}%")
        #           f"Avg Return: {jnp.mean(metrics['avg_return']):.4f} | "
        #           f"Avg Wait: {metrics['avg_wait']:.4f} | "
        #           f"Avg Travel: {metrics['avg_travel']:.4f} | "
        #           f"Step Time: {step_time:.2f}s")
            
        state_dict.update({
            'episode_return': jnp.zeros(config['batch_size']),
            'avg_travel': jnp.zeros(config['batch_size']),
            'avg_wait': jnp.zeros(config['batch_size']),
            'avg_return': jnp.zeros(config['batch_size']),
            'num_episodes': jnp.zeros(config['batch_size']),
            'episode_travel': jnp.zeros(config['batch_size']),
            'episode_wait': jnp.zeros(config['batch_size']),
            'visit_counts': jnp.zeros(env.num_nodes, dtype=jnp.int32),
            # 'loss': jnp.array(0.0),
            'policy_loss': jnp.array(0.0),
            'value_loss': jnp.array(0.0),
            'total_loss': jnp.array(0.0),
        })

    # Get final parameters
    final_V_params = state_dict['V_target_params']

    # save
    with open(args.params_dir + '.out', 'wb') as f:
        pickle.dump({'config': config, 'avg_returns': avg_returns, 'times': times}, f)
    with open(args.params_dir + '.params', 'wb') as f:
        pickle.dump({'V': final_V_params}, f)

    def greedy_V_policy(state: TaxiState) -> int:
        """Use the learned value function to select actions greedily"""
        curr = state.current_node
        neighbors = env.adj_list[curr]
        mask = state.neighbor_mask
        
        best_a = None
        best_v = -jnp.inf
        
        for i, (n, valid) in enumerate(zip(neighbors, mask)):
            if not valid:
                continue
            
            # Create next state without stepping (simulate transition)
            next_state, reward, done, info = env.step(state, i)
        
            if done:
                # If this action leads to completion, use the immediate reward
                value = reward
            else:
                next_obs = obs_fn_single(next_state)
                next_v = V_apply(final_V_params, next_obs.astype(float))
                value = reward + env.gamma * next_v
            
            if value > best_v:
                best_v, best_a = value, i
        
        return int(best_a) if best_a is not None else 0

    def sp_policy(state: TaxiState) -> int:
        """Smart greedy shortest path policy using travel time distances"""
        curr = int(state.current_node)
        pickup = int(state.pickup_node)
        
        # Use smart greedy approach for optimal pathfinding
        return int(smart_greedy_next_hop(curr, pickup, env.adj_list, env.travel_times, env.distances, env.neighbor_mask_static[curr]))

    def evaluate_policy(policy, starts, pickups, max_steps=100):
        """Evaluate a policy on a given initial state"""
        times = []
        paths = []
        rewards = []
        for start, pickup in zip(starts, pickups):
            total_time = 0.0
            total_reward = 0.0
            step_count = 0
            # Initialize a fresh single‐env state
            state = init_env(jax.random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
            paths = [state.current_node]  # Reset paths for each evaluation
            print(f"Start: {start}, Pickup: {pickup}, Initial node: {state.current_node}")
            while not state.done and step_count < max_steps:
                action = policy(state)
                state, reward, _, info = env.step(state, action)
                total_time += float(info['travel'] + info['wait'])
                step_count += 1
                print(f"Step {step_count}: Node {int(state.current_node)}, Action {action}, Travel: {info['travel']:.2f}, Wait: {info['wait']:.2f}")
                paths.append(state.current_node)
                total_reward += float(reward)
            times.append(total_time)
            rewards.append(total_reward)
        return onp.array(times), onp.array(paths), onp.array(rewards)

    # print("Evaluating learned policy...")
    for _ in range(20):
        B = args.num_agents  # number of agents
        key, eval_key = jax.random.split(key)
        
        # Sample evaluation start and pickup points
        eval_keys = jax.random.split(eval_key, 2*B+1)
        start_keys, pickup_keys, base_key = eval_keys[:B], eval_keys[B:2*B], eval_keys[-1]
        init_keys  = jax.random.split(base_key, B**2)
        eval_starts = jnp.array([jax.random.choice(k, env.fixed_starts) for k in start_keys])
        eval_pickups = jnp.array([jax.random.choice(k, env.fixed_pickups) for k in pickup_keys])
        
        batch_init = jax.vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0))
        
        returns_matrix = estimate_returns_batch(
            init_keys, V_apply, obs_fn_batch, batch_init,
            final_V_params,
            eval_starts, eval_pickups
        )
        
        # Solve optimal transport problem
        _, rl_assignment = optax.assignment.hungarian_algorithm(-returns_matrix)
        rl_pickups = eval_pickups[rl_assignment]

        print(f"Value Policy: starts - {eval_starts}, pickups - {rl_pickups}")

        v_time, v_paths, v_rewards = evaluate_policy(greedy_V_policy, eval_starts, rl_pickups)

        # Keep distance computation on GPU
        D = distances[eval_starts, :][:, eval_pickups]  # Direct JAX indexing
        _, sp_col = optax.assignment.hungarian_algorithm(D)
        sp_pickups = eval_pickups[sp_col]
        print(f"SP Policy: starts - {eval_starts}, pickups - {sp_pickups}")

        sp_time, sp_paths, sp_rewards = evaluate_policy(sp_policy, eval_starts, sp_pickups)

        print(f"Value Policy   avg time: {v_time.mean():.2f}")
        print(f"SP Policy      avg time: {sp_time.mean():.2f}")
        print(f"Value Policy   avg reward: {v_rewards.mean():.2f}")
        print(f"SP Policy      avg reward: {sp_rewards.mean():.2f}")

        fig = plot_rl_vs_shortest_path(v_paths, sp_paths, G, node_to_idx, idx_to_node, eval_starts, rl_pickups)
        fig.savefig(f"rl_vs_sp_{eval_starts}_{rl_pickups}.png")

# --- PPO Training ---
elif args.model == "ppo":
    obs_fn_single, obs_fn_batch = make_obs_fn(env, G, node_to_idx)
    
    # Load PPO configuration
    if args.config is None:
        raise ValueError("Must pass --config path to your JSON for PPO training")
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Override config with command line arguments
    config['batch_size'] = args.num_agents
    config['num_steps'] = args.epochs * config['eval_frequency']
    config['max_deg'] = env.max_deg
    config['cycle_length'] = args.cycle_length
    config['max_steps'] = 3*max_length
    
    # print("Initializing PPO networks...")
    
    # Initialize PPO
    init_fn = get_ppo_init_fn(env, config, obs_fn_single)
    key, policy_params, value_params, policy_apply, value_apply, policy_opt, value_opt, policy_opt_state, value_opt_state = init_fn(key)
    
    # print("Creating PPO training loop...")
    
    # Setup fixed evaluation set
    eval_starts = env.fixed_starts[:min(5, len(env.fixed_starts))]  # Use first 5 starts
    eval_pickups = env.fixed_pickups[:min(5, len(env.fixed_pickups))]  # Use first 5 pickups
    
    # print(f"Fixed evaluation set: {len(eval_starts)} starts, {len(eval_pickups)} pickups")
    # print(f"Evaluation starts: {eval_starts}")
    # print(f"Evaluation pickups: {eval_pickups}")
    
    # Create PPO training loop with evaluation
    agent_loop = get_ppo_agent_loop(
        env, config, obs_fn_batch, obs_fn_single, 
        policy_apply, value_apply, policy_opt, value_opt,
        eval_starts=eval_starts, eval_pickups=eval_pickups
    )
    
    # Initialize training state with proper environment states
    # Initialize environment states first
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, config['batch_size'])
    starts = jnp.stack([jax_random.choice(k, env.fixed_starts) for k in subkeys])
    pickups = jnp.stack([jax_random.choice(k, env.fixed_pickups) for k in subkeys])
    
    batch_init = vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0))
    env_states, _ = batch_init(subkeys, starts, pickups)
    
    state_dict = {
        'key': key,
        'policy_params': policy_params,
        'value_params': value_params,
        'policy_opt_state': policy_opt_state,
        'value_opt_state': value_opt_state,
        'opt_t': 0,
        'episode_return': jnp.zeros(config['batch_size']),
        'avg_return': 0.0,
        'num_episodes': 0,
        'env_states': env_states,
        'last_start': starts
    }
    
    # Initialize wandb for logging after config is loaded
    wandb.init(
        project="ride-sharing-ppo-evaluation",
        name=f"ppo-{args.env_type}-{args.epochs}epochs-{args.num_agents}agents-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            # Environment configuration
            "env_type": args.env_type,
            "num_layers": args.num_layers,
            "layer_width": args.layer_width,
            "offset": args.offset,
            "cycle_length": args.cycle_length,
            "no_congestion": args.no_congestion,
            "place_name": args.place_name,
            
            # Training configuration
            "model": args.model,
            "epochs": args.epochs,
            "num_agents": args.num_agents,
            "batch_size": args.batch_size,
            "num_steps": args.num_steps,
            "gamma": args.gamma,
            "lr": args.lr,
            
            # Environment details
            "num_nodes": env.num_nodes,
            "max_deg": env.max_deg,
            "max_steps": env.max_steps,
            "pickup_bonus": env.pickup_bonus,
            "timeout_penalty": env.timeout_penalty,
            
            # PPO configuration
            "ppo_config": config,
            
            # System configuration
            "cache_dir": args.cache_dir,
            "num_workers": args.num_workers,
            "traffic_params": traffic_params,
            
            # Evaluation configuration
            "eval_starts": eval_starts.tolist() if 'eval_starts' in locals() else [],
            "eval_pickups": eval_pickups.tolist() if 'eval_pickups' in locals() else [],
            
            # Metadata
            "timestamp": datetime.now().isoformat(),
            "git_commit": "unknown",  # Could be added with git commands
            "python_version": "3.10",
            "jax_version": jax.__version__,
        }
    )
    
    # print("Training metrics will be logged to WandB")
    
    # Compute shortest path baseline for comparison
    def evaluate_shortest_path_baseline(starts, pickups):
        """Evaluate shortest path policy on fixed set"""
        sp_times = []
        sp_rewards = []
        sp_steps = []
        sp_completed = []
        
        for start, pickup in zip(starts, pickups):
            total_time = 0.0
            total_reward = 0.0
            step_count = 0
            state = init_env(jax.random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
            
            while not state.done and step_count < env.max_steps:
                # Use smart greedy shortest path
                action = smart_greedy_next_hop(
                    state.current_node, state.pickup_node,
                    env.adj_list, env.travel_times, env.distances,
                    env.neighbor_mask_static[state.current_node]
                )
                state, reward, done, info = env.step(state, action)
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
            'completed': sp_completed
        }
    
    # Compute shortest path baseline
    # print("Computing shortest path baseline...")
    sp_baseline = evaluate_shortest_path_baseline(eval_starts, eval_pickups)
    
    # Training loop
    for epoch in range(args.epochs):
        state_dict, metrics = agent_loop(state_dict)
        
        # Calculate performance ratio vs shortest path
        eval_avg_reward = float(metrics.get('eval/avg_reward', 0.0))
        sp_avg_reward = sp_baseline['avg_reward']
        
        # Handle division by zero and negative rewards
        if abs(sp_avg_reward) < 1e-6:  # Use small epsilon instead of exact zero
            if abs(eval_avg_reward) < 1e-6:
                performance_ratio = 1.0  # Both are essentially 0, consider equal
            else:
                # If SP is 0 but PPO has reward, use a relative measure
                performance_ratio = 1.0 + (eval_avg_reward / 100.0)  # Scale by 100 for relative comparison
        else:
            performance_ratio = eval_avg_reward / sp_avg_reward
        
        # Comprehensive logging to WandB every epoch
        log_metrics = {
            # Core training metrics
            "training/epoch": epoch,
            "training/avg_return": float(metrics['avg_return']),
            "training/num_episodes": int(metrics['num_episodes']),
            # 'training/loss': float(metrics['loss']),
            "training/policy_loss": float(metrics['policy_loss']),
            "training/value_loss": float(metrics['value_loss']),
            "training/total_loss": float(metrics['total_loss']),
            
            # Learning curve specific metrics
            "learning/return_trend": float(metrics['avg_return']),  # For smoothing
            # 'learning/loss_trend': float(metrics['loss']),
            "learning/loss_trend": float(metrics['total_loss']),  # For smoothing
            "learning/learning_rate": float(1.0 / (epoch + 1)),  # Effective learning rate
            "learning/progress": float(epoch / args.epochs),  # Training progress
            
            # Performance metrics
            "performance/avg_return": float(metrics['avg_return']),
            "performance/episodes_completed": int(metrics['num_episodes']),
            # 'performance/loss': float(metrics['loss']),
            "performance/policy_loss": float(metrics['policy_loss']),
            "performance/value_loss": float(metrics['value_loss']),
            "performance/total_loss": float(metrics['total_loss']),
            
            # Training stability
            "stability/epoch": epoch,
            # 'stability/loss': float(metrics['loss']),
            "stability/total_loss": float(metrics['total_loss']),
            "stability/return": float(metrics['avg_return']),
            
            # PPO-specific metrics
            "ppo/buffer_size": int(metrics.get('buffer_size', 0)),
            "ppo/buffer_utilization": float(metrics.get('buffer_utilization', 0.0)),
            "ppo/learning_active": bool(metrics.get('learning_active', False)),
            
            # Evaluation metrics
            "evaluation/avg_reward": float(metrics.get('eval/avg_reward', 0.0)),
            "evaluation/avg_steps": float(metrics.get('eval/avg_steps', 0.0)),
            "evaluation/completion_rate": float(metrics.get('eval/completion_rate', 0.0)),
            "evaluation/sp_baseline_reward": float(sp_baseline['avg_reward']),
            "evaluation/sp_baseline_steps": float(sp_baseline['avg_steps']),
            "evaluation/sp_baseline_completion": float(sp_baseline['completion_rate']),
            "evaluation/performance_ratio": performance_ratio,
            "evaluation/improvement_over_sp": float(performance_ratio - 1.0),
            
            # Individual episode evaluation results
            "episodes/eval_rewards": metrics.get('eval/total_rewards', []),
            "episodes/eval_steps": metrics.get('eval/total_steps', []),
            "episodes/eval_completed": metrics.get('eval/completed', []),
            
            # Training efficiency
            "efficiency/epoch": epoch,
            "efficiency/return_per_episode": float(metrics['avg_return']) / max(1, int(metrics['num_episodes'])),
            # 'efficiency/loss_per_episode': float(metrics['loss']) / max(1, int(metrics['num_episodes'])),
            "efficiency/loss_per_episode": float(metrics['total_loss']) / max(1, int(metrics['num_episodes'])),
        }
        
        # Log to WandB every epoch
        wandb.log(log_metrics)
        
        # Print progress every 10 epochs
        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}: "
        #           f"Avg Return: {metrics['avg_return']:.3f}, "
        #           f"Episodes: {metrics['num_episodes']:.0f}, "
        #         #   f"Loss: {metrics['loss']:.6f}, "
        #           f"Policy Loss: {metrics['policy_loss']:.6f}, Value Loss: {metrics['value_loss']:.6f}, "
        #           f"Eval Reward: {eval_avg_reward:.3f}, "
        #           f"SP Ratio: {performance_ratio:.3f}")
    
    # print("PPO training completed!")
    
    # Log final summary to WandB
    final_metrics = {
        "final/total_epochs": args.epochs,
        "final/final_avg_return": float(metrics['avg_return']),
        # 'final/final_loss': float(metrics['loss']),
        "final/final_policy_loss": float(metrics['policy_loss']),
        "final/final_value_loss": float(metrics['value_loss']),
        "final/final_total_loss": float(metrics['total_loss']),
        "final/total_episodes": int(metrics['num_episodes']),
        "final/final_eval_reward": float(metrics.get('eval/avg_reward', 0.0)),
        "final/final_eval_steps": float(metrics.get('eval/avg_steps', 0.0)),
        "final/final_completion_rate": float(metrics.get('eval/completion_rate', 0.0)),
        "final/final_performance_ratio": performance_ratio,
        "final/improvement_over_sp": float(performance_ratio - 1.0),
        "final/training_successful": True,
    }
    
    # Log final summary
    wandb.log(final_metrics)
    
    # Log training summary as a table
    summary_table = wandb.Table(columns=["Metric", "Value"], data=[
        ["Total Epochs", int(args.epochs)],
        ["Final Avg Return", float(metrics['avg_return'])],
        # 'Final Loss': float(metrics['loss']),
        ["Final Policy Loss", float(metrics['policy_loss'])],
        ["Final Value Loss", float(metrics['value_loss'])],
        ["Final Total Loss", float(metrics['total_loss'])],
        ["Total Episodes", int(metrics['num_episodes'])],
        ["Final Eval Reward", float(metrics.get('eval/avg_reward', 0.0))],
        ["Final Eval Steps", float(metrics.get('eval/avg_steps', 0.0))],
        ["Final Completion Rate", float(metrics.get('eval/completion_rate', 0.0))],
        ["Performance Ratio", float(performance_ratio)],
        ["Improvement over SP", float((performance_ratio - 1.0) * 100)],
        ["SP Baseline Reward", float(sp_baseline['avg_reward'])],
        ["Training Duration", int(args.epochs)],
    ])
    
    wandb.log({"training_summary": summary_table})
    
    # Create policy function for evaluation
    def ppo_policy(state: TaxiState) -> int:
        """Use trained PPO policy for action selection"""
        obs = obs_fn_single(state)
        logits = policy_apply(state_dict['policy_params'], obs)
        
        # Apply action masking with padding to match logits shape
        neighbor_mask = state.neighbor_mask
        if neighbor_mask.shape[-1] != logits.shape[-1]:
            # Pad the neighbor mask to match logits shape
            padding = logits.shape[-1] - neighbor_mask.shape[-1]
            neighbor_mask = jnp.pad(neighbor_mask, (0, padding), constant_values=False)
        
        masked_logits = jnp.where(
            neighbor_mask,
            logits,
            -jnp.inf
        )
        
        # Select action greedily (or sample for exploration)
        action = jnp.argmax(masked_logits)
        return int(action)
    
    # Evaluate PPO policy
    # print("Evaluating PPO policy...")
    for _ in range(20):
        B = args.num_agents
        key, eval_key = jax.random.split(key)
        
        # Sample evaluation start and pickup points
        eval_keys = jax.random.split(eval_key, 2*B+1)
        start_keys, pickup_keys, base_key = eval_keys[:B], eval_keys[B:2*B], eval_keys[-1]
        eval_starts = jnp.array([jax.random.choice(k, env.fixed_starts) for k in start_keys])
        eval_pickups = jnp.array([jax.random.choice(k, env.fixed_pickups) for k in pickup_keys])
        
        # Use Hungarian algorithm matching based on estimated returns from the value network
        # Build a returns matrix for all (start_i, pickup_j) pairs and pick the assignment that maximizes returns
        init_keys = jax.random.split(base_key, B**2)
        batch_init = jax.vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0))
        returns_matrix = estimate_returns_batch(
            init_keys, value_apply, obs_fn_batch, batch_init,
            state_dict['value_params'],
            eval_starts, eval_pickups
        )
        _, ppo_assignment = optax.assignment.hungarian_algorithm(-returns_matrix)
        ppo_pickups = eval_pickups[ppo_assignment]
        
        # Evaluate PPO policy
        
        batch_step = jax.jit(vmap(env.step, in_axes=(0, 0)))
                
        # Initialize all agents in parallel with optimally matched pickups
        eval_keys = jax.random.split(eval_key, B)
        states, _ = batch_init(eval_keys, eval_starts, ppo_pickups)
        
        # Run parallel evaluation
        ppo_times = []
        ppo_rewards = []
        step_count = 0
        max_steps = 3*max_length
        
        while not jnp.all(states.done) and step_count < max_steps:
            # Get actions for all agents in parallel
            batch_obs = obs_fn_batch(states)
            batch_masks = states.neighbor_mask
            logits = policy_apply(state_dict['policy_params'], batch_obs)
            masked_logits = jnp.where(batch_masks, logits, -1e8)
            
            # Use a new random key for each step
            eval_key, action_key = jax.random.split(eval_key)
            actions = jax.random.categorical(action_key, masked_logits, axis=-1)
            
            # Step all environments in parallel
            states, rewards, dones, infos = batch_step(states, actions)
            
            # Accumulate times and rewards
            if step_count == 0:
                ppo_times = jnp.zeros(B)
                ppo_rewards = jnp.zeros(B)
            
            ppo_times += infos['travel'] + infos['wait']
            ppo_rewards += rewards
            step_count += 1
        
        ppo_times = ppo_times.tolist()
        ppo_rewards = ppo_rewards.tolist()
        
        # Evaluate shortest path for comparison in parallel
        # print(f"Evaluating shortest path policy...")
        
        # Shortest path matching
        D = distances[eval_starts, :][:, eval_pickups]  # Shape: (B, B)
        _, sp_assignment = optax.assignment.hungarian_algorithm(D)
        sp_pickups = eval_pickups[sp_assignment]
        
        # Initialize all agents in parallel (reuse batch_init from above)
        sp_states, _ = batch_init(eval_keys, eval_starts, sp_pickups)
        
        # Run parallel evaluation
        sp_times = []
        sp_rewards = []
        step_count = 0
        
        while not jnp.all(sp_states.done) and step_count < max_steps:
            # Get actions for all agents in parallel using sp_policy
            # Use vmap to apply smart_greedy_next_hop to all states
            def get_sp_action(state):
                return smart_greedy_next_hop(state.current_node, state.pickup_node, 
                                           env.adj_list, env.travel_times, env.distances, 
                                           env.neighbor_mask_static[state.current_node])
            
            sp_actions = jax.vmap(get_sp_action)(sp_states)
            
            # Step all environments in parallel
            sp_states, sp_rewards_step, sp_dones, sp_infos = batch_step(sp_states, sp_actions)
            
            # Accumulate times and rewards
            if step_count == 0:
                sp_times = jnp.zeros(B)
                sp_rewards = jnp.zeros(B)
            
            sp_times += sp_infos['travel'] + sp_infos['wait']
            sp_rewards += sp_rewards_step
            step_count += 1
        
        sp_times = sp_times.tolist()
        sp_rewards = sp_rewards.tolist()

        # Also produce trajectory plots for PPO vs SP (like the PI block does)
        def greedy_ppo_policy(state):
            obs = obs_fn_single(state)
            logits = policy_apply(state_dict['policy_params'], obs)
            # state.neighbor_mask is available for a single state
            neighbor_mask = state.neighbor_mask
            if logits.shape[-1] != neighbor_mask.shape[-1]:
                padding = logits.shape[-1] - neighbor_mask.shape[-1]
                neighbor_mask = jnp.pad(neighbor_mask, (0, padding), constant_values=False)
            masked_logits = jnp.where(neighbor_mask, logits, -jnp.inf)
            return int(jnp.argmax(masked_logits))

        # Local rollout for plotting (single-env per agent)
        def evaluate_policy_single(policy_fn, starts, pickups):
            times, paths, rewards = [], [], []
            for start, pickup in zip(onp.array(starts).tolist(), onp.array(pickups).tolist()):
                total_time, total_reward = 0.0, 0.0
                step_count = 0
                state = init_env(jax.random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
                traj = [state.current_node]
                while (not bool(state.done)) and step_count < 3*max_length:
                    action = policy_fn(state)
                    state, reward, _, info = env.step(state, action)
                    total_time += float(info['travel'] + info['wait'])
                    total_reward += float(reward)
                    traj.append(state.current_node)
                    step_count += 1
                times.append(total_time)
                rewards.append(total_reward)
                paths.append([int(x) for x in traj])
            return onp.array(times), paths, onp.array(rewards)

        # Evaluate single-agent rollouts for plotting (uses same starts and matched pickups)
        ppo_time_plot, ppo_paths, ppo_rewards_plot = evaluate_policy_single(greedy_ppo_policy, eval_starts, ppo_pickups)

        def sp_policy(state):
            return smart_greedy_next_hop(
                state.current_node, state.pickup_node,
                env.adj_list, env.travel_times, env.distances,
                env.neighbor_mask_static[state.current_node]
            )
        sp_time_plot, sp_paths, sp_rewards_plot = evaluate_policy_single(sp_policy, eval_starts, sp_pickups)

        # Ensure paths are plain lists of ints for plotting utilities
        ppo_paths_list = [list(map(int, path)) for path in ppo_paths]
        sp_paths_list = [list(map(int, path)) for path in sp_paths]
        fig = plot_rl_vs_shortest_path(ppo_paths_list, sp_paths_list, G, node_to_idx, idx_to_node, eval_starts, ppo_pickups)
        fig.savefig(f"ppo_vs_sp_{onp.array(eval_starts).tolist()}_{onp.array(ppo_pickups).tolist()}.png")
        print(f"PPO avg time: {np.mean(ppo_times):.2f}, SP avg time: {np.mean(sp_times):.2f}")
        # print(f"PPO avg reward: {np.mean(ppo_rewards):.2f}, SP avg reward: {np.mean(sp_rewards):.2f}")
        # Log final evaluation results to WandB
        final_eval_metrics = {
            "final_eval/ppo_avg_time": float(np.mean(ppo_times)),
            "final_eval/sp_avg_time": float(np.mean(sp_times)),
            "final_eval/ppo_vs_sp_ratio": float(np.mean(ppo_times) / np.mean(sp_times)),
            "final_eval/ppo_improvement": float((np.mean(sp_times) - np.mean(ppo_times)) / np.mean(sp_times) * 100),
            "final_eval/ppo_times": ppo_times,
            "final_eval/sp_times": sp_times,
        }
        
        wandb.log(final_eval_metrics)
        
        # Create evaluation comparison table
        eval_comparison_table = wandb.Table(columns=["Metric", "PPO", "Shortest Path", "Improvement"], data=[
            ["Average Time", float(np.mean(ppo_times)), float(np.mean(sp_times)), float((np.mean(sp_times) - np.mean(ppo_times)) / np.mean(sp_times) * 100)],
            ["Min Time", float(np.min(ppo_times)), float(np.min(sp_times)), float((np.min(sp_times) - np.min(ppo_times)) / np.min(sp_times) * 100)],
            ["Max Time", float(np.max(ppo_times)), float(np.max(sp_times)), float((np.max(sp_times) - np.max(ppo_times)) / np.max(sp_times) * 100)],
            ["Std Time", float(np.std(ppo_times)), float(np.std(sp_times)), 0.0],  # Use 0.0 instead of "N/A"
        ])
        
        wandb.log({"final_evaluation_comparison": eval_comparison_table})

# print("=== Training Complete ===")

# Finish wandb logging
wandb.finish()
