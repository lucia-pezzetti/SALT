import networkx as nx
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jax_random
import os
import pickle
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading

# Enable JAX optimizations for GPU
jax.config.update('jax_enable_x64', False)  # Use float32 for better GPU performance
jax.config.update('jax_compilation_cache_dir', None)  # Will be set by environment
import wandb
import json

from taxi_env_utils import build_adj_and_time_matrix, make_obs_fn
from taxi_env import TaxiEnv, init_env, TaxiState
from dqn_trainer import train
from models.q_network import QNetwork, QNetworkSimple, QNetworkUntied
from utils import build_env, estimate_returns_jit, EstimateReturnsState
from brute_force import brute_force_shortest_path, build_cost_matrix
from policy_improvement import get_init_fn, get_recurrent_fn, get_agent_loop, estimate_returns_batch, VFunction

import argparse
from functools import partial
import numpy as onp
from scipy.optimize import linear_sum_assignment
import optax
import pickle

# Thread-safe caching
_cache_lock = threading.Lock()

def load_or_compute_distance_matrix_parallel(G, node_to_idx, cache_file="manhattan_distances.pkl", num_workers=None):
    """Load precomputed distance matrix or compute and cache it using parallel processing."""
    if os.path.exists(cache_file):
        # print(f"Loading precomputed distance matrix from {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            return data['dist_mat'], data['hop_dist_mat'], data['max_length']
    
    # print("Computing distance matrix with parallel processing...")
    all_nodes = list(G.nodes())
    N = len(all_nodes)
    # print(f"Computing all-pairs shortest paths for {N} nodes using {num_workers or mp.cpu_count()} workers...")
    
    dist_mat = np.zeros((N, N), dtype=np.float32)
    hop_dist_mat = np.zeros((N, N), dtype=np.float32)
    max_length = 0
    
    # Parallel computation of distance matrices
    def compute_node_distances(node_batch):
        local_dist = np.zeros((len(node_batch), N), dtype=np.float32)
        local_hop = np.zeros((len(node_batch), N), dtype=np.float32)
        local_max = 0
        
        for i, u in enumerate(node_batch):
            ui = node_to_idx[u]
            # Travel time distances
            lengths = nx.single_source_dijkstra_path_length(G, u, weight="travel_time_congested")
            for v, d in lengths.items():
                vi = node_to_idx[v]
                local_dist[i, vi] = d
            
            # Hop distances
            hop_lengths = nx.single_source_dijkstra_path_length(G, u, weight=None)
            for v, d in hop_lengths.items():
                vi = node_to_idx[v]
                local_hop[i, vi] = d
                if d > local_max:
                    local_max = d
        
        return local_dist, local_hop, local_max
    
    # Split nodes into batches for parallel processing
    if num_workers is None:
        num_workers = min(mp.cpu_count(), N)
    
    batch_size = max(1, N // num_workers)
    node_batches = [all_nodes[i:i + batch_size] for i in range(0, N, batch_size)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(compute_node_distances, node_batches))
    
    # Combine results
    for i, (local_dist, local_hop, local_max) in enumerate(results):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, N)
        dist_mat[start_idx:end_idx] = local_dist
        hop_dist_mat[start_idx:end_idx] = local_hop
        max_length = max(max_length, local_max)
    
    # Cache the results
    # print(f"Caching distance matrix to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'dist_mat': dist_mat,
            'hop_dist_mat': hop_dist_mat,
            'max_length': max_length
        }, f)
    
    return dist_mat, hop_dist_mat, max_length

def load_or_build_graph(args, cache_file="manhattan_graph.pkl"):
    """Load cached graph or build and cache it."""
    if os.path.exists(cache_file):
        # print(f"Loading cached Manhattan graph from {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            return data['G'], data['node_to_idx'], data['idx_to_node'], data['fixed_starts_idx'], data['fixed_pickups_idx'], data['traffic_params']
    
    # print("Building Manhattan graph (this may take a while on first run)...")
    start_time = time.time()
    
    G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params = build_env(args)
    
    build_time = time.time() - start_time
    # print(f"Graph built in {build_time:.2f} seconds")
    
    # Cache the graph
    # print(f"Caching graph to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'G': G,
            'node_to_idx': node_to_idx,
            'idx_to_node': idx_to_node,
            'fixed_starts_idx': fixed_starts_idx,
            'fixed_pickups_idx': fixed_pickups_idx,
            'traffic_params': traffic_params
        }, f)
    
    return G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params

def optimize_jax_config():
    """Set optimal JAX configuration for performance."""
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
parser.add_argument("--cycle_length", type=int, default=0, help="Cycle length for the customised grid environment")
parser.add_argument("--no_congestion", type=bool, default=False, help="different types of roads have different congestion levels")
parser.add_argument("--place_name", type=str, default="Manhattan, New York City, New York, USA", help="Place name for the graph (used for Manhattan)")
parser.add_argument("--zone_shp", type=str, default="../data/processed/taxi_zones.shp", help="Path to the shapefile for zones (used for Manhattan)")
parser.add_argument("--num_agents", type=int, default=1, help="Number of agents in the environment")
parser.add_argument("--base_time", type=float, default=1.0, help="Base travel time for grid environment")
parser.add_argument("--max_steps", type=int, default=128, help="Maximum number of steps per episode")
parser.add_argument("--pickup_bonus", type=float, default=10.0, help="Bonus for picking up a passenger")
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
parser.add_argument("--params_dir", type=str, default=None, help="Output file for pretrained parameters")
parser.add_argument("--pretrain_ckpt", type=str, default="pretrained_params.pkl", help="Checkpoint file for pretrained parameters")
parser.add_argument("--model", type=str, default="pi", choices=["dqn", "pi"])
parser.add_argument("--pi_config", "-c", type=str, default="config.json", help="Path to pi configuration file")
parser.add_argument("--wandb_project", type=str, default="taxi-pi", help="WandB project name")
parser.add_argument("--use_untied", type=bool, default=True, help="untied heads for Q-network")
parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for caching graph and distance data")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel distance computation")
parser.add_argument("--eval_only", action="store_true", help="Skip training and only run evaluation with saved parameters")
parser.add_argument("--load_params", type=str, default=None, help="Path to saved parameters file for evaluation-only mode")
parser.add_argument("--fixed_eval", action="store_true", help="Use fixed starts and pickups for evaluation instead of random sampling")
parser.add_argument("--fixed_starts", nargs='+', type=int, default=None, help="Fixed start node indices for evaluation (e.g., --fixed_starts 1 2 3)")
parser.add_argument("--fixed_pickups", nargs='+', type=int, default=None, help="Fixed pickup node indices for evaluation (e.g., --fixed_pickups 4 5 6)")

args = parser.parse_args()

# Validate fixed evaluation parameters
if args.fixed_eval:
    if args.fixed_starts is None or args.fixed_pickups is None:
        raise ValueError("When using --fixed_eval, both --fixed_starts and --fixed_pickups must be provided")
    if len(args.fixed_starts) != len(args.fixed_pickups):
        raise ValueError("--fixed_starts and --fixed_pickups must have the same length")
    print(f"Fixed evaluation mode: starts={args.fixed_starts}, pickups={args.fixed_pickups}")

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
    graph_cache_file = os.path.join(args.cache_dir, "manhattan_graph_4zones.pkl")
else:
    graph_cache_file = os.path.join(args.cache_dir, f"simple_graph_{args.num_layers}layers_{args.offset}offset.pkl")
G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params = load_or_build_graph(args, graph_cache_file)

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
    distance_cache_file = os.path.join(args.cache_dir, "manhattan_distances_4zones.pkl")
else:
    distance_cache_file = os.path.join(args.cache_dir, f"simple_distances_{args.num_layers}layers_{args.offset}offset.pkl")
dist_mat, hop_dist_mat, max_length = load_or_compute_distance_matrix_parallel(
    G, node_to_idx, distance_cache_file, args.num_workers
)

# Place distance matrices on GPU for faster access
distances = jax.device_put(jnp.array(dist_mat, dtype=jnp.float32))
hop_distances = jax.device_put(jnp.array(hop_dist_mat, dtype=jnp.float32))
# print(f"Max shortest path: {max_length}")

# --- Create environment ---
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

print(f"Start idxs: {start_idxs}")
print(f"Pickup idxs: {pickup_idxs}")

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

# print("=== Starting Training ===")

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
            
            # Simple evaluation
            times = []
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
                
                times.append(total_time)
            
            print(f"Run {run+1}: DQN avg time: {np.mean(times):.2f}, Total steps across all agents: {total_steps_all_agents}")
            print("="*50)
    
    elif model_type == 'pi':
        print("Setting up value function...")
        
        config_file = args.pi_config if hasattr(args, 'pi_config') else "config.json"
        if not os.path.exists(config_file):
            print(f"ERROR: Config file {config_file} not found. Cannot recreate the exact model architecture.")
            print("Please ensure the config file exists and matches the training configuration.")
            exit(1)
        else:
            import json
            with open(config_file, "r") as f:
                config = json.load(f)
        
        print(f"Using config: {config}")
        
        # Create V_apply function
        V_net = hk.without_apply_rng(hk.transform(lambda obs: VFunction(config)(obs)))
        
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
            """Shortest path policy"""
            curr = state.current_node
            pickup = state.pickup_node
            neighbors = env.adj_list[curr]
            dists = env.distances[neighbors, pickup]
            mask = state.neighbor_mask
            dists = jnp.where(mask, dists, jnp.inf)
            return int(jnp.argmin(dists))

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

        # Evaluating learned policy
            
        if args.fixed_eval:
            # Use fixed starts and pickups
            eval_starts = jnp.array(args.fixed_starts)
            rl_pickups = jnp.array(args.fixed_pickups)
            sp_pickups = jnp.array(args.fixed_pickups)
            print(f"Using fixed evaluation: starts={eval_starts}, pickups={rl_pickups}")
            key, eval_key = jax.random.split(key)
            init_keys = jax.random.split(eval_key, B**2)
            print_trajectories = True

            v_time = evaluate_policy(greedy_V_policy, eval_starts, rl_pickups, print_trajectories=print_trajectories, run_id=run_id)
            sp_time = evaluate_policy(sp_policy, eval_starts, sp_pickups, print_trajectories=print_trajectories, run_id=run_id)

            print(f"Value Policy: starts - {eval_starts}, pickups - {rl_pickups}")
            print(f"SP Policy: starts - {eval_starts}, pickups - {sp_pickups}")

            print(f"Value Policy   avg time: {v_time.mean():.2f}")
            print(f"SP Policy      avg time: {sp_time.mean():.2f}")
        else:
            for run_id in range(20):
                B = args.num_agents
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

                print(f"Returns matrix: {returns_matrix}")
            
                # Solve optimal transport problem
                _, rl_assignment = optax.assignment.hungarian_algorithm(-returns_matrix)
                rl_pickups = eval_pickups[rl_assignment]

                # print(f"Value Policy: starts - {eval_starts}, pickups - {rl_pickups}")

                # Print trajectories for first few runs
                print_trajectories = run_id < 3
                # Keep distance computation on GPU
                D = distances[eval_starts, :][:, eval_pickups]  # Direct JAX indexing
                print(f"Distance matrix: {D}")
                _, sp_col = optax.assignment.hungarian_algorithm(D)
                sp_pickups = eval_pickups[sp_col]
                # print(f"SP Policy: starts - {eval_starts}, pickups - {sp_pickups}")

                v_time = evaluate_policy(greedy_V_policy, eval_starts, rl_pickups, print_trajectories=print_trajectories, run_id=run_id)
                sp_time = evaluate_policy(sp_policy, eval_starts, sp_pickups, print_trajectories=print_trajectories, run_id=run_id)

                print(f"Value Policy: starts - {eval_starts}, pickups - {rl_pickups}")
                print(f"SP Policy: starts - {eval_starts}, pickups - {sp_pickups}")

                print(f"Value Policy   avg time: {v_time.mean():.2f}")
                print(f"SP Policy      avg time: {sp_time.mean():.2f}")
    
    print("=== Evaluation Complete ===")
    exit(0)

# Initialize wandb for logging
wandb.init(
    project="ride-sharing-optimizedß",
    name=f"{args.env_type}-env-{args.epochs}epochs-{args.num_agents}agents",
    config={
        "env_type": args.env_type,
        "epochs": args.epochs,
        "num_agents": args.num_agents,
        "model": args.model,
        "num_nodes": env.num_nodes,
        "max_deg": env.max_deg,
        "cache_dir": args.cache_dir,
        "num_workers": args.num_workers,
    }
)

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

    # shortest-path greedy policy
    def sp_policy(G, state: TaxiState) -> int:
        curr = idx_to_node[int(state.current_node)]
        goal = idx_to_node[int(state.pickup_node)]
        path = nx.shortest_path(G, curr, goal, weight='travel_time_congested')
        return path

    # Rollout helper
    def eval_matching(starts, pickups, policy, G, node_to_idx) -> onp.ndarray:
        times = []
        for s, p in zip(onp.array(starts), onp.array(pickups)):
            # initialize a fresh single‐env state
            state = init_env(jax.random.PRNGKey(0), int(s), int(p), env.neighbor_mask_static)[0]
            total = 0.0
            if hasattr(policy, "__call__") and policy is agent_policy:
                while not bool(state.done):
                    a = policy(state)
                    state, _, _, info = env.step(state, a)
                    total += float(info['travel'] + info['wait'])
            else:
                # use the SP policy
                path = policy(G, state)
                for u in path[1:]:
                    a_idx = node_to_idx[u]
                    # find the action to take to get to that node
                    nbrs = env.adj_list[state.current_node]
                    poss = jnp.where(nbrs == a_idx, size=1)[0]
                    action = int(poss[0])
                    # step and accumulate true (travel + wait)
                    state, _, _, info = env.step(state, action)
                    total += float(info['travel'] + info['wait'])
            times.append(total)
        return onp.array(times)

    # Evaluating agent policy against SP baseline
    for _ in range(20):
        B = args.num_agents

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

        # Compute the RL‐matching via estimated returns
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
        )  # shape [B, B]
        R = jnp.array(R, dtype=jnp.float32)

        # solve the optimal transport problem
        _, rl_col = optax.assignment.hungarian_algorithm(-R)
        rl_pickups = pickups[rl_col]

        # SP‐matching via pure SP distances
        D = distances[starts, :][:, pickups]
        _, sp_col = optax.assignment.hungarian_algorithm(D)
        sp_pickups = pickups[sp_col]

        rl_on_rl_times = eval_matching(starts,   rl_pickups, agent_policy, G, node_to_idx)
        sp_on_sp_times = eval_matching(starts,   sp_pickups, sp_policy, G, node_to_idx)

        # Matching & Policy Evaluation
        print(f"RL matching + RL policy avg time: {rl_on_rl_times.mean():.2f}")
        print(f"SP matching + SP policy avg time: {sp_on_sp_times.mean():.2f}")

# --- Policy Improvement ---
elif args.model == "pi":
    obs_fn_single, obs_fn_batch = make_obs_fn(env, G, node_to_idx)
    # init
    if args.pi_config is None:
        raise ValueError("Must pass --pi_config path to your JSON for policy improvement")
    with open(args.pi_config, "r") as f:
        config = json.load(f)

    config['batch_size'] = args.num_agents
    config['num_steps'] = args.epochs * config['eval_frequency']
    # Cap simulations to prevent excessive tree search on large graphs
    config['num_simulations'] = min(2*max_length, 64)

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
    agent_loop = get_agent_loop(env, config, obs_fn_batch, V_apply, recurrent_fn, V_opt_update, get_V_params, epsilon_schedule)

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
        'loss': jnp.array(0.0),
        'episode_travel': jnp.zeros(config['batch_size']),
        'episode_wait': jnp.zeros(config['batch_size']),
        'episode_total_time': jnp.zeros(config['batch_size']),
        'avg_wait': jnp.zeros(config['batch_size']),
        'avg_travel': jnp.zeros(config['batch_size']),
        'avg_total_time': jnp.zeros(config['batch_size']),
    }

    avg_returns, times = [], []
    
    num_eval_steps = config['num_steps'] // config['eval_frequency']
    
    for i in range(num_eval_steps):
        step_start_time = time.time()
        state_dict, metrics = agent_loop(state_dict)
        step_time = time.time() - step_start_time
        
        # Evaluate shortest path policy for baseline comparison
        def sp_policy(state: TaxiState) -> int:
            """Shortest path policy"""
            curr = state.current_node
            pickup = state.pickup_node
            neighbors = env.adj_list[curr]
            dists = env.distances[neighbors, pickup]
            mask = state.neighbor_mask
            dists = jnp.where(mask, dists, jnp.inf)
            return int(jnp.argmin(dists))

        def evaluate_sp_policy(starts, pickups):
            """Evaluate shortest path policy on given start/pickup pairs"""
            sp_times = []
            for start, pickup in zip(starts, pickups):
                total_time = 0.0
                state = init_env(jax.random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
                step_count = 0
                while not state.done and step_count < env.max_steps:
                    action = sp_policy(state)
                    state, _, _, info = env.step(state, action)
                    total_time += float(info['travel'] + info['wait'])
                    step_count += 1
                
                if not state.done:
                    print(f"Warning: SP episode didn't complete after {env.max_steps} steps")
                    print(f"Start: {start}, Pickup: {pickup}, Final node: {state.current_node}")
                
                sp_times.append(total_time)
            return np.array(sp_times)

        # Evaluate SP policy on current starts/pickups
        starts = np.array(metrics['starts'])
        pickups = np.array(metrics['pickups'])
        sp_times = evaluate_sp_policy(starts, pickups)
        avg_sp_total_time = np.mean(sp_times)
        
        # Calculate performance ratio (learned policy / shortest path)
        if avg_sp_total_time > 0:
            performance_ratio = float(metrics['avg_total_time']) / float(avg_sp_total_time)
        else:
            performance_ratio = float('inf')  # SP evaluation failed
            print(f"Warning: SP evaluation returned zero time at step {i}")
        
        # Log metrics to wandb
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
            'training/visit_freq_entropy': float(-jnp.sum(metrics['visit_freq'] * jnp.log(metrics['visit_freq'] + 1e-8))),
            'training/opt_step': int(state_dict['opt_t']),
        }
        
        # Log additional state metrics
        if 'episode_return' in state_dict:
            log_metrics['training/episode_return'] = float(jnp.mean(state_dict['episode_return']))
        if 'num_episodes' in state_dict:
            log_metrics['training/num_episodes'] = float(jnp.sum(state_dict['num_episodes']))
        if 'visit_counts' in state_dict:
            log_metrics['training/total_visits'] = float(jnp.sum(state_dict['visit_counts']))
            log_metrics['training/unique_nodes_visited'] = float(jnp.sum(state_dict['visit_counts'] > 0))
        
        wandb.log(log_metrics, step=i)
        
        state_dict.update({
            'episode_return': jnp.zeros(config['batch_size']),
            'avg_travel': jnp.zeros(config['batch_size']),
            'avg_wait': jnp.zeros(config['batch_size']),
            'avg_return': jnp.zeros(config['batch_size']),
            'num_episodes': jnp.zeros(config['batch_size']),
            'episode_travel': jnp.zeros(config['batch_size']),
            'episode_wait': jnp.zeros(config['batch_size']),
            'visit_counts': jnp.zeros(env.num_nodes, dtype=jnp.int32),
            'loss': jnp.array(0.0),
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
        """Shortest path policy"""
        curr = state.current_node
        pickup = state.pickup_node
        neighbors = env.adj_list[curr]
        dists = env.distances[neighbors, pickup]
        mask = state.neighbor_mask
        dists = jnp.where(mask, dists, jnp.inf)
        return int(jnp.argmin(dists))

    def evaluate_policy(policy, starts, pickups, max_steps=100):
        """Evaluate a policy on a given initial state"""
        times = []
        for start, pickup in zip(starts, pickups):
            total_time = 0.0
            step_count = 0
            # Initialize a fresh single‐env state
            state = init_env(jax.random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
            while not state.done and step_count < max_steps:
                action = policy(state)
                state, _, _, info = env.step(state, action)
                total_time += float(info['travel'] + info['wait'])
                step_count += 1
            times.append(total_time)
        
        return onp.array(times)

    # Evaluating learned policy
    for _ in range(20):
        B = args.num_agents
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

        # print(f"Value Policy: starts - {eval_starts}, pickups - {rl_pickups}")

        v_time = evaluate_policy(greedy_V_policy, eval_starts, rl_pickups)

        D = distances[eval_starts, :][:, eval_pickups]  # Direct JAX indexing
        _, sp_col = optax.assignment.hungarian_algorithm(D)
        sp_pickups = eval_pickups[sp_col]
        # print(f"SP Policy: starts - {eval_starts}, pickups - {sp_pickups}")

        sp_time = evaluate_policy(sp_policy, eval_starts, sp_pickups)

        print(f"Value Policy   avg time: {v_time.mean():.2f}")
        print(f"SP Policy      avg time: {sp_time.mean():.2f}")

# Finish wandb logging
wandb.finish()
