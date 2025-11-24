import jax
import jax.numpy as jnp
from jax import random as jax_random, vmap
import os
import time
import threading

# Enable JAX optimizations for GPU
jax.config.update('jax_enable_x64', False)  # Use float32 for better GPU performance
jax.config.update('jax_compilation_cache_dir', None)  # Will be set by environment

from taxi_env_utils import build_adj_and_time_matrix, make_obs_fn, load_or_compute_distance_matrix_parallel, load_or_build_graph
from taxi_env import TaxiEnv, init_env
from utils import EstimateReturnsState
from modes.context import RunContext
from modes.eval_only import run_eval_only
from modes.train_dqn import run_dqn
from modes.train_mcts import run_mcts
from modes.train_ppo import run_ppo
from modes.train_q_learning import run_q_learning

import argparse

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
    
    # Reduce compilation memory usage
    # This helps prevent LLVM from trying to allocate large contiguous memory blocks
    try:
        jax.config.update('jax_platform_name', 'cpu')  # Ensure we're using CPU backend
    except:
        pass  # Ignore if already set
    
    # Set memory preallocation (respect environment variables if set)
    if 'XLA_PYTHON_CLIENT_PREALLOCATE' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'  # Reduced to prevent OOM
    if 'XLA_PYTHON_CLIENT_ALLOCATOR' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # Use platform allocator
    
    # Configure XLA flags to avoid contiguous memory allocation issues
    # This prevents LLVM from trying to allocate large contiguous memory sections
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags_parts = xla_flags.split() if xla_flags else []
    
    # Add flags to reduce memory pressure during compilation
    # Only use valid XLA flags to avoid crashes
    flags_to_add = [
        '--xla_cpu_enable_fast_math=false',  # Disable fast math to reduce memory usage
    ]
    
    for flag in flags_to_add:
        if flag not in xla_flags_parts:
            xla_flags_parts.append(flag)
    
    os.environ['XLA_FLAGS'] = ' '.join(xla_flags_parts)
    
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
parser.add_argument("--pickup_bonus", type=float, default=5.0, help="Bonus for picking up a passenger")
parser.add_argument("--timeout_penalty", type=float, default=-5.0, help="Penalty for timeout")
parser.add_argument("--n_expert_samples", type=int, default=5000, help="Number of expert samples for pretraining")
parser.add_argument("--hidden_dims", nargs='+', type=int, default=[512, 512], help="Hidden dimensions for the neural network")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for training")
parser.add_argument("--epochs", type=int, default=100_000, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--num_steps", type=int, default=128, help="Number of steps for training")
parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor for training")
parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon for epsilon-greedy policy (Q-learning default: 1.0)")
parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final epsilon for epsilon-greedy policy (Q-learning default: 0.01)")
parser.add_argument("--epsilon_decay_fraction", type=float, default=0.5, help="Fraction of total training steps over which epsilon decays (Q-learning, default: 0.5 = 50%%)")
parser.add_argument("--sp_bias_beta", type=float, default=2.0, help="Shortest-path bias strength for exploration (higher = more SP bias)")
parser.add_argument("--params_dir", type=str, default=None, help="Output file for pretrained parameters")
parser.add_argument(
    "--q_table_path",
    type=str,
    default=None,
    help="Path to save/load tabular Q-learning state when --discrete is set",
)
parser.add_argument("--pretrain_ckpt", type=str, default="pretrained_params.pkl", help="Checkpoint file for pretrained parameters")
parser.add_argument("--model", type=str, default="ppo", choices=["dqn", "mcts", "ppo"])
parser.add_argument("--config", "-c", type=str, default="config.json", help="Path to configuration file")
parser.add_argument("--wandb_project", type=str, default="taxi-mcts", help="WandB project name")
parser.add_argument("--use_untied", type=bool, default=True, help="untied heads for Q-network")
parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for caching graph and distance data")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel distance computation")
parser.add_argument("--eval_only", action="store_true", help="Skip training and only run evaluation with saved parameters")
parser.add_argument("--load_params", type=str, default=None, help="Path to saved parameters file for evaluation-only mode")
parser.add_argument("--fixed_eval", action="store_true", help="Use fixed starts and pickups for evaluation instead of random sampling")
parser.add_argument("--fixed_starts", nargs='+', type=int, default=None, help="Fixed start node indices for evaluation (e.g., --fixed_starts 1 2 3)")
parser.add_argument("--fixed_pickups", nargs='+', type=int, default=None, help="Fixed pickup node indices for evaluation (e.g., --fixed_pickups 4 5 6)")
parser.add_argument("--plot_traveling_times", action="store_true", help="Create a plot comparing traveling times between RL and shortest path for each initial-destination pair")
parser.add_argument(
    "--all_nodes_starts_pickups",
    action="store_true",
    help="Use every node in the graph as an eligible fixed start and pickup location",
)
parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility (affects starts/pickups selection)")
parser.add_argument("--discrete", action="store_true", help="Use discrete time discretization (dt=5) and tabular Q-learning instead of PPO")
parser.add_argument("--pretrain_enabled", action="store_true", help="Enable shortest path pretraining for Q-learning (discrete mode only)")
parser.add_argument("--num_pretrain_episodes", type=int, default=10000, help="Number of pretraining episodes using shortest path rollouts (for Q-learning)")

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
    # graph_cache_file = os.path.join(args.cache_dir, "manhattan_graph_4zones_1pair.pkl")
    graph_cache_file = None
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
    # distance_cache_file = os.path.join(args.cache_dir, "manhattan_distances_4zones_1pair.pkl")
    distance_cache_file = None
else:
    distance_cache_file = os.path.join(args.cache_dir, f"simple_distances_{args.num_layers}layers_{args.offset}offset.pkl")
dist_mat, hop_dist_mat, max_length, paths_dict = load_or_compute_distance_matrix_parallel(
    G, node_to_idx, distance_cache_file, args.num_workers
)

# Place distance matrices on GPU for faster access
distances = jax.device_put(jnp.array(dist_mat, dtype=jnp.float32))
hop_distances = jax.device_put(jnp.array(hop_dist_mat, dtype=jnp.float32))
# print(f"Max shortest path: {max_length}")

# --- Compute normalized node coordinates for Euclidean distance in reward ---
from taxi_env_utils import compute_normalized_node_coordinates
node_coordinates = compute_normalized_node_coordinates(G, node_to_idx)

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
    node_coordinates=node_coordinates,  # Pass normalized coordinates for Euclidean distance
    pickup_bonus=args.pickup_bonus,
    timeout_penalty=args.timeout_penalty,
    gamma=args.gamma,   
)

# --- Observation function ---
obs_fn_single, obs_fn_batch = make_obs_fn(env, G, node_to_idx)

# --- Initialize batched environment states for metrics logging and evaluation ---
num_envs = args.num_agents
key = jax_random.PRNGKey(args.seed)
key1, key2 = jax_random.split(key)
all_keys = jax_random.split(key2, 2 * num_envs)
start_keys, pickup_keys = all_keys[:num_envs], all_keys[num_envs:]
key3, eval_key = jax_random.split(key1)

sample_start = vmap(lambda k: jax_random.choice(k, fixed_starts_idx))(start_keys)
sample_pickup = vmap(lambda k: jax_random.choice(k, fixed_pickups_idx))(pickup_keys)
start_idxs = jax.device_put(sample_start)
pickup_idxs = jax.device_put(sample_pickup)

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

# Modular early-dispatch for eval-only (keeps legacy block below unreachable)
ctx = RunContext(
    env=env,
    G=G,
    node_to_idx=node_to_idx,
    idx_to_node=idx_to_node,
    obs_fn_single=obs_fn_single,
    obs_fn_batch=obs_fn_batch,
    fixed_starts_idx=fixed_starts_idx,
    fixed_pickups_idx=fixed_pickups_idx,
    distances=distances,
    hop_distances=hop_distances,
    neighbor_mask_static=neighbor_mask_static,
    estimate_state=estimate_state,
    max_length=max_length,
    paths_dict=paths_dict,
)
if args.eval_only:
    run_eval_only(args, ctx)
    exit(0)

# Early-dispatch to modular training handlers
else:
    # If discrete mode is enabled, use Q-learning regardless of model selection
    if args.discrete:
        run_q_learning(args, ctx)
        exit(0)
    
    # Otherwise, route to the selected model
    if args.model == "dqn":
        run_dqn(args, ctx)
        exit(0)
    elif args.model == "mcts":
        run_mcts(args, ctx)
        exit(0)
    elif args.model == "ppo":
        run_ppo(args, ctx)
        exit(0)