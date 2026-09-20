import jax
import jax.numpy as jnp
from jax import random as jax_random, vmap
import json
import os
import subprocess
import sys
import time
import threading

jax.config.update('jax_enable_x64', False)
jax.config.update('jax_compilation_cache_dir', None)

from taxi_env_utils import apply_minimum_edge_travel_time, build_adj_and_time_matrix, make_obs_fn, load_or_compute_distance_matrix_parallel, load_or_build_graph, build_noise_mask
from taxi_env import TaxiEnv, init_env, pickup_bonus_reward_from_seconds
from utils import EstimateReturnsState, load_graph
from modes.context import RunContext

import argparse

# Thread-safe caching
_cache_lock = threading.Lock()


def _json_safe(value):
    """Convert small runtime metadata values into JSON-serializable objects."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _git_value(*args):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=os.getcwd(),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def print_run_trace(args):
    """Print the effective runtime configuration into the redirected training log."""
    git_status = _git_value("status", "--short")
    metadata = {
        "argv": sys.argv,
        "working_directory": os.path.basename(os.getcwd()),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "git_status_short": git_status,
        "python_version": sys.version.split()[0],
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "environment": {
            key: os.environ.get(key)
            for key in [
                "CONDA_DEFAULT_ENV",
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "JAX_COMPILATION_CACHE_DIR",
                "JAX_COMPILATION_CACHE_BASE",
                "WANDB_MODE",
                "WANDB_PROJECT",
            ]
        },
        "args": vars(args),
    }
    print("=== Run Trace ===")
    print(json.dumps(_json_safe(metadata), indent=2, sort_keys=True))
    print("=== End Run Trace ===")


def optimize_jax_config():
    """Set optimal JAX configuration"""
    # Use environment variable for cache dir if set, otherwise create one
    cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', f"/tmp/jax_cache_{os.getpid()}")
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', cache_dir)
    
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_enable_compilation_cache', True)
    
    # Set memory preallocation (respect environment variables if set)
    if 'XLA_PYTHON_CLIENT_PREALLOCATE' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
    if 'XLA_PYTHON_CLIENT_ALLOCATOR' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    # Configure XLA flags to avoid contiguous memory allocation issues
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags_parts = xla_flags.split() if xla_flags else []
    
    # Add flags to reduce memory pressure during compilation
    flags_to_add = [
        # '--xla_cpu_enable_fast_math=false',  # CPU-specific, disabled for GPU
    ]
    
    for flag in flags_to_add:
        if flag not in xla_flags_parts:
            xla_flags_parts.append(flag)
    
    os.environ['XLA_FLAGS'] = ' '.join(xla_flags_parts)
    

parser = argparse.ArgumentParser(description="Discrete tabular Q-learning ride-sharing simulator")
parser.add_argument("--env_type", type=str, choices=["manhattan", "simple"], default="manhattan", help="Type of environment to use")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the customised grid environment")
parser.add_argument("--layer_width", type=int, default=3, help="Width of each layer in the customised grid environment")
parser.add_argument("--offset", type=float, default=0.0, help="Offset for the customised grid environment")
parser.add_argument("--random_offsets", action="store_true", help="Use random offsets for each traffic light instead of uniform offset")
parser.add_argument("--cycle_length", type=int, default=90, help="Cycle length for the customised grid environment")
parser.add_argument("--no_congestion", type=bool, default=False, help="different types of roads have different congestion levels")
parser.add_argument("--noise", action="store_true", help="Inject random congestion noise on primary/secondary roads (per-edge random multiplier on travel times)")
parser.add_argument("--noise_level", type=float, default=0.2, help="Noise magnitude: primary roads get U(1, 1+level), secondary get U(1, 1+1.5*level). Default 0.2")
parser.add_argument("--place_name", type=str, default="Manhattan, New York City, New York, USA", help="Place name for the graph (used for Manhattan)")
parser.add_argument("--zone_shp", type=str, default="../data/processed/taxi_zones.shp", help="Path to the shapefile for zones (used for Manhattan)")
parser.add_argument(
    "--manhattan_area",
    nargs="+",
    default=["south_manhattan"],
    help=(
        "Manhattan area preset or explicit zone names. Presets: south_manhattan, "
        "small_manhattan_area, upper_east_side_small. For explicit zones, pass "
        "quoted names, e.g. --manhattan_area 'Upper East Side North' 'Yorkville West'."
    ),
)
parser.add_argument("--num_agents", type=int, default=1, help="Number of agents in the environment")
parser.add_argument("--base_time", type=float, default=1.0, help="Base travel time for grid environment")
parser.add_argument("--max_steps", type=int, default=300, help="Maximum number of steps per episode")
parser.add_argument("--pickup_bonus", type=float, default=50.0, help="Bonus for picking up a passenger")
parser.add_argument(
    "--pickup_bonus_seconds",
    type=float,
    default=None,
    help=(
        "Pickup bonus expressed as seconds of avoided travel. It is divided by 60 "
        "to match the reward units and overrides --pickup_bonus when provided."
    ),
)
parser.add_argument(
    "--timeout_penalty",
    type=float,
    default=None,
    help="Deprecated and ignored. Horizon truncations use negative shortest-path remaining travel time.",
)
parser.add_argument("--n_expert_samples", type=int, default=5000, help="Number of expert samples for pretraining")
parser.add_argument("--hidden_dims", nargs='+', type=int, default=[512, 512], help="Hidden dimensions for the neural network")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for training")
parser.add_argument("--epochs", type=int, default=100_000, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--num_steps", type=int, default=128, help="Number of steps for training")
parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor for training")
parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon for epsilon-greedy policy (Q-learning default: 1.0)")
parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final epsilon for epsilon-greedy policy (Q-learning default: 0.01)")
parser.add_argument("--epsilon_decay_fraction", type=float, default=0.9, help="Fraction of total training steps over which epsilon decays (Q-learning, default: 0.5 = 50%%)")
parser.add_argument("--sp_bias_beta", type=float, default=2.0, help="Shortest-path bias strength for exploration (higher = more SP bias)")
parser.add_argument("--params_dir", type=str, default=None, help="Output file for pretrained parameters")
parser.add_argument(
    "--q_table_path",
    type=str,
    default=None,
    help="Path to save/load tabular Q-learning state when --discrete is set",
)
parser.add_argument(
    "--init_q_table_path",
    type=str,
    default=None,
    help="Path to a saved Q-table file to use for initialization (discrete mode only). The Q-table will be loaded to initialize the agent, but hyperparameters will be taken from current arguments.",
)
parser.add_argument("--pretrain_ckpt", type=str, default="pretrained_params.pkl", help=argparse.SUPPRESS)
parser.add_argument("--config", "-c", type=str, default="config.json", help=argparse.SUPPRESS)
parser.add_argument("--wandb_project", type=str, default="ride-sharing-q-learning", help="WandB project name")
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
parser.add_argument(
    "--start_zones",
    nargs='+',
    default=None,
    help="One or more zones to use for start points only. Can be LocationIDs (integers) or zone names (strings). Example: --start_zones 100 or --start_zones 'Upper East Side North' 'Yorkville West'. If not provided, all nodes are used. If only --start_zones is provided, all nodes are used for pickups.",
)
parser.add_argument(
    "--pickup_zones",
    nargs='+',
    default=None,
    help="One or more zones to use for pickup points only. Can be LocationIDs (integers) or zone names (strings). Example: --pickup_zones 200 or --pickup_zones 'Lenox Hill West' 'Upper East Side South'. If not provided, all nodes are used. If only --pickup_zones is provided, all nodes are used for starts.",
)
parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility (affects starts/pickups selection)")
parser.add_argument("--discrete", action="store_true", help="Use discrete time discretization and tabular Q-learning instead of PPO")
parser.add_argument("--dt", type=float, default=1.0, help="Time discretization step in seconds (default: 1.0, used when --discrete is set)")
parser.add_argument(
    "--min_edge_travel_time_seconds",
    type=float,
    default=0.0,
    help=(
        "Floor each physical edge travel time to this many seconds before building "
        "runtime and shortest-path data. Zero preserves the original edge times."
    ),
)
parser.add_argument("--q_table_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Q-table dtype for discrete Q-learning. Use float16/bfloat16 to reduce memory usage.")
parser.add_argument("--pretrain_enabled", action="store_true", help="Enable shortest path pretraining for Q-learning (discrete mode only)")
parser.add_argument("--num_pretrain_episodes", type=int, default=10000, help="Number of pretraining episodes using shortest path rollouts (for Q-learning)")
parser.add_argument("--pretrain_learning_rate", type=float, default=None, help="Learning rate for pretraining (default: None = use agent's LR). Recommended: 0.01-0.05 when using --init_from_shortest_paths")
parser.add_argument("--init_from_shortest_paths", action="store_true", help="Initialize Q-table from shortest path travel times (discrete mode only)")
parser.add_argument("--init_all_time_slices", action="store_true", help="Initialize Q-table for all time slices (up to 100) instead of just time=0 (requires --init_from_shortest_paths)")
parser.add_argument("--eval_only_sp", action="store_true", help="Run shortest-path baselines (continuous + discrete) and exit (discrete/Q-learning pipeline)")
parser.add_argument("--eval_frequency", type=int, default=100, help="Q-learning evaluation frequency in episodes")
parser.add_argument("--checkpoint_frequency", type=int, default=0, help="Save the Q-table every N training episodes. Use 0 to save only at the end.")
parser.add_argument("--episode_offset", type=int, default=0, help="Number of already-completed episodes when resuming chunked training; used for epsilon/matching schedules.")
parser.add_argument("--total_epochs_for_schedule", type=int, default=None, help="Total intended episodes across all chunks, used to keep epsilon/matching schedules consistent while running shorter chunks.")
parser.add_argument("--skip_final_eval", action="store_true", help="Skip the expensive final evaluation block. Useful for intermediate chunks that only need to checkpoint the Q-table.")
parser.add_argument("--read_only_q_table", action="store_true", help="Load --q_table_path without saving it back. Useful for concurrent evaluation-only jobs.")
parser.add_argument("--final_eval_iterations", type=int, default=50, help="Number of final trajectory-evaluation instances per seed. Figure 4 in the paper used 500 per seed.")
parser.add_argument("--eval_seed", type=int, default=None, help="Optional RNG seed for final evaluation. Defaults to --seed.")
parser.add_argument("--print_eval_trajectories", action="store_true", help="Print per-agent final-evaluation trajectories to the log. Intended for small evaluation-only sanity checks.")
parser.add_argument("--print_q_cycle_diagnostics", action="store_true", help="For failed learned-Q trajectories, print compact diagnostics for long two-node cycles (requires --print_eval_trajectories).")
parser.add_argument("--sample_starts_from_three_fixed", action="store_true", help="Sample starting nodes with repetition from three fixed nodes (chosen at start of training and kept fixed). Pickups still sampled from all nodes.")
parser.add_argument("--sample_pickups_from_three_fixed", action="store_true", help="Sample pickup nodes with repetition from three fixed nodes (chosen at start of training and kept fixed). Starts sampled uniformly from all nodes.")
parser.add_argument("--three_fixed_selection_method", type=str, default="random", choices=["random", "degree", "closeness", "betweenness"], help="Method to select the 3 fixed nodes: 'random' (default), 'degree' (degree centrality), 'closeness' (closeness centrality), 'betweenness' (betweenness centrality)")
parser.add_argument("--no_round_trip", action="store_true", help="Disable return trips in Q-learning training. Only forward trips (start→pickup) are run; the return leg (pickup→start) is skipped.")
parser.add_argument("--eval_assignment_baselines", action="store_true", help="At final evaluation, also report two lightweight dispatch baselines that reuse the SAME learned routing policy but replace SALT's optimal-transport assignment: (1) random assignment, and (2) myopic nominal-shortest-path (static-distance Hungarian) assignment. Isolates the value of the OT layer.")
parser.add_argument("--eval_reassignment_baselines", action="store_true", help="At final evaluation, also report shortest-path routing baselines that RE-solve the agent-target Hungarian assignment every K timesteps on offline distances (receding-horizon dispatch), for each K in --reassignment_periods. Contrasts with the static shortest-path baseline (assign once at t=0).")
parser.add_argument("--reassignment_periods", type=str, default="5,10", help="Comma-separated reassignment intervals K (in timesteps) for --eval_reassignment_baselines.")

args = parser.parse_args()

if args.timeout_penalty is not None:
    print(
        "Warning: --timeout_penalty is deprecated and ignored; horizon truncations "
        "use negative nominal shortest-path remaining travel time."
    )
if args.min_edge_travel_time_seconds < 0:
    raise ValueError("--min_edge_travel_time_seconds must be non-negative")
if args.pickup_bonus_seconds is not None:
    args.pickup_bonus = pickup_bonus_reward_from_seconds(args.pickup_bonus_seconds)
if args.checkpoint_frequency < 0:
    raise ValueError("--checkpoint_frequency must be non-negative")
if args.episode_offset < 0:
    raise ValueError("--episode_offset must be non-negative")
if args.total_epochs_for_schedule is not None and args.total_epochs_for_schedule <= 0:
    raise ValueError("--total_epochs_for_schedule must be positive when provided")
if args.final_eval_iterations <= 0:
    raise ValueError("--final_eval_iterations must be positive")
if args.read_only_q_table and args.epochs != 0:
    raise ValueError("--read_only_q_table is intended for evaluation-only runs with --epochs 0")
if args.print_q_cycle_diagnostics and not args.print_eval_trajectories:
    raise ValueError("--print_q_cycle_diagnostics requires --print_eval_trajectories")
if args.total_epochs_for_schedule is not None and args.total_epochs_for_schedule < args.episode_offset + args.epochs:
    print(
        "Warning: --total_epochs_for_schedule is smaller than "
        "--episode_offset + --epochs; epsilon/matching schedules will clamp at the end."
    )

# Validate fixed evaluation parameters
if args.fixed_eval:
    if args.fixed_starts is None or args.fixed_pickups is None:
        raise ValueError("When using --fixed_eval, both --fixed_starts and --fixed_pickups must be provided")
    if len(args.fixed_starts) != len(args.fixed_pickups):
        raise ValueError("--fixed_starts and --fixed_pickups must have the same length")

# Convert string numbers to integers if they look like numbers
def convert_zone(zone):
    """Convert zone to appropriate type (int if numeric, str otherwise)."""
    if isinstance(zone, str):
        try:
            return int(zone)
        except ValueError:
            return zone
    return zone

# Validate and convert zones if provided
if args.start_zones is not None:
    if len(args.start_zones) < 1:
        raise ValueError("--start_zones must contain at least 1 zone (LocationIDs or zone names)")
    args.start_zones = [convert_zone(z) for z in args.start_zones]

if args.pickup_zones is not None:
    if len(args.pickup_zones) < 1:
        raise ValueError("--pickup_zones must contain at least 1 zone (LocationIDs or zone names)")
    args.pickup_zones = [convert_zone(z) for z in args.pickup_zones]

# Optimize JAX configuration
optimize_jax_config()
print_run_trace(args)

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
        "_modelq_learning"
        ".pkl"
    )

# --- Load or build the environment with caching ---
if args.env_type == "manhattan":
    graph_cache_file = None
else:
    graph_cache_file = os.path.join(args.cache_dir, f"simple_graph_{args.num_layers}layers_{args.offset}offset.pkl")
G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params = load_or_build_graph(args, graph_cache_file)

floored_edge_count = apply_minimum_edge_travel_time(
    G,
    args.min_edge_travel_time_seconds,
)

print(f"Graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
if args.min_edge_travel_time_seconds > 0:
    print(
        "[TRAVEL TIME FLOOR] "
        f"minimum={args.min_edge_travel_time_seconds:g}s, "
        f"floored_edge_records={floored_edge_count}, dt={args.dt:g}s"
    )
else:
    print("[TRAVEL TIME FLOOR] disabled; original edge travel times retained")
if args.pickup_bonus_seconds is not None:
    print(
        "[PICKUP BONUS] "
        f"{args.pickup_bonus_seconds:g}s equivalent = {args.pickup_bonus:.6f} reward units"
    )
print(f"starts: {fixed_starts_idx}, pickups: {fixed_pickups_idx}")

# --- Build graph structures ---
start_time = time.time()
adj_list, travel_times, neighbor_mask_static = build_adj_and_time_matrix(
    G, node_to_idx=node_to_idx
)

# Build per-edge noise mask for dynamic (per-step) congestion noise
if getattr(args, 'noise', False):
    noise_mask = build_noise_mask(
        G, node_to_idx,
        noise_level=getattr(args, 'noise_level', 0.2),
        max_deg=adj_list.shape[1],
    )
else:
    noise_mask = None  # No noise — fully deterministic travel times

# Place graph structures on device
adj_list = jax.device_put(jnp.array(adj_list, dtype=jnp.int32))
travel_times = jax.device_put(jnp.array(travel_times, dtype=jnp.float32))
neighbor_mask_static = jax.device_put(jnp.array(neighbor_mask_static, dtype=bool))

# Precompute shortest-path distance matrix
if args.env_type == "manhattan":
    distance_cache_file = None
else:
    distance_cache_file = os.path.join(args.cache_dir, f"simple_distances_{args.num_layers}layers_{args.offset}offset.pkl")
dist_mat, hop_dist_mat, max_length, paths_dict = load_or_compute_distance_matrix_parallel(
    G, node_to_idx, distance_cache_file, args.num_workers
)

# Place distance matrices on device
distances = jax.device_put(jnp.array(dist_mat, dtype=jnp.float32))
hop_distances = jax.device_put(jnp.array(hop_dist_mat, dtype=jnp.float32))

# Validate flags
if getattr(args, 'sample_starts_from_three_fixed', False) and getattr(args, 'sample_pickups_from_three_fixed', False):
    raise ValueError("Cannot use both --sample_starts_from_three_fixed and --sample_pickups_from_three_fixed at the same time. Use only one.")

# Helper function to select 3 fixed nodes using the specified method
def select_three_fixed_nodes(G, node_to_idx, args, node_to_zone=None, target="starts"):
    """
    Select 3 fixed nodes using the three_fixed_selection_method.
    Returns: list of 3 node indices
    """
    import networkx as nx
    import pickle
    from utils import load_graph
    
    all_nodes_list = list(node_to_idx.keys())
    if len(all_nodes_list) < 3:
        raise ValueError(f"Not enough nodes in graph ({len(all_nodes_list)}). Need at least 3 nodes for --sample_{target}_from_three_fixed.")
    
    # Get node_to_zone mapping if not provided
    if node_to_zone is None and args.env_type == "manhattan":
        # Try to get from cache first
        potential_cache_files = [
            os.path.join(args.cache_dir, "manhattan_graph.pkl"),
            os.path.join(args.cache_dir, f"manhattan_graph_{args.place_name.replace(' ', '_').replace(',', '')}.pkl"),
        ]
        
        for cache_file in potential_cache_files:
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                        node_to_zone = data.get('node_to_zone')
                        if node_to_zone is not None:
                            graph_nodes_set = set(all_nodes_list)
                            node_to_zone = {node: zone for node, zone in node_to_zone.items() if node in graph_nodes_set}
                            print(f"Loaded zone mapping from cache: {len(node_to_zone)} nodes mapped to zones")
                            break
                except Exception as e:
                    print(f"Could not load zone mapping from {cache_file}: {e}")
                    continue
        
        # If not in cache, recompute it
        if node_to_zone is None:
            print("Loading zone mapping to ensure nodes from different zones...")
            try:
                sample_node = list(G.nodes())[0] if len(G.nodes()) > 0 else None
                if sample_node and 'zone' in G.nodes[sample_node]:
                    node_to_zone = {node: G.nodes[node].get('zone') for node in all_nodes_list if 'zone' in G.nodes[node]}
                    print(f"Found zone mapping in graph attributes: {len(node_to_zone)} nodes mapped")
                else:
                    print("Recomputing zone mapping from shapefile...")
                    _, _, node_to_zone_temp, _, _ = load_graph(
                        place_name=args.place_name,
                        zone_shp=args.zone_shp,
                        no_congestion=args.no_congestion,
                        manhattan_area=args.manhattan_area,
                    )
                    graph_nodes_set = set(all_nodes_list)
                    node_to_zone = {node: zone for node, zone in node_to_zone_temp.items() if node in graph_nodes_set}
                    print(f"Computed zone mapping: {len(node_to_zone)} nodes mapped to zones")
            except Exception as e:
                print(f"Warning: Could not load zone mapping: {e}")
            node_to_zone = None
    
    selection_method = getattr(args, 'three_fixed_selection_method', 'random')
    
    # Helper function to select nodes ensuring different zones
    def select_nodes_from_different_zones(candidate_nodes_with_scores, node_to_zone_mapping):
        """Select 3 nodes ensuring they come from different zones."""
        selected_nodes = []
        selected_zones = set()
        
        if node_to_zone_mapping is None or len(node_to_zone_mapping) == 0:
            print("  Warning: No zone mapping available. Selecting top 3 nodes without zone constraint.")
            for node, score in candidate_nodes_with_scores:
                selected_nodes.append(node)
                if len(selected_nodes) == 3:
                    break
            return selected_nodes, set()
        
        print(f"  Checking zones for top candidates...")
        zone_counts = {}
        for node, score in candidate_nodes_with_scores[:20]:
            zone = node_to_zone_mapping.get(node)
            if zone is not None:
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        print(f"  Zones in top 20 candidates: {len(zone_counts)} unique zones")
        if len(zone_counts) > 0:
            print(f"  Top zones: {sorted(zone_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        for node, score in candidate_nodes_with_scores:
            node_zone = node_to_zone_mapping.get(node)
            if node_zone is not None and node_zone not in selected_zones:
                selected_nodes.append(node)
                selected_zones.add(node_zone)
                print(f"  Selected node {node} from zone {node_zone} (score: {score:.6f})")
                if len(selected_nodes) == 3:
                    break
            elif node_zone is None:
                continue
        
        if len(selected_nodes) < 3:
            print(f"  Warning: Only found {len(selected_nodes)} nodes from different zones.")
            for node, score in candidate_nodes_with_scores:
                if node not in selected_nodes:
                    node_zone = node_to_zone_mapping.get(node, "unknown")
                    selected_nodes.append(node)
                    selected_zones.add(node_zone)
                    print(f"  Fallback: Selected node {node} from zone {node_zone} (score: {score:.6f})")
                    if len(selected_nodes) == 3:
                        break
        
        print(f"  Final selection: {len(selected_zones)} unique zones: {sorted(selected_zones)}")
        return selected_nodes, selected_zones
    
    if selection_method == "random":
        if node_to_zone is not None:
            zone_to_nodes = {}
            for node in all_nodes_list:
                zone = node_to_zone.get(node)
                if zone is not None:
                    if zone not in zone_to_nodes:
                        zone_to_nodes[zone] = []
                    zone_to_nodes[zone].append(node)
            
            available_zones = list(zone_to_nodes.keys())
            if len(available_zones) < 3:
                print(f"  Warning: Only {len(available_zones)} zones available. Some nodes may be from the same zone.")
            
            rng_key = jax_random.PRNGKey(args.seed)
            permuted_zone_indices = jax_random.permutation(rng_key, jnp.arange(len(available_zones)))
            selected_zones_list = [available_zones[int(idx)] for idx in permuted_zone_indices[:3]]
            
            selected_nodes = []
            for zone in selected_zones_list:
                zone_nodes = zone_to_nodes[zone]
                node_idx = jax_random.randint(rng_key, (1,), 0, len(zone_nodes))[0]
                selected_nodes.append(zone_nodes[int(node_idx)])
                rng_key = jax_random.split(rng_key, 1)[0]
            
            print(f"Selected 3 nodes using random selection (one per zone): {selected_nodes}")
            print(f"  Zones: {[node_to_zone.get(n, 'unknown') for n in selected_nodes]}")
        else:
            rng_key = jax_random.PRNGKey(args.seed)
            permuted_indices = jax_random.permutation(rng_key, jnp.arange(len(all_nodes_list)))
            selected_indices = permuted_indices[:3]
            selected_nodes = [all_nodes_list[int(idx)] for idx in selected_indices]
            print(f"Selected 3 nodes using random selection: {selected_nodes}")
        
    elif selection_method == "degree":
        print("Computing degree centrality...")
        degree_centrality = nx.degree_centrality(G)
        sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        selected_nodes, selected_zones = select_nodes_from_different_zones(sorted_nodes, node_to_zone)
        print(f"Selected 3 nodes using degree centrality (one per zone): {selected_nodes}")
        centrality_values = [f"{degree_centrality[n]:.4f}" for n in selected_nodes]
        print(f"  Degree centrality values: {centrality_values}")
        if node_to_zone:
            print(f"  Zones: {[node_to_zone.get(n, 'unknown') for n in selected_nodes]}")
        
    elif selection_method == "closeness":
        print("Computing closeness centrality (this may take a moment for large graphs)...")
        closeness_centrality = nx.closeness_centrality(G)
        valid_nodes = [(node, cent) for node, cent in closeness_centrality.items() if cent > 0]
        if len(valid_nodes) < 3:
            print(f"  Warning: Only {len(valid_nodes)} nodes have non-zero closeness centrality. Using all nodes.")
            valid_nodes = list(closeness_centrality.items())
        sorted_nodes = sorted(valid_nodes, key=lambda x: x[1], reverse=True)
        selected_nodes, selected_zones = select_nodes_from_different_zones(sorted_nodes, node_to_zone)
        print(f"Selected 3 nodes using closeness centrality (one per zone): {selected_nodes}")
        centrality_values = [f"{closeness_centrality[n]:.4f}" for n in selected_nodes]
        print(f"  Closeness centrality values: {centrality_values}")
        if node_to_zone:
            print(f"  Zones: {[node_to_zone.get(n, 'unknown') for n in selected_nodes]}")
        
    elif selection_method == "betweenness":
        print("Computing betweenness centrality (this may take a while for large graphs)...")
        betweenness_centrality = nx.betweenness_centrality(G)
        sorted_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
        selected_nodes, selected_zones = select_nodes_from_different_zones(sorted_nodes, node_to_zone)
        print(f"Selected 3 nodes using betweenness centrality (one per zone): {selected_nodes}")
        centrality_values = [f"{betweenness_centrality[n]:.4f}" for n in selected_nodes]
        print(f"  Betweenness centrality values: {centrality_values}")
        if node_to_zone:
            print(f"  Zones: {[node_to_zone.get(n, 'unknown') for n in selected_nodes]}")
    
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    # Convert selected nodes to indices
    selected_indices = [node_to_idx[node] for node in selected_nodes if node in node_to_idx]
    if len(selected_indices) != 3:
        raise ValueError(f"Failed to select 3 nodes. Got {len(selected_indices)} nodes: {selected_indices}")
    
    return selected_indices

# Handle restricted starts/pickups sampling if flags are enabled (BEFORE creating environment)
if getattr(args, 'sample_starts_from_three_fixed', False):
    print("Selecting 3 fixed start nodes...")
    fixed_starts_idx = select_three_fixed_nodes(G, node_to_idx, args, target="starts")
    print(f"Restricted starts (3 fixed nodes, indices): {fixed_starts_idx}")
    # Set pickups to all nodes for uniform random sampling
    all_nodes_list = list(node_to_idx.keys())
    fixed_pickups_idx = list(range(len(all_nodes_list)))
    print(f"Pickups set to all nodes for uniform random sampling: {len(fixed_pickups_idx)} nodes")

if getattr(args, 'sample_pickups_from_three_fixed', False):
    print("Selecting 3 fixed pickup nodes...")
    fixed_pickups_idx = select_three_fixed_nodes(G, node_to_idx, args, target="pickups")
    print(f"Restricted pickups (3 fixed nodes, indices): {fixed_pickups_idx}")
    # Set starts to all nodes for uniform random sampling
    all_nodes_list = list(node_to_idx.keys())
    fixed_starts_idx = list(range(len(all_nodes_list)))
    print(f"Starts set to all nodes for uniform random sampling: {len(fixed_starts_idx)} nodes")

# Ensure all fixed arrays are on device (CPU/GPU based on JAX_PLATFORMS) with consistent dtypes
fixed_starts_idx = jax.device_put(jnp.array(fixed_starts_idx, dtype=jnp.int32))
fixed_pickups_idx = jax.device_put(jnp.array(fixed_pickups_idx, dtype=jnp.int32))

# Compute normalized node coordinates for Euclidean distance in reward
from taxi_env_utils import compute_normalized_node_coordinates
node_coordinates = compute_normalized_node_coordinates(G, node_to_idx)

# Create environment
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
    gamma=args.gamma,
    noise_mask=noise_mask,  # Per-step congestion noise (None = deterministic)
)

# Observation function
obs_fn_single, obs_fn_batch = make_obs_fn(env, G, node_to_idx)

# Initialize batched environment states for metrics logging and evaluation
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
    raise ValueError("This polished entrypoint only supports discrete Q-learning training. Use --eval_only_sp for the shortest-path baseline.")

if not args.discrete:
    raise ValueError("This polished entrypoint only supports the discrete tabular Q-learning pipeline. Pass --discrete.")

from modes.train_q_learning import run_q_learning

run_q_learning(args, ctx)
