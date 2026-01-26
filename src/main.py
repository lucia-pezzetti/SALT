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
from utils import EstimateReturnsState, load_graph
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
    # Note: Platform is set via JAX_PLATFORMS environment variable (CPU/GPU)
    # try:
    #     jax.config.update('jax_platform_name', 'cpu')  # Ensure we're using CPU backend
    # except:
    #     pass  # Ignore if already set
    
    # Set memory preallocation (respect environment variables if set)
    if 'XLA_PYTHON_CLIENT_PREALLOCATE' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'  # 80% memory for GPU
    if 'XLA_PYTHON_CLIENT_ALLOCATOR' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # Use platform allocator
    
    # Configure XLA flags to avoid contiguous memory allocation issues
    # This prevents LLVM from trying to allocate large contiguous memory sections
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags_parts = xla_flags.split() if xla_flags else []
    
    # Add flags to reduce memory pressure during compilation
    # Only use valid XLA flags to avoid crashes
    # Note: CPU-specific flag removed for GPU compatibility
    flags_to_add = [
        # '--xla_cpu_enable_fast_math=false',  # CPU-specific, disabled for GPU
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
parser.add_argument("--random_offsets", action="store_true", help="Use random offsets for each traffic light instead of uniform offset")
parser.add_argument("--cycle_length", type=int, default=90, help="Cycle length for the customised grid environment")
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
parser.add_argument("--pretrain_enabled", action="store_true", help="Enable shortest path pretraining for Q-learning (discrete mode only)")
parser.add_argument("--num_pretrain_episodes", type=int, default=10000, help="Number of pretraining episodes using shortest path rollouts (for Q-learning)")
parser.add_argument("--pretrain_learning_rate", type=float, default=None, help="Learning rate for pretraining (default: None = use agent's LR). Recommended: 0.01-0.05 when using --init_from_shortest_paths")
parser.add_argument("--init_from_shortest_paths", action="store_true", help="Initialize Q-table from shortest path travel times (discrete mode only)")
parser.add_argument("--init_all_time_slices", action="store_true", help="Initialize Q-table for all time slices (up to 100) instead of just time=0 (requires --init_from_shortest_paths)")
parser.add_argument("--eval_only_sp", action="store_true", help="Run shortest-path baselines (continuous + discrete) and exit (discrete/Q-learning pipeline)")
parser.add_argument("--sample_starts_from_three_fixed", action="store_true", help="Sample starting nodes with repetition from three fixed nodes (chosen at start of training and kept fixed). Pickups still sampled from all nodes.")
parser.add_argument("--three_fixed_selection_method", type=str, default="random", choices=["random", "degree", "closeness", "betweenness"], help="Method to select the 3 fixed nodes: 'random' (default), 'degree' (degree centrality), 'closeness' (closeness centrality), 'betweenness' (betweenness centrality)")

args = parser.parse_args()

# Validate fixed evaluation parameters
if args.fixed_eval:
    if args.fixed_starts is None or args.fixed_pickups is None:
        raise ValueError("When using --fixed_eval, both --fixed_starts and --fixed_pickups must be provided")
    if len(args.fixed_starts) != len(args.fixed_pickups):
        raise ValueError("--fixed_starts and --fixed_pickups must have the same length")

# Validate zone-based filtering parameters
# Default: both None (all nodes)
# If only one is provided, the other defaults to None (all nodes)
# If both are provided, use both as specified

# Convert string numbers to integers if they look like numbers
def convert_zone(zone):
    """Convert zone to appropriate type (int if numeric, str otherwise)."""
    if isinstance(zone, str):
        # Try to convert to int if it's a numeric string
        try:
            return int(zone)
        except ValueError:
            # It's a zone name, keep as string
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
    # if specified, load the graph from the cache file
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

# Precompute shortest-path distance matrix
if args.env_type == "manhattan":
    # distance_cache_file = os.path.join(args.cache_dir, "manhattan_distances_4zones_1pair.pkl")
    distance_cache_file = None
else:
    distance_cache_file = os.path.join(args.cache_dir, f"simple_distances_{args.num_layers}layers_{args.offset}offset.pkl")
dist_mat, hop_dist_mat, max_length, paths_dict = load_or_compute_distance_matrix_parallel(
    G, node_to_idx, distance_cache_file, args.num_workers
)

# Place distance matrices on GPU
distances = jax.device_put(jnp.array(dist_mat, dtype=jnp.float32))
hop_distances = jax.device_put(jnp.array(hop_dist_mat, dtype=jnp.float32))

# Handle restricted starts sampling if flag is enabled (BEFORE creating environment)
if getattr(args, 'sample_starts_from_three_fixed', False):
    import networkx as nx
    import pickle
    from utils import load_graph
    
    all_nodes_list = list(node_to_idx.keys())
    if len(all_nodes_list) < 3:
        raise ValueError(f"Not enough nodes in graph ({len(all_nodes_list)}). Need at least 3 nodes for --sample_starts_from_three_fixed.")
    
    # Get node_to_zone mapping to ensure nodes come from different zones
    node_to_zone = None
    if args.env_type == "manhattan":
        # Try to get from cache first (check the same cache file used for graph loading)
        # For manhattan, graph_cache_file is typically None, so we'll need to recompute
        # But first check if there's a cached version we can use
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
                            # Filter to only nodes in our graph
                            graph_nodes_set = set(all_nodes_list)
                            node_to_zone = {node: zone for node, zone in node_to_zone.items() if node in graph_nodes_set}
                            print(f"Loaded zone mapping from cache: {len(node_to_zone)} nodes mapped to zones")
                            break
                except Exception as e:
                    print(f"Could not load zone mapping from {cache_file}: {e}")
                    continue
        
        # If not in cache, recompute it (only for manhattan graphs)
        if node_to_zone is None:
            print("Loading zone mapping to ensure nodes from different zones...")
            # We need to get the zone mapping, but load_graph returns a filtered graph
            # So we need to get it from the graph that was already loaded
            # The zone mapping should be in the graph's node attributes or we need to recompute it
            # Let's try to get it from the already loaded graph by checking node attributes
            # or by loading the full graph temporarily
            try:
                # Try to get zone info from graph node attributes if available
                sample_node = list(G.nodes())[0] if len(G.nodes()) > 0 else None
                if sample_node and 'zone' in G.nodes[sample_node]:
                    # Zones are stored in node attributes
                    node_to_zone = {node: G.nodes[node].get('zone') for node in all_nodes_list if 'zone' in G.nodes[node]}
                    print(f"Found zone mapping in graph attributes: {len(node_to_zone)} nodes mapped")
                else:
                    # Need to recompute by loading the graph
                    print("Recomputing zone mapping from shapefile...")
                    _, _, node_to_zone_temp, _, _ = load_graph(
                        place_name=args.place_name,
                        zone_shp=args.zone_shp,
                        no_congestion=args.no_congestion
                    )
                    # Filter to only nodes in our graph
                    graph_nodes_set = set(all_nodes_list)
                    node_to_zone = {node: zone for node, zone in node_to_zone_temp.items() if node in graph_nodes_set}
                    print(f"Computed zone mapping: {len(node_to_zone)} nodes mapped to zones")
            except Exception as e:
                print(f"Warning: Could not load zone mapping: {e}")
                node_to_zone = None
        
        # Debug: Print zone distribution
        if node_to_zone is not None and len(node_to_zone) > 0:
            zone_counts = {}
            for node in all_nodes_list:
                zone = node_to_zone.get(node)
                if zone is not None:
                    zone_counts[zone] = zone_counts.get(zone, 0) + 1
            print(f"Zone distribution: {len(zone_counts)} unique zones, {sum(zone_counts.values())} nodes with zones")
            print(f"  Top zones by node count: {sorted(zone_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
        else:
            print("Warning: No zone mapping available. Cannot ensure nodes from different zones.")
    
    selection_method = getattr(args, 'three_fixed_selection_method', 'random')
    
    # Helper function to select nodes ensuring different zones
    def select_nodes_from_different_zones(candidate_nodes_with_scores, node_to_zone_mapping):
        """
        Select 3 nodes ensuring they come from different zones.
        candidate_nodes_with_scores: list of (node, score) tuples, sorted by score (descending)
        Returns: list of 3 nodes from different zones
        """
        selected_nodes = []
        selected_zones = set()
        
        if node_to_zone_mapping is None or len(node_to_zone_mapping) == 0:
            print("  Warning: No zone mapping available. Selecting top 3 nodes without zone constraint.")
            # No zone mapping available (e.g., simple graph), just take top 3
            for node, score in candidate_nodes_with_scores:
                selected_nodes.append(node)
                if len(selected_nodes) == 3:
                    break
            return selected_nodes, set()
        
        # Debug: show zone distribution in top candidates
        print(f"  Checking zones for top candidates...")
        zone_counts = {}
        for node, score in candidate_nodes_with_scores[:20]:  # Check top 20
            zone = node_to_zone_mapping.get(node)
            if zone is not None:
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        print(f"  Zones in top 20 candidates: {len(zone_counts)} unique zones")
        if len(zone_counts) > 0:
            print(f"  Top zones: {sorted(zone_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        for node, score in candidate_nodes_with_scores:
            # Check if node has a zone and if we need a node from that zone
            node_zone = node_to_zone_mapping.get(node)
            if node_zone is not None and node_zone not in selected_zones:
                selected_nodes.append(node)
                selected_zones.add(node_zone)
                print(f"  Selected node {node} from zone {node_zone} (score: {score:.6f})")
                if len(selected_nodes) == 3:
                    break
            elif node_zone is None:
                # Node doesn't have a zone, skip it for now (we want nodes with zones)
                continue
        
        if len(selected_nodes) < 3:
            # Fallback: if we can't find 3 nodes from different zones, use what we have
            print(f"  Warning: Only found {len(selected_nodes)} nodes from different zones.")
            # Fill remaining slots with any available nodes (even if same zone)
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
        # Randomly select 3 nodes from different zones
        if node_to_zone is not None:
            # Group nodes by zone
            zone_to_nodes = {}
            for node in all_nodes_list:
                zone = node_to_zone.get(node)
                if zone is not None:
                    if zone not in zone_to_nodes:
                        zone_to_nodes[zone] = []
                    zone_to_nodes[zone].append(node)
            
            # Randomly select one node from each of 3 different zones
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
            # No zone mapping, use original random selection
            rng_key = jax_random.PRNGKey(args.seed)
            permuted_indices = jax_random.permutation(rng_key, jnp.arange(len(all_nodes_list)))
            selected_indices = permuted_indices[:3]
            selected_nodes = [all_nodes_list[int(idx)] for idx in selected_indices]
            print(f"Selected 3 nodes using random selection: {selected_nodes}")
        
    elif selection_method == "degree":
        # Select top node from each of 3 different zones by degree centrality
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
        # Select top node from each of 3 different zones by closeness centrality
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
        # Select top node from each of 3 different zones by betweenness centrality
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
    
    # Verify that nodes are from different zones
    if node_to_zone is not None and len(node_to_zone) > 0:
        selected_node_zones = [node_to_zone.get(node, None) for node in selected_nodes]
        unique_zones = set(z for z in selected_node_zones if z is not None)
        print(f"\nZone verification:")
        print(f"  Selected nodes: {selected_nodes}")
        print(f"  Their zones: {selected_node_zones}")
        print(f"  Unique zones: {len(unique_zones)} ({sorted(unique_zones) if unique_zones else 'None'})")
        if len(unique_zones) < 3:
            print(f"  WARNING: Only {len(unique_zones)} unique zones found! Some nodes may be from the same zone.")
            # Try to fix by selecting from different zones more aggressively
            if len(unique_zones) < 3 and selection_method != "random":
                print(f"  Attempting to fix by selecting from different zones...")
                # Group nodes by zone and select top from each zone
                zone_to_top_node = {}
                if selection_method == "degree":
                    centrality_dict = nx.degree_centrality(G)
                elif selection_method == "closeness":
                    centrality_dict = nx.closeness_centrality(G)
                elif selection_method == "betweenness":
                    centrality_dict = nx.betweenness_centrality(G)
                else:
                    centrality_dict = None
                
                if centrality_dict:
                    for node in all_nodes_list:
                        zone = node_to_zone.get(node)
                        if zone is not None:
                            if zone not in zone_to_top_node:
                                zone_to_top_node[zone] = (node, centrality_dict.get(node, 0))
                            else:
                                current_score = centrality_dict.get(node, 0)
                                if current_score > zone_to_top_node[zone][1]:
                                    zone_to_top_node[zone] = (node, current_score)
                    
                    # Select top node from each of 3 different zones
                    sorted_zones = sorted(zone_to_top_node.items(), key=lambda x: x[1][1], reverse=True)
                    selected_nodes = [zone_to_top_node[zone][0] for zone, _ in sorted_zones[:3]]
                    selected_node_zones = [node_to_zone.get(node, None) for node in selected_nodes]
                    unique_zones = set(z for z in selected_node_zones if z is not None)
                    print(f"  After fix: {len(unique_zones)} unique zones: {sorted(unique_zones)}")
    
    # Convert selected nodes to indices
    fixed_starts_idx = [node_to_idx[node] for node in selected_nodes if node in node_to_idx]
    if len(fixed_starts_idx) != 3:
        raise ValueError(f"Failed to select 3 nodes. Got {len(fixed_starts_idx)} nodes: {fixed_starts_idx}")
    print(f"Restricted starts (3 fixed nodes, indices): {fixed_starts_idx}")

# Ensure all fixed arrays are on GPU with consistent dtypes
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
    timeout_penalty=args.timeout_penalty,
    gamma=args.gamma,   
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