import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import networkx as nx
from typing import Callable, Dict, Tuple
import os
import pickle
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

from taxi_env import TaxiEnv, TaxiState
from utils import load_graph, fixed_starts_pickups, load_simple_graph

# --- Graph conversion utilities ---
def build_adj_and_time_matrix(G: nx.DiGraph, max_deg=None, node_to_idx: dict = None):
    node_list = list(G.nodes())
    num_nodes = len(node_list)

    if max_deg is None:
        max_deg = max(dict(G.out_degree()).values())

    adj = -np.ones((num_nodes, max_deg), dtype=int)
    times = np.zeros((num_nodes, max_deg), dtype=np.float32)
    neighbor_mask = np.zeros((num_nodes, max_deg), dtype=bool)

    for i, node in enumerate(node_list):
        neighbors = list(G.successors(node))
        n_neighbors = len(neighbors)
        for j, nbr in enumerate(neighbors[:max_deg]):
            best_k = min(G[node][nbr], key=lambda k: G[node][nbr][k]['travel_time_congested'])
            time_min = G[node][nbr][best_k]['travel_time_congested']
            time_sec = time_min * 60.0
            adj[i, j] = node_to_idx[nbr]
            times[i, j] = time_sec
        if 0 < n_neighbors < max_deg:
            # pad with first neighbor
            adj[i, n_neighbors:] = adj[i, 0]
            times[i, n_neighbors:] = times[i, 0]
        neighbor_mask[i, :n_neighbors] = True

    # Ensure arrays are on GPU with proper dtypes
    return jax.device_put(jnp.array(adj, dtype=jnp.int32)), jax.device_put(jnp.array(times, dtype=jnp.float32)), jax.device_put(jnp.array(neighbor_mask, dtype=bool))

def load_or_compute_distance_matrix_parallel(G, node_to_idx, cache_file="manhattan_distances.pkl", num_workers=None):
    """
    Load precomputed distance matrix or compute and cache it using parallel processing.
    """
    if os.path.exists(cache_file):
        # print(f"Loading precomputed distance matrix from {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            # Return None for paths_dict to use smart greedy approach
            return data['dist_mat'], data['hop_dist_mat'], data['max_length'], None
    
    # print("Computing distance matrix with parallel processing (hybrid approach - no paths)...")
    all_nodes = list(G.nodes())
    N = len(all_nodes)
    # print(f"Computing all-pairs shortest paths for {N} nodes using {num_workers or mp.cpu_count()} workers...")
    
    dist_mat = np.zeros((N, N), dtype=np.float32)
    hop_dist_mat = np.zeros((N, N), dtype=np.float32)
    max_length = 0
    
    # Parallel computation of distance matrices (NO PATH STORAGE)
    def compute_node_distances(node_batch):
        local_dist = np.zeros((len(node_batch), N), dtype=np.float32)
        local_hop = np.zeros((len(node_batch), N), dtype=np.float32)
        local_max = 0
        
        for i, u in enumerate(node_batch):
            ui = node_to_idx[u]
            # Travel time distances ONLY (no path storage) - convert to seconds
            lengths = nx.single_source_dijkstra_path_length(G, u, weight="travel_time_congested")
            for v, d in lengths.items():
                vi = node_to_idx[v]
                local_dist[i, vi] = d * 60.0  # Convert minutes to seconds
            
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
    
    # Combine results (NO PATH MERGING)
    for i, (local_dist, local_hop, local_max) in enumerate(results):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, N)
        dist_mat[start_idx:end_idx] = local_dist
        hop_dist_mat[start_idx:end_idx] = local_hop
        max_length = max(max_length, local_max)
    
    # Cache the results (NO PATHS_DICT)
    # print(f"Caching distance matrix to {cache_file} (hybrid approach)")
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'dist_mat': dist_mat,
            'hop_dist_mat': hop_dist_mat,
            'max_length': max_length,
            # No paths_dict - using smart greedy approach instead
        }, f)
    
    return dist_mat, hop_dist_mat, max_length, None

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


def build_env(args):
    """
    Build graph G, node_to_idx mapping, fixed start/pickup indices.
    Returns: G, node_to_idx, fixed_starts_idx, fixed_pickups_idx
    """
    if args.env_type == 'manhattan':
        # --- Load and preprocess Manhattan graph ---
        G, nodes_gdf, node_to_zone, zone_to_nodes = load_graph(place_name = args.place_name, zone_shp = args.zone_shp, no_congestion= args.no_congestion)

        fixed_starts_idx, fixed_pickups_idx, node_to_idx, idx_to_node = fixed_starts_pickups(
            G, nodes_gdf, node_to_zone, zone_to_nodes, all = False
        )
        fixed_starts_idx  = jnp.array(fixed_starts_idx, dtype=jnp.int32)
        fixed_pickups_idx = jnp.array(fixed_pickups_idx, dtype=jnp.int32)
        print(f"Fixed starts: {fixed_starts_idx}")
        print(f"Fixed pickups: {fixed_pickups_idx}")

    elif args.env_type == 'simple':
        G = load_simple_graph(num_layers=args.num_layers, width=args.layer_width, no_congestion=args.no_congestion)
        # index mappings
        nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        idx_to_node = [n for n, _ in sorted(node_to_idx.items(), key=lambda x: x[1])]

        # only one fixed start (node 0) and one fixed pickup (last one)
        fixed_starts_idx = jnp.array([node_to_idx[n] for n in nodes[:args.layer_width]], dtype=jnp.int32)
        fixed_pickups_idx = jnp.array([node_to_idx[n] for n in nodes[-args.layer_width:]], dtype=jnp.int32)

    else:
        raise ValueError(f"Unknown env_type: {args.env_type}")
    
    traffic_params = build_traffic_params(G, node_to_idx, args.cycle_length, args.offset, seed=42)

    return G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params


def compute_normalized_node_coordinates(
    G: nx.DiGraph,
    node_to_idx: Dict[int, int]
) -> jnp.ndarray:
    """
    Compute normalized lat/lon coordinates for all nodes.
    Returns: [num_nodes, 2] array of normalized coordinates
    """
    idx_to_node = [n for n, _ in sorted(node_to_idx.items(), key=lambda x: x[1])]
    lats = jax.device_put(jnp.array([G.nodes[n]['y'] for n in idx_to_node], dtype=jnp.float32))
    lons = jax.device_put(jnp.array([G.nodes[n]['x'] for n in idx_to_node], dtype=jnp.float32))
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    latlon = jax.device_put(jnp.stack([
        (lats - lat_min) / (lat_max - lat_min),
        (lons - lon_min) / (lon_max - lon_min)
    ], axis=-1))  # [N,2]
    return latlon


def make_obs_fn(
    env: TaxiEnv,
    G: nx.DiGraph,
    node_to_idx: Dict[int,int],
    include_neighbor_info: bool = True
) -> Tuple[
    Callable[[TaxiState], jnp.ndarray],
    Callable[[TaxiState], jnp.ndarray]
]:
    """
    Returns two functions:
    #   - single_obs: TaxiState -> (state_feats [6], action_feats [max_deg,2], global_feats [3*N])
    #   - obs_fn_batch: batched TaxiState -> dict of trajectories
      - single_obs: TaxiState -> observation vector [9] 
        Features: [current_pos(2), pickup_pos(2), relative_pos(2), distance(1), angle(1), time(1)]
      - obs_fn_batch: batched TaxiState -> batched observation vectors [B, 9]
    """
    # Precompute normalized lat/lon per node - ensure on GPU
    latlon = compute_normalized_node_coordinates(G, node_to_idx)
    
    # Precompute max distance for normalization (if using neighbor info)
    max_dist = float(env.distances.max()) if include_neighbor_info else 0.0

    def single_obs(s: TaxiState) -> jnp.ndarray:
        """
        Optimized observation function with reduced computations and better memory layout.
        
        Performance optimizations:
        1. Pre-computed normalized coordinates on GPU
        2. Reduced memory allocations
        3. Optimized mathematical operations
        4. Better memory access patterns
        """
        # Use pre-computed normalized lat/lon coordinates (already on GPU)
        xy_c = latlon[s.current_node]   # [2] - direct GPU memory access
        xy_p = latlon[s.pickup_node]    # [2] - direct GPU memory access
        
        # Compute relative position (direction vector from current to pickup)
        # relative_pos = xy_p - xy_c  # [2] - single operation
        
        # Compute distance and angle efficiently (avoid expensive operations)
        # distance_sq = jnp.sum(relative_pos * relative_pos)  # [1] - faster than linalg.norm
        # distance = jnp.sqrt(distance_sq + 1e-8)  # [1] - add small epsilon for numerical stability
        
        # # Compute angle efficiently
        # angle = jnp.arctan2(relative_pos[1], relative_pos[0])  # [1] - direct computation
        
        # Base features: current_pos(2), pickup_pos(2), time(1) = 5 dims
        base_dims = 5
        neighbor_dims = 2 * env.max_deg if include_neighbor_info else 0
        total_dims = base_dims + neighbor_dims
        
        obs = jnp.zeros(total_dims, dtype=jnp.float32)
        
        # Fill in base values
        obs = obs.at[0:2].set(xy_c)           # [2] - current position
        obs = obs.at[2:4].set(xy_p)           # [2] - pickup position  
        obs = obs.at[4].set(s.time)           # [1] - time
        
        # Add neighbor information if enabled (critical for matching expert decisions)
        if include_neighbor_info:
            # Get travel times to each neighbor
            neighbor_travel_times = env.travel_times[s.current_node]  # [max_deg]
            # Normalize travel times (divide by max travel time for stability)
            neighbor_travel_times_norm = neighbor_travel_times / (env.max_travel_time + 1e-8)
            
            # Get distances from each neighbor to pickup
            neighbors = env.adj_list[s.current_node]  # [max_deg]
            neighbor_distances = env.distances[neighbors, s.pickup_node]  # [max_deg]
            # Normalize distances (divide by max distance for stability - precomputed)
            neighbor_distances_norm = neighbor_distances / (max_dist + 1e-8)
            
            # Mask invalid neighbors (set to large value instead of 0 to distinguish from valid)
            # Use mask to zero out invalid positions
            neighbor_travel_times_norm = jnp.where(s.neighbor_mask, neighbor_travel_times_norm, 0.0)
            neighbor_distances_norm = jnp.where(s.neighbor_mask, neighbor_distances_norm, 0.0)
            
            # Store neighbor info: [travel_times (max_deg), distances (max_deg)]
            obs = obs.at[5:5+env.max_deg].set(neighbor_travel_times_norm)
            obs = obs.at[5+env.max_deg:5+2*env.max_deg].set(neighbor_distances_norm)
        
        return obs

    # JIT and batched versions with optimizations
    single_obs = jax.jit(single_obs)
    
    # Pre-compile the vmap for better performance
    single_obs_batched = jax.jit(jax.vmap(single_obs))

    @jax.jit
    def obs_fn_batch(batch: TaxiState) -> jnp.ndarray:
        """Optimized batch observation function with pre-compiled vmap"""
        return single_obs_batched(batch)

    return single_obs, obs_fn_batch



def get_global_state(env, batch) -> jnp.ndarray:

    t = batch.time  # jnp.ndarray, shape [B]
    B = t.shape[0]  # batch size
    N = env.num_nodes
    
    global_feats = []
    periods = jnp.broadcast_to(env.periods, (B, N))
    green   = jnp.broadcast_to(env.green_durations, (B, N))
    green_ratios = green / periods
    offs    = jnp.broadcast_to(env.offsets, (B, N))

    # compute time within current cycle
    cycle_pos = jnp.mod(t[:, None] + offs, periods)
    # determine if currently in green phase
    is_green = (cycle_pos < green).astype(jnp.float32)

    # compute time until next switch
    time_to_switch = jnp.where(
        is_green,
        green - cycle_pos,
        periods - cycle_pos
    ) 

    global_feats = jnp.concatenate([
        green_ratios,
        is_green,
        time_to_switch
    ], axis=1)

    return global_feats

# --- Example init_state_fn ---
def make_init_state_fn(env: TaxiEnv, num_envs: int, rng_key):
    """
    Returns a batched initial TaxiState array and new rng_key.
    """
    def init_state():
        def one(env_state_key):
            state, new_key = env.reset(env_state_key)
            return state, new_key
        keys = jax.random.split(rng_key, num_envs)
        states_keys = list(map(one, keys))
        states = jnp.stack([sk[0] for sk in states_keys])
        new_key = jax.random.fold_in(rng_key, num_envs)
        return states, new_key
    return init_state

def build_traffic_params(G: nx.DiGraph,
                         node_to_idx: Dict[int,int],
                         cycle_length: float = 60.0,
                         offset: float = 0.0,
                         seed: int = 0,
                        ) -> Dict[int, Tuple[float,float,float]]:
    """
    For each intersection (node), look at the highway‐types of
    all incident edges and pick a cycle/green split:
      • Motorway/Trunk: always green (no light)
      • Primary:         cycle/green = 5/6
      • Secondary:       cycle/green = 2/3
      • Tertiary:        cycle/green = 1/6
      • Otherwise:       cycle/green = 1/2
    We then pick a fixed offset per each node.
    """
    rng = np.random.default_rng(seed)
    traffic_params = {}
    for node, idx in node_to_idx.items():
        # collect all highway‐types on edges touching this node
        types = []
        # For a MultiDiGraph:
        for u, v, key, data in G.edges(node, keys=True, data=True):
            hw = data.get("highway", "unclassified")
            if isinstance(hw, list):
                types.extend(hw)
            else:
                types.append(hw)
        # decide cycle & green based on priority
        if any(t in ("motorway", "trunk") for t in types):
            cycle, green = cycle_length, cycle_length    # effectively always green
        elif any(t == "primary" for t in types):
            cycle, green = cycle_length, cycle_length * 0.5
        elif any(t == "secondary" for t in types):
            cycle, green = cycle_length, cycle_length * 0.3
        elif any(t == "tertiary" for t in types):
            cycle, green = cycle_length, cycle_length #* 1.0/6.0
        elif any(t in ("residential", "living_street") for t in types):
            cycle, green = cycle_length, cycle_length * 0.3
        else:
            cycle, green = cycle_length, cycle_length

        # random phase offset
        offset = offset
        # --- VALIDITY CHECKS ---
        # assert cycle > 0, f"Cycle length for node {node} must be positive, got {cycle}"
        # assert green > 0, f"Green duration must be positive, got {green}"
        # assert green <= cycle, (
        #     f"Green duration ({green}) exceeds cycle ({cycle}) at node {node}"
        # )
        # assert 0 <= offset < cycle, (
        #     f"Offset {offset:.2f} not in [0, {cycle}) for node {node}"
        # )

        traffic_params[node_to_idx[node]] = (cycle, green, offset)

    missing = set(node_to_idx.values()) - set(traffic_params.keys())
    assert not missing, f"Missing params for node indices: {sorted(missing)}"

    return traffic_params
