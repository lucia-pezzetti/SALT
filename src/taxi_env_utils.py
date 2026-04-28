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

# Graph conversion utilities
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
            # Pad with first neighbor
            adj[i, n_neighbors:] = adj[i, 0]
            times[i, n_neighbors:] = times[i, 0]
        neighbor_mask[i, :n_neighbors] = True

    # Align arrays
    return jax.device_put(jnp.array(adj, dtype=jnp.int32)), jax.device_put(jnp.array(times, dtype=jnp.float32)), jax.device_put(jnp.array(neighbor_mask, dtype=bool))


def build_noise_mask(G: nx.DiGraph, node_to_idx: dict, noise_level: float = 0.2, max_deg: int = None) -> np.ndarray:
    """
    Build a per-edge noise ceiling array of shape [num_nodes, max_deg].

    For each edge, the value is the maximum *extra* multiplicative noise that
    will be applied at runtime:
      - Primary roads:   noise_level       (default 0.2 → up to +20%)
      - Secondary roads: 1.5 * noise_level (default 0.3 → up to +30%)
      - All others:      0.0              (no noise)

    At each environment step the actual travel time is:
        travel * (1 + U(0, noise_mask[curr, action]))
    so a mask value of 0.0 means the edge is deterministic.
    """
    node_list = list(G.nodes())
    num_nodes = len(node_list)
    if max_deg is None:
        max_deg = max(dict(G.out_degree()).values())

    mask = np.zeros((num_nodes, max_deg), dtype=np.float32)

    n_primary = 0
    n_secondary = 0

    for i, node in enumerate(node_list):
        neighbors = list(G.successors(node))
        n_neighbors = len(neighbors)
        for j, nbr in enumerate(neighbors[:max_deg]):
            best_k = min(G[node][nbr], key=lambda k: G[node][nbr][k]['travel_time_congested'])
            hw = G[node][nbr][best_k].get('highway', 'unclassified')
            if isinstance(hw, list):
                hw = hw[0] if hw else 'unclassified'

            if hw == 'primary':
                mask[i, j] = noise_level
                n_primary += 1
            elif hw == 'secondary':
                mask[i, j] = 1.5 * noise_level
                n_secondary += 1

        # Pad with first edge's value (same as adj_list padding)
        if 0 < n_neighbors < max_deg:
            mask[i, n_neighbors:] = mask[i, 0]

    print(f"[NOISE] Built noise mask (level={noise_level}): "
          f"{n_primary} primary edges (up to +{noise_level*100:.0f}%), "
          f"{n_secondary} secondary edges (up to +{1.5*noise_level*100:.0f}%)")

    return mask

def load_or_compute_distance_matrix_parallel(G, node_to_idx, cache_file="manhattan_distances.pkl", num_workers=None):
    """
    Load precomputed distance matrix or compute and cache it using parallel processing.
    """
    if cache_file is not None and os.path.exists(cache_file):
        # print(f"Loading precomputed distance matrix from {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            # Return None for paths_dict to use smart greedy approach
            return data['dist_mat'], data['hop_dist_mat'], data['max_length'], None
    
    # Compute distance matrix with parallel processing
    all_nodes = list(G.nodes())
    N = len(all_nodes)
    
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
            # Travel time distances - convert to seconds
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
    
    # Combine results
    for i, (local_dist, local_hop, local_max) in enumerate(results):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, N)
        dist_mat[start_idx:end_idx] = local_dist
        hop_dist_mat[start_idx:end_idx] = local_hop
        max_length = max(max_length, local_max)
    
    # Cache the results - only if cache_file is provided
    if cache_file is not None:
        # print(f"Caching distance matrix to {cache_file} (hybrid approach)")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'dist_mat': dist_mat,
                'hop_dist_mat': hop_dist_mat,
                'max_length': max_length,
            }, f)
    
    return dist_mat, hop_dist_mat, max_length, None

def load_or_build_graph(args, cache_file="manhattan_graph.pkl"):
    """Load cached graph or build and cache it."""
    if cache_file is not None and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            traffic_params = data['traffic_params']
            
            # Check if traffic_params is in wrong format and needs conversion
            if isinstance(traffic_params, dict):
                # Rebuild traffic params in correct format
                G = data['G']
                node_to_idx = data['node_to_idx']
                max_deg = max(dict(G.out_degree()).values())
                periods, green_durations, offsets = build_traffic_params(
                    G, node_to_idx, args.cycle_length, args.offset, seed=42, max_deg=max_deg, random_offsets=getattr(args, 'random_offsets', False)
                )
                traffic_params = (periods, green_durations, offsets)
                # Update cache
                with open(cache_file, 'wb') as cache_f:
                    data['traffic_params'] = traffic_params
                    pickle.dump(data, cache_f)
            
            # If seed is provided, recompute starts/pickups
            # Otherwise use cached values
            seed = getattr(args, 'seed', None)
            if seed is not None and args.env_type == 'manhattan':
                # Recompute starts/pickups
                G = data['G']
                nodes_gdf = data.get('nodes_gdf')
                node_to_zone = data.get('node_to_zone')
                zone_to_nodes = data.get('zone_to_nodes')
                
                # If we don't have the metadata, reload it
                if nodes_gdf is None or node_to_zone is None or zone_to_nodes is None:
                    G, nodes_gdf, node_to_zone, zone_to_nodes, zone_name_to_locationID = load_graph(
                        place_name=args.place_name, 
                        zone_shp=args.zone_shp, 
                        no_congestion=args.no_congestion
                    )
                else:
                    # Get zone_name_to_locationID from cache or reload if needed
                    zone_name_to_locationID = data.get('zone_name_to_locationID')
                    if zone_name_to_locationID is None:
                        # Reload just to get the mapping
                        _, _, _, _, zone_name_to_locationID = load_graph(
                            place_name=args.place_name, 
                            zone_shp=args.zone_shp, 
                            no_congestion=args.no_congestion
                        )
                
                start_zones = getattr(args, 'start_zones', None)
                pickup_zones = getattr(args, 'pickup_zones', None)
                
                fixed_starts_idx, fixed_pickups_idx, node_to_idx, idx_to_node = fixed_starts_pickups(
                    G,
                    nodes_gdf,
                    node_to_zone,
                    zone_to_nodes,
                    all=args.all_nodes_starts_pickups,
                    seed=seed,
                    start_zones=start_zones,
                    pickup_zones=pickup_zones,
                    zone_name_to_locationID=zone_name_to_locationID,
                )
                fixed_starts_idx = jnp.array(fixed_starts_idx, dtype=jnp.int32)
                fixed_pickups_idx = jnp.array(fixed_pickups_idx, dtype=jnp.int32)
                return data['G'], node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params
            
            G = data['G']
            node_to_idx = data['node_to_idx']
            idx_to_node = data['idx_to_node']
            fixed_starts_idx = data['fixed_starts_idx']
            fixed_pickups_idx = data['fixed_pickups_idx']

            if getattr(args, 'all_nodes_starts_pickups', False):
                num_nodes = len(node_to_idx)
                fixed_starts_idx = jnp.arange(num_nodes, dtype=jnp.int32)
                fixed_pickups_idx = jnp.arange(num_nodes, dtype=jnp.int32)

            return G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params
    
    # If not cached, build the graph
    start_time = time.time()
    
    G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params = build_env(args)
    
    build_time = time.time() - start_time

    # If specified, cache the graph
    if cache_file is not None:
        # Get zone metadata for caching (if Manhattan env)
        # We need to get this from build_env, but since build_env is called separately,
        # we'll reload it here for caching purposes
        nodes_gdf = None
        node_to_zone = None
        zone_to_nodes = None
        zone_name_to_locationID = None
        if args.env_type == 'manhattan':
            # Reload to get zone metadata for caching
            _, nodes_gdf, node_to_zone, zone_to_nodes, zone_name_to_locationID = load_graph(
                place_name=args.place_name,
                zone_shp=args.zone_shp,
                no_congestion=args.no_congestion
            )
        
        # Cache graph
        with open(cache_file, 'wb') as f:
            cache_data = {
                'G': G,
                'node_to_idx': node_to_idx,
                'idx_to_node': idx_to_node,
                'fixed_starts_idx': fixed_starts_idx,
                'fixed_pickups_idx': fixed_pickups_idx,
                'traffic_params': traffic_params
            }
            # Add zone metadata if available
            if nodes_gdf is not None:
                cache_data['nodes_gdf'] = nodes_gdf
            if node_to_zone is not None:
                cache_data['node_to_zone'] = node_to_zone
            if zone_to_nodes is not None:
                cache_data['zone_to_nodes'] = zone_to_nodes
            if zone_name_to_locationID is not None:
                cache_data['zone_name_to_locationID'] = zone_name_to_locationID
            pickle.dump(cache_data, f)
    
    return G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params


def build_env(args):
    """
    Build graph G, node_to_idx mapping, fixed start/pickup indices.
    Returns: G, node_to_idx, fixed_starts_idx, fixed_pickups_idx
    """
    if args.env_type == 'manhattan':
        # --- Load and preprocess Manhattan graph ---
        G, nodes_gdf, node_to_zone, zone_to_nodes, zone_name_to_locationID = load_graph(place_name = args.place_name, zone_shp = args.zone_shp, no_congestion= args.no_congestion)

        seed = getattr(args, 'seed', None)
        start_zones = getattr(args, 'start_zones', None)
        pickup_zones = getattr(args, 'pickup_zones', None)
                
        fixed_starts_idx, fixed_pickups_idx, node_to_idx, idx_to_node = fixed_starts_pickups(
            G,
            nodes_gdf,
            node_to_zone,
            zone_to_nodes,
            all=args.all_nodes_starts_pickups,
            seed=seed,
            start_zones=start_zones,
            pickup_zones=pickup_zones,
            zone_name_to_locationID=zone_name_to_locationID,
        )
        fixed_starts_idx  = jnp.array(fixed_starts_idx, dtype=jnp.int32)
        fixed_pickups_idx = jnp.array(fixed_pickups_idx, dtype=jnp.int32)
        print(f"Fixed starts: {fixed_starts_idx}")
        print(f"Fixed pickups: {fixed_pickups_idx}")

    elif args.env_type == 'simple':
        G = load_simple_graph(num_layers=args.num_layers, width=args.layer_width, no_congestion=args.no_congestion)
        # Index mappings
        nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        idx_to_node = [n for n, _ in sorted(node_to_idx.items(), key=lambda x: x[1])]

        # Only one fixed start (node 0) and one fixed pickup (last one)
        fixed_starts_idx = jnp.array([node_to_idx[n] for n in nodes[:args.layer_width]], dtype=jnp.int32)
        fixed_pickups_idx = jnp.array([node_to_idx[n] for n in nodes[-args.layer_width:]], dtype=jnp.int32)

    else:
        raise ValueError(f"Unknown env_type: {args.env_type}")

    if getattr(args, 'all_nodes_starts_pickups', False):
        num_nodes = len(node_to_idx)
        fixed_starts_idx = jnp.arange(num_nodes, dtype=jnp.int32)
        fixed_pickups_idx = jnp.arange(num_nodes, dtype=jnp.int32)
    
    # Compute max_deg for traffic params
    max_deg = max(dict(G.out_degree()).values())
    periods, green_durations, offsets = build_traffic_params(
        G, node_to_idx, args.cycle_length, args.offset, seed=42, max_deg=max_deg, random_offsets=getattr(args, 'random_offsets', False)
    )
    traffic_params = (periods, green_durations, offsets)

    return G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params


def compute_normalized_node_coordinates(
    G: nx.DiGraph,
    node_to_idx: Dict[int, int]
) -> jnp.ndarray:
    """
    Compute normalized lat/lon coordinates for all nodes in the graph.
    
    Normalizes coordinates based on the min/max of the filtered nodes in G
    (i.e., nodes from the selected zones). After normalization:
    - min lat/lon becomes 0
    - max lat/lon becomes 1
    - All other values are in [0, 1]
    
    Args:
        G: Graph containing only the filtered nodes (from selected zones)
        node_to_idx: Mapping from node IDs to indices
        
    Returns:
        [num_nodes, 2] array of normalized coordinates [lat, lon] in [0, 1]
    """
    idx_to_node = [n for n, _ in sorted(node_to_idx.items(), key=lambda x: x[1])]
    lats = jax.device_put(jnp.array([G.nodes[n]['y'] for n in idx_to_node], dtype=jnp.float32))
    lons = jax.device_put(jnp.array([G.nodes[n]['x'] for n in idx_to_node], dtype=jnp.float32))
    
    # Find min/max for the filtered nodes
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    
    # Normalize: (value - min) / (max - min)
    # Handle edge case where all nodes have same lat/lon (avoid division by zero)
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    
    # Normalize latitude: min -> 0, max -> 1
    normalized_lats = jnp.where(
        lat_range > 1e-8,  # If range is non-zero
        (lats - lat_min) / lat_range,
        jnp.zeros_like(lats)  # If all same, set to 0
    )
    
    # Normalize longitude: min -> 0, max -> 1
    normalized_lons = jnp.where(
        lon_range > 1e-8,  # If range is non-zero
        (lons - lon_min) / lon_range,
        jnp.zeros_like(lons)  # If all same, set to 0
    )
    
    latlon = jax.device_put(jnp.stack([
        normalized_lats,
        normalized_lons
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
      - single_obs: TaxiState -> observation vector [5] 
      - obs_fn_batch: batched TaxiState -> batched observation vectors [B, 5]
    """
    # Precompute normalized lat/lon per node
    latlon = compute_normalized_node_coordinates(G, node_to_idx)

    def single_obs(s: TaxiState) -> jnp.ndarray:
        """
        Observation function
        """
        # Use pre-computed normalized lat/lon coordinates
        xy_c = latlon[s.current_node]   # [2] - direct GPU memory access
        xy_p = latlon[s.pickup_node]    # [2] - direct GPU memory access
        
        # Compute relative position (direction vector from current to pickup)
        # relative_pos = xy_p - xy_c  # [2] - single operation
        
        # Compute distance and angle efficiently (avoid expensive operations)
        # distance_sq = jnp.sum(relative_pos * relative_pos)  # [1] - faster than linalg.norm
        # distance = jnp.sqrt(distance_sq + 1e-8)  # [1] - add small epsilon for numerical stability
        
        # # Compute angle efficiently
        # angle = jnp.arctan2(relative_pos[1], relative_pos[0])  # [1] - direct computation
        
        # Pre-allocate result array for better memory layout
        obs = jnp.zeros(5, dtype=jnp.float32)
        
        # Fill in values with optimized assignments
        obs = obs.at[0:2].set(xy_c)           # [2] - current position
        obs = obs.at[2:4].set(xy_p)           # [2] - pickup position  
        # obs = obs.at[4:6].set(relative_pos)   # [2] - direction vector
        # obs = obs.at[6].set(distance)         # [1] - distance
        # obs = obs.at[7].set(angle)            # [1] - angle
        obs = obs.at[4].set(s.time)           # [1] - time
        # obs = obs.at[4].set(0.0)                # [1] - time
        
        return obs

    single_obs = jax.jit(single_obs)
    
    # Pre-compile the vmap
    single_obs_batched = jax.jit(jax.vmap(single_obs))

    @jax.jit
    def obs_fn_batch(batch: TaxiState) -> jnp.ndarray:
        """Batch observation function"""
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

# Example init_state_fn
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
                         max_deg: int = None,
                         random_offsets: bool = False,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each edge (at the end of each edge), look at the highway type of that edge
    and pick a cycle/green split:
      • Motorway/Trunk: always green (no light)
      • Primary:         cycle/green = 60/30 (50% green)
      • Secondary:       cycle/green = 60/18 (30% green)
      • Tertiary:        cycle/green = 60/60 (always green)
      • Residential/Living street: cycle/green = 60/18 (30% green)
      • Otherwise:       cycle/green = 60/60 (always green)
    
    Args:
        random_offsets: If True, each traffic light gets a random offset between 0 and cycle_length.
                       If False, all traffic lights use the same offset value.
    
    Returns arrays of shape [num_nodes, max_deg] for periods, green_durations, and offsets.
    """
    node_list = list(G.nodes())
    num_nodes = len(node_list)
    
    if max_deg is None:
        max_deg = max(dict(G.out_degree()).values())
    
    periods = np.zeros((num_nodes, max_deg), dtype=np.float32)
    green_durations = np.zeros((num_nodes, max_deg), dtype=np.float32)
    offsets_arr = np.zeros((num_nodes, max_deg), dtype=np.float32)
    
    # Initialize random number generator for random offsets if needed
    if random_offsets:
        rng = np.random.RandomState(seed)
    
    for i, node in enumerate(node_list):
        neighbors = list(G.successors(node))
        n_neighbors = len(neighbors)
        for j, nbr in enumerate(neighbors[:max_deg]):
            # Get the best edge (same logic as build_adj_and_time_matrix)
            best_k = min(G[node][nbr], key=lambda k: G[node][nbr][k]['travel_time_congested'])
            edge_data = G[node][nbr][best_k]
            
            # Get highway type from this specific edge
            hw = edge_data.get("highway", "unclassified")
            if isinstance(hw, list):
                hw = hw[0] if hw else "unclassified"
            
            # decide cycle & green based on edge's highway type
            if hw in ("motorway", "trunk"):
                cycle, green = cycle_length, cycle_length    # effectively always green
            elif hw == "primary":
                cycle, green = cycle_length, cycle_length * 0.5
            elif hw == "secondary":
                cycle, green = cycle_length, cycle_length * 0.3
            elif hw == "tertiary":
                cycle, green = cycle_length, cycle_length
            elif hw in ("residential", "living_street"):
                cycle, green = cycle_length, cycle_length * 0.3
            else:
                cycle, green = cycle_length, cycle_length
            
            periods[i, j] = cycle
            green_durations[i, j] = green
            if random_offsets:
                # Generate random integer offset between 0 and cycle_length (exclusive)
                offsets_arr[i, j] = float(rng.randint(0, int(cycle_length)))
            else:
                offsets_arr[i, j] = offset
        
        # Pad with first edge's params if needed (same as adj_list padding)
        if 0 < n_neighbors < max_deg:
            periods[i, n_neighbors:] = periods[i, 0]
            green_durations[i, n_neighbors:] = green_durations[i, 0]
            offsets_arr[i, n_neighbors:] = offsets_arr[i, 0]
    
    return periods, green_durations, offsets_arr
