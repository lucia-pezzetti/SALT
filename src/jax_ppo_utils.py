import numpy as np
import jax
import jax.numpy as jnp
import networkx as nx
from jax_taxi_env import init_env

# --- Graph conversion utilities ---
def build_adj_and_time_matrix(G: nx.DiGraph, max_deg=None, node_to_idx: dict = None):
    node_list = list(G.nodes())
    num_nodes = len(node_list)

    if max_deg is None:
        max_deg = max(dict(G.out_degree()).values())

    adj = -np.ones((num_nodes, max_deg), dtype=int)
    times = np.zeros((num_nodes, max_deg), dtype=np.float32)
    neighbor_mask = np.zeros((num_nodes, max_deg), dtype=np.float32)


    for i, node in enumerate(node_list):
        neighbors = list(G.successors(node))
        n_neighbors = len(neighbors)
        
        for j, nbr in enumerate(neighbors[:max_deg]):
            best_k = min(G[node][nbr], key=lambda k: G[node][nbr][k]['travel_time_congested'])
            time = G[node][nbr][best_k]['travel_time_congested'] * 60  # seconds
            adj[i, j] = node_to_idx[nbr]
            times[i, j] = time
        
        # Padding (only if needed)
        if n_neighbors > 0 and n_neighbors < max_deg:
            adj[i, n_neighbors:] = adj[i, 0]
            times[i, n_neighbors:] = times[i, 0]

        # Neighbor mask: 1 for real neighbors, 0 for padded
        neighbor_mask[i, :n_neighbors] = 1.0

    return jnp.array(adj), jnp.array(times), jnp.array(neighbor_mask)


# --- Observation extraction and normalisation function ---
def make_obs_fn(G: nx.DiGraph, node_to_idx: dict, max_steps: int):
    """Observation function using lat/lon with delta and normalized step."""
    
    idx_to_node = [node for node, idx in sorted(node_to_idx.items(), key=lambda x: x[1])]
    lats = jnp.array([G.nodes[n]['y'] for n in idx_to_node], dtype=jnp.float32)
    lons = jnp.array([G.nodes[n]['x'] for n in idx_to_node], dtype=jnp.float32)

    lat_min, lat_max = jnp.min(lats), jnp.max(lats)
    lon_min, lon_max = jnp.min(lons), jnp.max(lons)

    # Normalize lat/lon to [0,1]
    norm_lats = (lats - lat_min) / (lat_max - lat_min)
    norm_lons = (lons - lon_min) / (lon_max - lon_min)

    # Stack them for easy indexing: shape [N, 2]
    latlon = jnp.stack([norm_lats, norm_lons], axis=-1)

    def get_xy(index):
        return latlon[index]

    def single_obs(s):
        current_xy = get_xy(s.current_node)
        pickup_xy = get_xy(s.pickup_node)
        norm_step = jnp.array([s.step_count / max_steps], dtype=jnp.float32)

        return jnp.concatenate([current_xy, pickup_xy, norm_step], axis=-1)
    
    @jax.jit
    def obs_fn_batch(states):
        return jax.vmap(single_obs)(states)

    @jax.jit
    def obs_fn(state):
        return jax.vmap(single_obs)(state)

    return obs_fn, obs_fn_batch

# --- Example init_state_fn ---
def make_init_state_fn(fixed_starts, fixed_pickups, distances, num_envs):
    def init_state():
        return [init_env(None, fixed_starts, fixed_pickups, distances) for _ in range(num_envs)]
    return init_state
