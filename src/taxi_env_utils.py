import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import networkx as nx
from taxi_env import TaxiEnv, TaxiState
from typing import Callable, Dict, Tuple

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

    return jnp.array(adj), jnp.array(times), jnp.array(neighbor_mask)

def make_obs_fn(
    env: TaxiEnv,
    G: nx.DiGraph,
    node_to_idx: Dict[int,int]
) -> Tuple[
    Callable[[TaxiState], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    Callable[[TaxiState], Dict[str, jnp.ndarray]]
]:
    """
    Returns two functions:
      - single_obs: TaxiState -> (state_feats [6], action_feats [max_deg,2], global_feats [3*N])
      - obs_fn_batch: batched TaxiState -> dict of trajectories
    """
    # Precompute normalized lat/lon per node
    idx_to_node = [n for n, _ in sorted(node_to_idx.items(), key=lambda x: x[1])]
    lats = jnp.array([G.nodes[n]['y'] for n in idx_to_node], dtype=jnp.float32)
    lons = jnp.array([G.nodes[n]['x'] for n in idx_to_node], dtype=jnp.float32)
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    latlon = jnp.stack([
        (lats - lat_min) / (lat_max - lat_min),
        (lons - lon_min) / (lon_max - lon_min)
    ], axis=-1)  # [N,2]

    def single_obs(s: TaxiState) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # -- State features --
        xy_c = latlon[s.current_node]   # [2]
        xy_p = latlon[s.pickup_node]    # [2]
        time = jnp.expand_dims(s.time, axis=-1)
        obs = jnp.concatenate([xy_c, xy_p, time], axis=-1)

        return obs

    # JIT and batched versions
    single_obs = jax.jit(single_obs)
    single_obs_batched = jax.vmap(single_obs)

    @jax.jit
    def obs_fn_batch(batch: TaxiState) -> Dict[str, jnp.ndarray]:
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
            cycle, green = cycle_length, 5.0/6.0 * cycle_length
        elif any(t == "secondary" for t in types):
            cycle, green = cycle_length, 2.0/3.0 * cycle_length
        elif any(t == "tertiary" for t in types):
            cycle, green = cycle_length, 1.0/6.0 * cycle_length
        elif any(t in ("residential", "living_street") for t in types):
            cycle, green = cycle_length, 1.0/6.0 * cycle_length
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
