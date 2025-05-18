import numpy as np
import jax
import jax.numpy as jnp
import networkx as nx
from taxi_env import JAXRideEnv, TaxiState
from typing import Callable, Dict, Tuple

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
            time_sec = G[node][nbr][best_k]['travel_time_congested'] * 60.0  # seconds
            adj[i, j] = node_to_idx[nbr]
            times[i, j] = time_sec
        if 0 < n_neighbors < max_deg:
            # pad with first neighbor
            adj[i, n_neighbors:] = adj[i, 0]
            times[i, n_neighbors:] = times[i, 0]
        neighbor_mask[i, :n_neighbors] = 1.0

    return jnp.array(adj), jnp.array(times), jnp.array(neighbor_mask)

# --- Observation extraction function ---
def make_obs_fn(env: JAXRideEnv,
                G: nx.DiGraph,
                node_to_idx: Dict[int,int],
                max_steps: int
               ) -> Callable[[TaxiState], Dict[str, jnp.ndarray]]:
    """
    Returns a batched obs_fn that maps a TaxiState batch to:
      - state_feats: [B, 7]  (curr_xy, pickup_xy, delta_xy, norm_step)
      - action_feats: [B, max_deg, 2]  (travel_time, expected_wait)
    """
    # Precompute normalized lat/lon per node
    idx_to_node = [n for n, _ in sorted(node_to_idx.items(), key=lambda x: x[1])]
    lats = jnp.array([G.nodes[n]['y'] for n in idx_to_node], dtype=jnp.float32)
    lons = jnp.array([G.nodes[n]['x'] for n in idx_to_node], dtype=jnp.float32)
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    latlon = jnp.stack([(lats - lat_min)/(lat_max - lat_min),
                        (lons - lon_min)/(lon_max - lon_min)], axis=-1)  # [N,2]

    max_deg = env.max_deg

    def single_obs(s: TaxiState):
        # --- State-level features ---
        curr, drop = s.current_node, s.pickup_node
        xy_c = latlon[curr]                  # [2]
        xy_p = latlon[drop]                  # [2]
        delta = xy_p - xy_c                  # [2]
        step = jnp.array(s.step_count, dtype=jnp.float32)
        norm_step = (step / max_steps)[..., None]
        state_feats = jnp.concatenate([xy_c, xy_p, delta, norm_step], axis=-1)  # [7]

        # --- Per-action features ---
        nbrs    = env.adj_list[curr]         # [max_deg]
        travel  = env.travel_times[curr]     # [max_deg]
        # broadcast s.time to match [max_deg]
        time = jnp.array(s.time, dtype=jnp.float32)        # shape=()
        arrival = time[..., None] + travel
        per     = env.periods[nbrs]          # [max_deg]
        green   = env.green_durations[nbrs]  # [max_deg]
        offset  = env.offsets[nbrs]          # [max_deg]
        cycle   = (arrival + offset) % per    # [max_deg]
        wait    = jnp.where(cycle < green, 0.0, per - cycle)  # [max_deg]
        action_feats = jnp.stack([travel, wait], axis=-1)     # [max_deg,2]

        return state_feats, action_feats

    @jax.jit
    def obs_fn(batch: TaxiState):
        sf, af = jax.vmap(single_obs)(batch)
        return {'state_feats': sf, 'action_feats': af}

    return single_obs, obs_fn

# --- Example init_state_fn ---
def make_init_state_fn(env: JAXRideEnv, num_envs: int, rng_key):
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
                         seed: int = 0
                        ) -> Dict[int, Tuple[float,float,float]]:
    """
    For each intersection (node), look at the highway‐types of
    all incident edges and pick a cycle/green split:
      • Motorway/Trunk: always green (no light)
      • Primary:         60s cycle, 30s green
      • Secondary:       60s cycle, 25s green
      • Tertiary:        50s cycle, 20s green
      • Otherwise:       40s cycle, 20s green
    We then pick a random offset in [0, cycle) so lights aren’t all in lock‐step.
    """
    rng = np.random.default_rng(seed)
    traffic_params = {}
    for node, idx in node_to_idx.items():
        # collect all highway‐types on edges touching this node
        types = []
        # For a MultiDiGraph:
        for u, v, key, data in G.edges(node, keys=True, data=True):
            hw = data.get("highway", "unclassified")
            # OSM sometimes gives a list of types
            if isinstance(hw, list):
                types.extend(hw)
            else:
                types.append(hw)
        # decide cycle & green based on priority
        if any(t in ("motorway", "trunk") for t in types):
            cycle, green = 1.0, 1.0    # effectively always green
        elif any(t == "primary" for t in types):
            cycle, green = 60.0, 30.0
        elif any(t == "secondary" for t in types):
            cycle, green = 60.0, 25.0
        elif any(t == "tertiary" for t in types):
            cycle, green = 50.0, 20.0
        else:
            cycle, green = 40.0, 20.0

        # random phase offset so lights aren’t synced
        offset = float(rng.uniform(0, cycle))
        # --- VALIDITY CHECKS ---
        assert cycle > 0, f"Cycle length for node {node} must be positive, got {cycle}"
        assert green > 0, f"Green duration must be positive, got {green}"
        assert green <= cycle, (
            f"Green duration ({green}) exceeds cycle ({cycle}) at node {node}"
        )
        assert 0 <= offset < cycle, (
            f"Offset {offset:.2f} not in [0, {cycle}) for node {node}"
        )

        traffic_params[node_to_idx[node]] = (cycle, green, offset)

    # ensure coverage
    missing = set(node_to_idx.values()) - set(traffic_params.keys())
    assert not missing, f"Missing params for node indices: {sorted(missing)}"

    return traffic_params
