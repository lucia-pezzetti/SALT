import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from typing import Callable, Dict, Tuple

import jax
from jax import numpy as jnp
from flax import linen as nn
from flax import struct

from taxi_env_utils import build_traffic_params
from taxi_env import TaxiEnv

# --- Load and preprocess graph ---
def build_env(args):
    """
    Build graph G, node_to_idx mapping, fixed start/pickup indices.
    Returns: G, node_to_idx, fixed_starts_idx, fixed_pickups_idx
    """
    if args.env_type == 'manhattan':
        # --- Load and preprocess Manhattan graph ---
        G, nodes_gdf, node_to_zone, zone_to_nodes = load_graph(place_name = args.place_name, zone_shp = args.zone_shp, no_congestion= args.no_congestion)

        fixed_starts_idx, fixed_pickups_idx, node_to_idx, idx_to_node = fixed_starts_pickups(
            G, nodes_gdf, node_to_zone, zone_to_nodes, all = True
        )
        fixed_starts_idx  = jnp.array(fixed_starts_idx, dtype=jnp.int32)
        fixed_pickups_idx = jnp.array(fixed_pickups_idx, dtype=jnp.int32)

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


def load_graph(place_name: str, zone_shp: str, network_type: str = "drive", no_congestion: bool = False) -> nx.DiGraph:
    """
    Download and preprocess the OSMnx graph for a place, keeping only the
    largest strongly-connected component and adding 'congested_time'.
    Removes self-loops from the graph.
    """
    # Download & basic routing attributes
    G = ox.graph.graph_from_place(place_name, network_type=network_type)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Extract largest SCC
    if not nx.is_strongly_connected(G):
        sccs = nx.strongly_connected_components(G)
        largest = max(sccs, key=len)
        G_scc = G.subgraph(largest).copy()
    else:
        G_scc = G.copy()

    # Remove self-loops
    G_scc.remove_edges_from(nx.selfloop_edges(G_scc))

    # Alias travel_time → congested_time
    for u, v, k, data in G_scc.edges(keys=True, data=True):
        data['congested_time'] = data.get('travel_time', data.get('length', 0) / 10)

    if no_congestion:
        multipliers = {
            'motorway': 1, 'trunk': 1, 'primary': 1,
            'secondary': 1, 'tertiary': 1, 'residential': 1,
            'service': 1, 'living_street': 1, 'default': 1
        }
    else:
        multipliers = {
            'motorway': 1.2, 'trunk': 1.3, 'primary': 1.5,
            'secondary': 1.7, 'tertiary': 1.9, 'residential': 2.0,
            'service': 2.2, 'living_street': 2.5, 'default': 2.0
        }
    apply_congestion_model(G_scc, multipliers)

    # Map zones to nodes, then filter to Financial District
    locationID_to_nodes, zone_to_nodes, node_to_zone, nodes_gdf = compute_zone_mappings(G_scc, zone_shp_path=zone_shp)
    # zone_names = ["Financial District South", "Financial District North", "Battery Park", "Battery Park City", "World Trade Center", "Seaport", "TriBeCa/Civic Center", "Chinatown", "Lower East Side", "Two Bridges/Seward Park", "Little Italy/NoLiTa", "SoHo", "Hudson Sq", "Alphabet City", "East Village", "Greenwich Village South", "Greenwich Village North", "West Village", "Meatpacking/West Village West"] 
    zone_names = ["Upper East Side North", "Yorkville West", "Upper East Side South", "Lenox Hill East"]  
    gdf_zones = gpd.read_file(zone_shp).to_crs("EPSG:4326")
    filtered_zones = gdf_zones[gdf_zones["zone"].isin(zone_names)]
    loc_ids = filtered_zones["LocationID"].tolist()
    selected_nodes = [n for loc_id in loc_ids for n in zone_to_nodes.get(loc_id, [])]
    G_scc = G_scc.subgraph(selected_nodes).copy()

    # Prune trivial nodes
    to_remove = [
        v for v in G_scc.nodes()
        if (G_scc.in_degree(v) == 1 and G_scc.out_degree(v) == 1 and list(G_scc.predecessors(v))[0] == list(G_scc.successors(v))[0])
        or (G_scc.in_degree(v) == 1 and G_scc.out_degree(v) == 0)
        or (G_scc.in_degree(v) == 0 and G_scc.out_degree(v) == 1)
    ]
    G_scc.remove_nodes_from(to_remove)
    largest_cc = max(nx.strongly_connected_components(G_scc), key=len)
    G_scc = G_scc.subgraph(largest_cc).copy()

    return G_scc, nodes_gdf, node_to_zone, zone_to_nodes


def load_taxi_data(rides_path: str,
                   lookup_path: str,
                   target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Load yellow taxi CSVs, filter to Manhattan rides on target_date.
    Returns filtered pandas DataFrame.
    """
    df = pd.read_csv(rides_path, low_memory=False)
    zone_lookup = pd.read_csv(lookup_path)
    manhattan_ids = set(
        zone_lookup[zone_lookup['Borough']=='Manhattan']['LocationID']
    )

    df['pickup_dt'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['dropoff_dt'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    is_manhattan = (
        df['PULocationID'].isin(manhattan_ids) &
        df['DOLocationID'].isin(manhattan_ids)
    )
    is_same_date = (
    (df['pickup_dt'].dt.date == target_date.date()) &
    (df['dropoff_dt'].dt.date == target_date.date())
    )
    filtered = df[is_manhattan & is_same_date].copy()
    return filtered


def compute_zone_mappings(G_scc: nx.DiGraph,
                          zone_shp_path: str):
    """
    Build mappings from LocationID → list of node IDs and node → zone.
    Also returns the nodes GeoDataFrame (with 'zone' column).
    """
    # Nodes as GeoDataFrame
    nodes = ox.graph_to_gdfs(G_scc, nodes=True, edges=False)
    nodes['geometry'] = nodes.apply(
        lambda row: Point(row['x'], row['y']), axis=1
    )
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry='geometry', crs="EPSG:4326")

    # Load taxi zones shapefile
    gdf_zones = gpd.read_file(zone_shp_path).to_crs("EPSG:4326")

    # Spatial join
    nodes_in_zones = gpd.sjoin(
        nodes_gdf, gdf_zones, how='inner', predicate='within'
    )

    locationID_to_nodes = {
        loc_id: list(group.index)
        for loc_id, group in nodes_in_zones.groupby('LocationID')
    }

    # Build dicts
    zone_to_nodes = {
        zone: list(group.index)
        for zone, group in nodes_in_zones.groupby('LocationID')
    }
    node_to_zone = {
        node: zone for zone, nodes in zone_to_nodes.items() for node in nodes
    }
    nodes_gdf['zone'] = nodes_gdf.index.map(node_to_zone)

    return locationID_to_nodes, zone_to_nodes, node_to_zone, nodes_gdf


def apply_congestion_model(G_scc: nx.DiGraph,
                           multipliers: dict = None) -> None:
    """
    Compute freeflow and congested travel times on each edge using BPR-like alpha.
    Modifies G_scc in place, adding:
      - travel_time_freeflow (minutes)
      - travel_time_congested (minutes)
      - bpr_alpha
    """
    if multipliers is None:
        multipliers = {
            'motorway': 1.2, 'trunk': 1.3, 'primary': 1.5,
            'secondary': 1.7, 'tertiary': 1.9, 'residential': 2.0,
            'service': 2.2, 'living_street': 2.5, 'default': 2.0
        }

    for u, v, k, data in G_scc.edges(keys=True, data=True):
        speed_kph = data.get('speed_kph', 30)
        length_m = data.get('length', 0)
        t_free_min = (length_m/1000)/speed_kph*60
        hw = data.get('highway', 'default')
        if isinstance(hw, list): hw = hw[0]
        alpha = multipliers.get(hw, multipliers['default'])
        t_cong_min = t_free_min * alpha
        data['travel_time_freeflow'] = t_free_min
        data['travel_time_congested'] = t_cong_min
        data['bpr_alpha'] = alpha

def compute_distributions(filtered_df: pd.DataFrame):
    """
    From filtered taxi DataFrame, compute:
      - pickup_dist: DataFrame P(pickup_node | hour)
      - cond_dest_dist: DataFrame P(dropoff_node | hour, pickup_node)
    """
    filtered_df['pickup_hour'] = filtered_df['pickup_dt'].dt.hour
    # P(pickup | hour)
    pu_counts = (
        filtered_df
          .groupby(['pickup_hour','PULocationID'])
          .size()
          .unstack(fill_value=0)
    )
    pickup_dist = pu_counts.div(pu_counts.sum(axis=1), axis=0)

    # P(dest | hour, pickup)
    do_counts = (
        filtered_df
          .groupby(['pickup_hour','PULocationID','DOLocationID'])
          .size()
          .unstack(fill_value=0)
    )
    cond_dest_dist = do_counts.div(do_counts.sum(axis=1), axis=0)

    return pickup_dist, cond_dest_dist


def full_setup(place_name: str,
               rides_csv: str,
               lookup_csv: str,
               zone_shp: str,
               target_date: pd.Timestamp):
    """
    Convenience wrapper that returns:
      G_scc,
      zone_to_nodes,
      node_to_zone,
      nodes_gdf,
      filtered_df,
      pickup_dist,
      cond_dest_dist
    """
    G_scc = load_graph(place_name)
    filtered = load_taxi_data(rides_csv, lookup_csv, target_date)
    lid2n, z2n, n2z, nodes_gdf = compute_zone_mappings(G_scc, zone_shp)
    apply_congestion_model(G_scc)
    pu_dist, cd_dist = compute_distributions(filtered)
    return G_scc, lid2n, z2n, n2z, nodes_gdf, filtered, pu_dist, cd_dist


def fixed_starts_pickups(G: nx.DiGraph,
                         nodes_gdf: gpd.GeoDataFrame,
                         node_to_zone: dict,
                         zone_to_nodes: dict,
                         all: bool = False) -> tuple:
    """
    Choose fixed start & pickup nodes for the environment.
    If all=True, use all nodes as starts/pickups.
    Otherwise, use one node per zone.
    """

    all_nodes = list(G.nodes())
    # print(f"Number of nodes: {len(all_nodes)}")
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    idx_to_node = [n for n, _ in sorted(node_to_idx.items(), key=lambda x: x[1])]

    # Choose fixed start & pickup sets (here: all nodes)
    if all:
        fixed_starts_idx = list(range(len(all_nodes)))
        fixed_pickups_idx = list(range(len(all_nodes)))
    else:
        # choose fixed start & pickup nodes
        fixed_starts = []
        fixed_pickups = []

        nodes_gdf['zone'] = nodes_gdf.index.map(node_to_zone)

        # Fix a node for every zone
        for loc_id, nodes in zone_to_nodes.items():
            if nodes:
                # Use the first node in the list for each zone
                fixed_starts.append(nodes[0])
                fixed_pickups.append(nodes[0])

        fixed_starts_idx = [node_to_idx[int(n)] for n in fixed_starts if int(n) in node_to_idx]
        fixed_pickups_idx = [node_to_idx[int(n)] for n in fixed_pickups if int(n) in node_to_idx]

    return fixed_starts_idx, fixed_pickups_idx, node_to_idx, idx_to_node


# ---- debugging graph ----
def load_simple_graph(
        num_layers: int = 2, 
        width: int =3, 
        layer_spacing: float = 1.0, 
        node_spacing: float = 1.0, 
        no_congestion: bool = False
    ) -> nx.MultiDiGraph:
    """
    Build a “simple” layered grid with `num_layers` of 3-node rings:
      layer 0 = starts, layer 1 = first intermediates, etc.

    Args:
        num_layers: how many of the predefined 3-node layers to include (max 6)

    Returns:
        G: a MultiDiGraph containing only those layers, fully connected
           between consecutive layers, with lengths & highway tags.
    """
    # fixed coords for up to 6 layers of 3 nodes each
    # coords = {
    #     0: (0.0, 0.0),   1: (0.0, 1.0),   2: (0.0, 2.0),
    #     3: (1.0, 0.0),   4: (1.0, 1.0),   5: (1.0, 2.0),
    #     6: (2.0, 0.0),   7: (2.0, 1.0),   8: (2.0, 2.0),
    #     9: (3.0, 0.0),  10: (3.0, 1.0),  11: (3.0, 2.0),
    #    12: (4.0, 0.0),  13: (4.0, 1.0),  14: (4.0, 2.0),
    #    15: (5.0, 0.0),  16: (5.0, 1.0),  17: (5.0, 2.0),
    #    18: (6.0, 0.0),  19: (6.0, 1.0),  20: (6.0, 2.0),
    #    21: (7.0, 0.0),  22: (7.0, 1.0),  23: (7.0, 2.0),
    #    24: (8.0, 0.0),  25: (8.0, 1.0),  26: (8.0, 2.0),
    #    27: (9.0, 0.0),  28: (9.0, 1.0),  29: (9.0, 2.0),
    #    30: (10.0, 0.0), 31: (10.0, 1.0), 32: (10.0, 2.0),

    # }

    # # six hard-coded 3-node layers
    # all_layers = [
    #     [0, 1, 2],    # layer 0: starts
    #     [3, 4, 5],    # layer 1
    #     [6, 7, 8],    # layer 2
    #     [9,10,11],    # layer 3
    #     [12,13,14],   # layer 4
    #     [15,16,17],   # layer 5
    #     [18,19,20],   # layer 6
    #     [21,22,23],   # layer 7
    #     [24,25,26],   # layer 8
    #     [27,28,29],   # layer 9
    #     [30,31,32],   # layer 10
    # ]

    # 1) build coords & layer lists
    coords = {}
    layers = []
    node_id = 0
    for i in range(num_layers):
        ys = np.arange(width) * node_spacing
        xs = np.full(width, i * layer_spacing)
        layer = []
        for x, y in zip(xs, ys):
            coords[node_id] = (x, y)
            layer.append(node_id)
            node_id += 1
        layers.append(layer)

    # # clamp to available layers
    # if num_layers < 1 or num_layers > len(all_layers):
    #     raise ValueError(f"num_layers must be in [1..{len(all_layers)}], got {num_layers}")
    # layers = all_layers[:num_layers]

    # clamp to available coords
    # coords = {k: v for k, v in coords.items() if k < num_layers * 3}

    # (optional) define special highways if you need them
    # primary_edges = {(1,4), (4,7), (7,10), (10,13), (13,16), (16,19), (19,22), (22,25), (25,28), (28,31)}
    # secondary_edges = {}
    # tertiary_edges = {(0,3), (3,6), (6,9), (9,12), (12,15), (15,18), (18,21), (21,24), (24,27), (27,30), 
    #                   (2,5), (5,8), (8,11), (11,14), (14,17), (17,20), (20,23), (23,26), (26,29), (29,32)}
    # highway_edges = {}
    # residential_edges = {}

    mid = width // 2
    primary_edges   = {(layers[i][mid],   layers[i+1][mid])   for i in range(num_layers-1)}
    secondary_edges = set()
    tertiary_edges  = set()
    for w in range(width):
        if w == mid:
            continue
        tertiary_edges |= {(layers[i][w],    layers[i+1][w])    for i in range(num_layers-1)}
    highway_edges = set()
    residential_edges = set()


    def tag_for(u: int, v: int) -> str:
        if (u,v) in primary_edges:   return "primary"
        if (u,v) in secondary_edges:  return "secondary"
        if (u,v) in tertiary_edges:    return "tertiary"
        if (u,v) in highway_edges:   return "motorway"
        if (u,v) in residential_edges: return "residential"
        return "residential"

    G = nx.MultiDiGraph()
    # add nodes with positions
    for nid, (x,y) in coords.items():
        G.add_node(nid, x=x, y=y)

    # connect each layer i → i+1
    for i in range(len(layers)-1):
        src = layers[i]
        dst = layers[i+1]
        for u in src:
            for v in dst:
                dx, dy = coords[v][0] - coords[u][0], coords[v][1] - coords[u][1]
                length = np.hypot(dx, dy) * 1000.0
                hw = tag_for(u, v)
                G.add_edge(u, v, length=length, highway=hw)
                G.add_edge(v, u, length=length, highway=hw)

    # apply whatever congestion model you have
    if no_congestion:
        multipliers = {
            'motorway': 1, 'trunk': 1, 'primary': 1,
            'secondary': 1, 'tertiary': 1, 'residential': 1,
            'service': 1, 'living_street': 1, 'default': 1
        }
    else:
        multipliers = {
            'motorway': 1.2, 'trunk': 1.3, 'primary': 1.5,
            'secondary': 1.7, 'tertiary': 1.9, 'residential': 2.0,
            'service': 2.2, 'living_street': 2.5, 'default': 2.0
        }
    apply_congestion_model(G, multipliers)
    return G

@struct.dataclass
class EstimateReturnsState:
    """Pre-computed arrays to avoid recreation"""
    discounts: jnp.ndarray
    
    @classmethod
    def create(cls, rollout_steps: int, gamma: float):
        return cls(discounts=gamma ** jnp.arange(rollout_steps))

def estimate_returns(
    env: TaxiEnv,
    params,
    model: nn.Module,
    obs_fn_batch: Callable,
    init_env_fn: Callable,
    starts: jnp.ndarray,
    pickups: jnp.ndarray,
    estimate_state: EstimateReturnsState,  # Pre-computed arrays
    rollout_steps: int = 10,
) -> jnp.ndarray:
    N = starts.shape[0]
    
    # Use pre-computed discounts
    discounts = estimate_state.discounts
    
    # Rest of the function remains the same...
    grid_s, grid_p = jnp.meshgrid(starts, pickups, indexing="ij")
    s_rep = grid_s.ravel()
    p_rep = grid_p.ravel()
    
    states, _ = init_env_fn(s_rep, p_rep)
    acc_init = jnp.zeros(states.current_node.shape[0])
    carry_init = (states, acc_init)
    
    def body(carry, t_and_discount):
        t, discount = t_and_discount
        st, acc = carry
        obs = obs_fn_batch(st)
        # q = model.apply(params, obs['state_feats'], obs['action_feats'], 
        #                st.neighbor_mask, obs['global_feats'])
        q = model.apply(params, obs, st.neighbor_mask)
        act = jnp.argmax(q, axis=-1)
        nxt, r, _, _ = env.step(st, act)
        new_acc = acc + discount * r
        return (nxt, new_acc), None
    
    (final_state, total_ret), _ = jax.lax.scan(
        body,
        carry_init,
        (jnp.arange(rollout_steps), discounts)
    )
    return total_ret.reshape((N, N)).astype(jnp.float32)

# JIT with fixed static args
estimate_returns_jit = jax.jit(
    estimate_returns,
    static_argnums=(2, 3, 4, 8)  # model, obs_fn_batch, init_env_fn, rollout_steps
)