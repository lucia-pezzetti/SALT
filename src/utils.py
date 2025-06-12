import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point


def load_graph(place_name: str, zone_shp: str, network_type: str = "drive") -> nx.DiGraph:
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

    apply_congestion_model(G_scc)

    # Map zones to nodes, then filter to Financial District
    locationID_to_nodes, zone_to_nodes, node_to_zone, nodes_gdf = compute_zone_mappings(G_scc, zone_shp_path=zone_shp)
    zone_names = ["Financial District South", "Financial District North", "Battery Park"]
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
        print(f"Edge {u} → {v} (key={k}): speed={speed_kph} kph, length={length_m} m")
        t_free_min = (length_m/1000)/speed_kph*60
        hw = data.get('highway', 'default')
        if isinstance(hw, list): hw = hw[0]
        alpha = multipliers.get(hw, multipliers['default'])
        t_cong_min = t_free_min * alpha
        data['travel_time_freeflow'] = t_free_min
        data['travel_time_congested'] = t_cong_min
        data['bpr_alpha'] = alpha
        print(f"  Freeflow time: {data['travel_time_freeflow']:.2f} min, Congested time: {data['travel_time_congested']:.2f} min, alpha: {data['bpr_alpha']}")


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
    print(f"Number of nodes: {len(all_nodes)}")
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
def load_simple_graph():
    # Fixed coords
    coords = {
        0: (0.0, 1.0),   # start
        1: (1.0, 0.0),   # inter A.1
        2: (1.0, 1.0),   # inter B.1
        3: (1.0, 2.0),   # inter C.1
        4: (2.0, 0.0),   # inter A.2
        5: (2.0, 1.0),   # inter B.2
        6: (2.0, 2.0),   # inter C.2
        # 7: (3.0, 0.0),   # inter A.3
        # 8: (3.0, 1.0),   # inter B.3
        # 9: (3.0, 2.0),   # inter C.3
        # 10: (4.0, 0.0),  # inter A.4
        # 11: (4.0, 1.0),  # inter B.4
        # 12: (4.0, 2.0),  # inter C.4
        # 13: (5.0, 0.0),  # inter A.5
        # 14: (5.0, 1.0),  # inter B.5
        # 15: (5.0, 2.0),   # inter C.5
        # 16: (6.0, 1.0),  # end
        7: (3.0, 1.0),  # end
    }

    # “layers” of nodes
    layers = [
        [0],           # start
        [1, 2, 3],     # first intermediates
        [4, 5, 6],     # second intermediates
        # [7, 8, 9],     # third intermediates
        # [10, 11, 12],  # fourth intermediates
        # [13, 14, 15],  # fifth intermediates
        # [16]           # end
        [7],          # end
    ]

    # two special routes to break shortest path
    primary_edges = {(0, 1), (1, 0), (4, 7), (7, 4)}
    tertiary_edges = {(0, 2), (2, 0), (5, 7), (7, 5)}


    # # build quick lookup sets of directed edges
    # def make_edge_set(path):
    #     return {
    #         (u, v) for u, v in zip(path, path[1:])
    #     } | {
    #         (v, u) for u, v in zip(path, path[1:])
    #     }

    # primary_edges  = make_edge_set(primary_route)
    # tertiary_edges = make_edge_set(tertiary_route)

    def tag_for(u, v):
        if (u, v) in primary_edges:
            return "primary"
        if (u, v) in tertiary_edges:
            return "tertiary"
        return "secondary"

    G = nx.MultiDiGraph()
    # add all nodes
    for nid, (x,y) in coords.items():
        G.add_node(nid, x=x, y=y)

    def connect_layer(i):
        if i >= len(layers) - 1:
            return
        src, dst = layers[i], layers[i+1]
        for u in src:
            for v in dst:
                dx, dy = coords[v][0] - coords[u][0], coords[v][1] - coords[u][1]
                length = np.hypot(dx, dy) * 1000.0
                hw = tag_for(u, v)
                G.add_edge(u, v, length=length, highway=hw)
                G.add_edge(v, u, length=length, highway=hw)
        connect_layer(i+1)


    connect_layer(0)

    apply_congestion_model(G)
    return G