import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def load_graph(place_name: str, network_type: str = "drive") -> nx.DiGraph:
    """
    Download and preprocess the OSMnx graph for a place, keeping only the
    largest strongly-connected component and adding 'congested_time'.
    """
    # 1) Download & basic routing attributes
    G = ox.graph.graph_from_place(place_name, network_type=network_type)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # 2) Extract largest SCC
    if not nx.is_strongly_connected(G):
        sccs = nx.strongly_connected_components(G)
        largest = max(sccs, key=len)
        G_scc = G.subgraph(largest).copy()
    else:
        G_scc = G.copy()

    # 3) Alias travel_time → congested_time
    for u, v, k, data in G_scc.edges(keys=True, data=True):
        data['congested_time'] = data.get('travel_time', data.get('length', 0) / 10)

    return G_scc


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
                          zone_shp_path: str) -> (dict, dict, gpd.GeoDataFrame):
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


def compute_distributions(filtered_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
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
