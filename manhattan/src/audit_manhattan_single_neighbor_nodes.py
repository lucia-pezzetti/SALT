#!/usr/bin/env python3
"""Audit and optionally prune non-start Manhattan nodes with one unique neighbor."""

import argparse
import ast
import csv
import pickle
import re
from pathlib import Path

import networkx as nx


SOUTH_MANHATTAN_ZONES = [
    "Alphabet City", "Battery Park", "Battery Park City", "Chinatown",
    "Clinton East", "Clinton West", "East Chelsea", "East Village",
    "Financial District North", "Financial District South", "Flatiron",
    "Garment District", "Gramercy", "Greenwich Village North",
    "Greenwich Village South", "Hudson Sq", "Kips Bay", "Little Italy/NoLiTa",
    "Lower East Side", "Meatpacking/West Village West", "Midtown Center",
    "Midtown East", "Midtown North", "Midtown South", "Murray Hill",
    "Penn Station/Madison Sq West", "Seaport", "SoHo",
    "Stuy Town/Peter Cooper Village", "Sutton Place/Turtle Bay North",
    "Times Sq/Theatre District", "TriBeCa/Civic Center",
    "Two Bridges/Seward Park", "UN/Turtle Bay South", "Union Sq",
    "West Chelsea/Hudson Yards", "West Village", "World Trade Center",
]


GRAPH_SIZE_RE = re.compile(r"Graph loaded:\s*(\d+)\s+nodes")
STARTS_RE = re.compile(r"Restricted starts \(3 fixed nodes, indices\):\s*(\[[^]]*\])")


def parse_training_log(path):
    text = Path(path).read_text(errors="replace")
    graph_matches = GRAPH_SIZE_RE.findall(text)
    start_matches = STARTS_RE.findall(text)
    if not graph_matches:
        raise ValueError(f"Could not find the graph node count in {path}")
    if not start_matches:
        raise ValueError(f"Could not find restricted start indices in {path}")
    starts = ast.literal_eval(start_matches[-1])
    if not isinstance(starts, list) or not all(isinstance(value, int) for value in starts):
        raise ValueError(f"Invalid restricted start list in {path}: {starts!r}")
    return int(graph_matches[-1]), starts


def unique_neighbors(graph, node):
    return set(graph.predecessors(node)) | set(graph.successors(node))


def single_neighbor_nonstarts(graph, protected_nodes):
    protected = set(protected_nodes)
    return [
        node
        for node in graph.nodes
        if node not in protected and len(unique_neighbors(graph, node)) == 1
    ]


def prune_candidates(graph, protected_nodes, recursive=False):
    pruned = graph.copy()
    removed = []
    while True:
        candidates = single_neighbor_nonstarts(pruned, protected_nodes)
        if not candidates:
            break
        removed.extend(candidates)
        pruned.remove_nodes_from(candidates)
        if not recursive:
            break
    return pruned, removed


def reconstruct_training_topology(place_name, zone_shp, manhattan_area):
    """Reproduce the topology-changing portion of utils.load_graph."""
    import geopandas as gpd
    import osmnx as ox

    if manhattan_area != "south_manhattan":
        raise ValueError("This standalone audit currently supports south_manhattan only")

    graph = ox.graph.graph_from_place(place_name, network_type="drive")
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    if not nx.is_strongly_connected(graph):
        largest = max(nx.strongly_connected_components(graph), key=len)
        graph = graph.subgraph(largest).copy()
    else:
        graph = graph.copy()
    graph.remove_edges_from(nx.selfloop_edges(graph))

    nodes_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=False)
    zones_gdf = gpd.read_file(zone_shp).to_crs("EPSG:4326")
    nodes_in_zones = gpd.sjoin(nodes_gdf, zones_gdf, how="inner", predicate="within")
    zone_to_nodes = {
        location_id: list(group.index)
        for location_id, group in nodes_in_zones.groupby("LocationID")
    }
    filtered_zones = zones_gdf[zones_gdf["zone"].isin(SOUTH_MANHATTAN_ZONES)]
    selected_nodes = [
        node
        for location_id in filtered_zones["LocationID"].tolist()
        for node in zone_to_nodes.get(location_id, [])
    ]
    graph = graph.subgraph(selected_nodes).copy()

    # Preserve the existing preprocessing pass exactly. MultiGraph degrees count
    # parallel edges; the audit below intentionally uses unique neighbors instead.
    old_trivial_nodes = [
        node
        for node in graph.nodes
        if (
            graph.in_degree(node) == 1
            and graph.out_degree(node) == 1
            and list(graph.predecessors(node))[0] == list(graph.successors(node))[0]
        )
        or (graph.in_degree(node) == 1 and graph.out_degree(node) == 0)
        or (graph.in_degree(node) == 0 and graph.out_degree(node) == 1)
    ]
    graph.remove_nodes_from(old_trivial_nodes)
    largest = max(nx.strongly_connected_components(graph), key=len)
    return graph.subgraph(largest).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--zone-shp", default="../data/processed/taxi_zones.shp")
    parser.add_argument("--place-name", default="Manhattan, New York City, New York, USA")
    parser.add_argument("--manhattan-area", default="south_manhattan")
    parser.add_argument("--recursive", action="store_true", help="Repeat pruning until no candidates remain")
    parser.add_argument("--output-csv", default="results/manhattan_single_neighbor_nodes.csv")
    parser.add_argument(
        "--pruned-graph-output",
        default=None,
        help="Optional pickle output. The original graph and source data are never modified.",
    )
    args = parser.parse_args()

    expected_nodes, start_indices = parse_training_log(args.training_log)

    graph = reconstruct_training_topology(
        args.place_name,
        args.zone_shp,
        args.manhattan_area,
    )
    nodes = list(graph.nodes)
    if len(nodes) != expected_nodes:
        raise RuntimeError(
            "Reconstructed graph does not match the training graph: "
            f"training log has {expected_nodes} nodes, reconstructed graph has {len(nodes)}. "
            "Do not prune until the same OSM graph/cache is available."
        )
    if any(index < 0 or index >= len(nodes) for index in start_indices):
        raise RuntimeError(f"Start indices are outside the reconstructed graph: {start_indices}")

    node_to_idx = {node: index for index, node in enumerate(nodes)}
    protected_starts = {nodes[index] for index in start_indices}
    candidates = single_neighbor_nonstarts(graph, protected_starts)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "node_index",
                "osm_node_id",
                "sole_neighbor_index",
                "sole_neighbor_osm_id",
                "unique_predecessors",
                "unique_successors",
                "incoming_edges",
                "outgoing_edges",
                "eligible_start",
                "eligible_pickup",
            ],
        )
        writer.writeheader()
        for node in sorted(candidates, key=node_to_idx.get):
            neighbor = next(iter(unique_neighbors(graph, node)))
            writer.writerow({
                "node_index": node_to_idx[node],
                "osm_node_id": node,
                "sole_neighbor_index": node_to_idx[neighbor],
                "sole_neighbor_osm_id": neighbor,
                "unique_predecessors": len(set(graph.predecessors(node))),
                "unique_successors": len(set(graph.successors(node))),
                "incoming_edges": graph.in_degree(node),
                "outgoing_edges": graph.out_degree(node),
                "eligible_start": node in protected_starts,
                # The current experiment explicitly sets pickups to all graph nodes.
                "eligible_pickup": True,
            })

    pruned, removed = prune_candidates(graph, protected_starts, recursive=args.recursive)
    print(f"Training graph nodes: {len(graph):,}")
    print(f"Protected start indices: {start_indices}")
    print(f"Protected start OSM IDs: {sorted(protected_starts)}")
    print(f"One-pass candidates: {len(candidates):,}")
    print(f"Nodes removed ({'recursive' if args.recursive else 'one pass'}): {len(removed):,}")
    print(f"Remaining nodes: {len(pruned):,}")
    print(f"Remaining graph strongly connected: {nx.is_strongly_connected(pruned)}")
    print("Eligible pickups removed: "
          f"{len(removed):,} (the current experiment allows every graph node as a pickup)")
    print(f"Wrote candidate report: {output_csv}")

    if args.pruned_graph_output:
        output_graph = Path(args.pruned_graph_output)
        output_graph.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "G": pruned,
            "removed_osm_node_ids": removed,
            "protected_start_osm_node_ids": sorted(protected_starts),
            "source_node_to_idx": node_to_idx,
            "recursive": args.recursive,
        }
        with output_graph.open("wb") as handle:
            pickle.dump(payload, handle)
        print(f"Wrote pruned graph artifact: {output_graph}")


if __name__ == "__main__":
    main()
