#!/usr/bin/env python3
"""
Command-line interface for the Manhattan Ride-Sharing Simulator.

Usage examples:
    # Run multi-agent simulation with optimal transport matching
    python run_simulator.py --multi_agent --num_agents 5 --num_pickups 5
    
    # Run with random trips (simple mode)
    python run_simulator.py --num_trips 3 --seed 42
    
    # Run with specific zones
    python run_simulator.py --start_zones "Upper East Side North" --pickup_zones "Lenox Hill West" --num_trips 2
    
    # Save as HTML with custom filename
    python run_simulator.py --output simulation.html
    
    # Use 3 fixed start nodes selected by betweenness centrality (same as training)
    python run_simulator.py --multi_agent --num_agents 3 --sample_starts_from_three_fixed --three_fixed_selection_method betweenness
    
    # Customize animation speed
    python run_simulator.py --speed 20 --fps 60
"""

import argparse
import numpy as np
import networkx as nx
from simulator import ManhattanSimulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manhattan Ride-Sharing Simulator with Plotly Animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run multi-agent simulation with optimal transport matching
  python run_simulator.py --multi_agent --num_agents 5 --num_pickups 5

  # Generate 3 random trips and show animation (simple mode)
  python run_simulator.py --num_trips 3

  # Use specific zones for start and pickup
  python run_simulator.py --start_zones "Upper East Side North" "Yorkville West" \\
                          --pickup_zones "Lenox Hill West" --num_trips 2

  # Save animation to file
  python run_simulator.py --output my_simulation.html
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--multi_agent",
        action="store_true",
        help="Run multi-agent simulation with optimal transport matching"
    )
    
    # Environment settings
    parser.add_argument(
        "--place_name", 
        type=str, 
        default="Manhattan, New York City, New York, USA",
        help="OSMnx place name for the road network"
    )
    parser.add_argument(
        "--zone_shp", 
        type=str, 
        default="../data/processed/taxi_zones.shp",
        help="Path to the taxi zones shapefile"
    )
    parser.add_argument(
        "--no_congestion", 
        action="store_true",
        help="Disable congestion multipliers on roads"
    )
    parser.add_argument(
        "--cycle_length",
        type=float,
        default=90.0,
        help="Traffic light cycle length in seconds"
    )
    parser.add_argument(
        "--random_offsets",
        action="store_true",
        help="Use random offsets for traffic lights"
    )
    
    # Multi-agent settings
    parser.add_argument(
        "--num_agents",
        type=int,
        default=5,
        help="Number of agents (multi-agent mode)"
    )
    parser.add_argument(
        "--num_pickups",
        type=int,
        default=None,
        help="Number of pickups to sample (defaults to num_agents)"
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=3600.0, # 1 hour
        help="Maximum simulation time in seconds (multi-agent mode)"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="Simulation time step in seconds (multi-agent mode)"
    )
    parser.add_argument(
        "--q_table",
        type=str,
        default=None,
        help="Path to Q-table file for greedy policy (default: use shortest path)"
    )
    
    # Fixed node selection (matching main.py training pipeline)
    parser.add_argument(
        "--sample_starts_from_three_fixed",
        action="store_true",
        help="Sample starting nodes from 3 fixed nodes (matching main.py training). Use same seed and selection method as training."
    )
    parser.add_argument(
        "--sample_pickups_from_three_fixed",
        action="store_true",
        help="Sample pickup nodes from 3 fixed nodes (matching main.py training). Use same seed and selection method as training."
    )
    parser.add_argument(
        "--three_fixed_selection_method",
        type=str,
        default="random",
        choices=["random", "degree", "closeness", "betweenness"],
        help="Method to select 3 fixed nodes: 'random', 'degree', 'closeness', 'betweenness' (must match training)"
    )
    parser.add_argument(
        "--round_trip",
        action="store_true",
        help="Agents travel back to start after reaching pickup"
    )
    parser.add_argument(
        "--reassignment_interval",
        type=float,
        default=120.0,
        help="Seconds between reassignment batches. Free agents accumulate and are reassigned together every DT seconds (default: 120)"
    )
    
    # Trip settings (simple mode)
    parser.add_argument(
        "--num_trips", 
        type=int, 
        default=3,
        help="Number of trips to simulate (simple mode)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--start_zones", 
        nargs='+', 
        type=str, 
        default=None,
        help="Zone names for start locations"
    )
    parser.add_argument(
        "--pickup_zones", 
        nargs='+', 
        type=str, 
        default=None,
        help="Zone names for pickup (destination) locations"
    )
    
    # Animation settings
    parser.add_argument(
        "--speed", 
        type=float, 
        default=10.0,
        help="Speed factor for animation (higher = faster)"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=30,
        help="Frames per second for animation"
    )
    parser.add_argument(
        "--no_paths", 
        action="store_true",
        help="Hide the route paths on the map"
    )
    parser.add_argument(
        "--title", 
        type=str, 
        default="Manhattan Ride-Sharing Simulation (Shortest Path Policy)",
        help="Title for the animation"
    )
    
    # Output settings
    parser.add_argument(
        "--output", 
        type=str, 
        default="manhattan_simulation.html",
        help="Output HTML file path"
    )
    
    # List available zones
    parser.add_argument(
        "--list_zones", 
        action="store_true",
        help="List available zone names and exit"
    )
    
    return parser.parse_args()


def select_three_fixed_nodes(G, node_to_idx, seed, selection_method, node_to_zone=None, target="starts"):
    """
    Select 3 fixed nodes using the specified selection method.
    This function matches the logic in main.py exactly to ensure the same nodes are selected.
    
    Args:
        G: NetworkX graph
        node_to_idx: Dict mapping node IDs to indices
        seed: Random seed for reproducibility
        selection_method: One of 'random', 'degree', 'closeness', 'betweenness'
        node_to_zone: Optional dict mapping node IDs to zone IDs
        target: Either 'starts' or 'pickups' (for logging)
    
    Returns:
        List of 3 node IDs (not indices)
    """
    all_nodes_list = list(node_to_idx.keys())
    if len(all_nodes_list) < 3:
        raise ValueError(f"Not enough nodes in graph ({len(all_nodes_list)}). Need at least 3 nodes.")
    
    rng = np.random.default_rng(seed)
    
    # Helper function to select nodes ensuring different zones
    def select_nodes_from_different_zones(candidate_nodes_with_scores, node_to_zone_mapping):
        """Select 3 nodes ensuring they come from different zones."""
        selected_nodes = []
        selected_zones = set()
        
        if node_to_zone_mapping is None or len(node_to_zone_mapping) == 0:
            print("  Warning: No zone mapping available. Selecting top 3 nodes without zone constraint.")
            for node, score in candidate_nodes_with_scores:
                selected_nodes.append(node)
                if len(selected_nodes) == 3:
                    break
            return selected_nodes, set()
        
        print(f"  Checking zones for top candidates...")
        zone_counts = {}
        for node, score in candidate_nodes_with_scores[:20]:
            zone = node_to_zone_mapping.get(node)
            if zone is not None:
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        print(f"  Zones in top 20 candidates: {len(zone_counts)} unique zones")
        if len(zone_counts) > 0:
            print(f"  Top zones: {sorted(zone_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        for node, score in candidate_nodes_with_scores:
            node_zone = node_to_zone_mapping.get(node)
            if node_zone is not None and node_zone not in selected_zones:
                selected_nodes.append(node)
                selected_zones.add(node_zone)
                print(f"  Selected node {node} from zone {node_zone} (score: {score:.6f})")
                if len(selected_nodes) == 3:
                    break
            elif node_zone is None:
                continue
        
        if len(selected_nodes) < 3:
            print(f"  Warning: Only found {len(selected_nodes)} nodes from different zones.")
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
        if node_to_zone is not None:
            zone_to_nodes = {}
            for node in all_nodes_list:
                zone = node_to_zone.get(node)
                if zone is not None:
                    if zone not in zone_to_nodes:
                        zone_to_nodes[zone] = []
                    zone_to_nodes[zone].append(node)
            
            available_zones = list(zone_to_nodes.keys())
            if len(available_zones) < 3:
                print(f"  Warning: Only {len(available_zones)} zones available. Some nodes may be from the same zone.")
            
            permuted_zone_indices = rng.permutation(len(available_zones))
            selected_zones_list = [available_zones[int(idx)] for idx in permuted_zone_indices[:3]]
            
            selected_nodes = []
            for zone in selected_zones_list:
                zone_nodes = zone_to_nodes[zone]
                node_idx = rng.integers(0, len(zone_nodes))
                selected_nodes.append(zone_nodes[int(node_idx)])
            
            print(f"Selected 3 nodes using random selection (one per zone): {selected_nodes}")
            print(f"  Zones: {[node_to_zone.get(n, 'unknown') for n in selected_nodes]}")
        else:
            permuted_indices = rng.permutation(len(all_nodes_list))
            selected_indices = permuted_indices[:3]
            selected_nodes = [all_nodes_list[int(idx)] for idx in selected_indices]
            print(f"Selected 3 nodes using random selection: {selected_nodes}")
        
    elif selection_method == "degree":
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
    
    return selected_nodes


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Manhattan Ride-Sharing Simulator")
    print("=" * 60)
    print()
    
    # Initialize simulator
    print("Initializing simulator...")
    sim = ManhattanSimulator(
        place_name=args.place_name,
        zone_shp=args.zone_shp,
        no_congestion=args.no_congestion,
        cycle_length=args.cycle_length,
        random_offsets=args.random_offsets,
        seed=args.seed if args.seed else 42
    )
    
    # List zones if requested
    if args.list_zones:
        print("\nAvailable zones:")
        print("-" * 40)
        for zone_name in sorted(sim.zone_name_to_locationID.keys()):
            loc_id = sim.zone_name_to_locationID[zone_name]
            node_count = len(sim.zone_to_nodes.get(loc_id, []))
            print(f"  {zone_name} (LocationID: {loc_id}, Nodes: {node_count})")
        return
    
    if args.multi_agent:
        # Multi-agent simulation with optimal transport matching
        run_multi_agent_simulation(sim, args)
    else:
        # Simple mode with predefined trips
        run_simple_simulation(sim, args)


def run_multi_agent_simulation(sim, args):
    """Run multi-agent simulation with optimal transport matching."""
    # Default num_pickups to num_agents if not specified
    num_pickups = args.num_pickups if args.num_pickups is not None else args.num_agents
    
    # Validate flags
    if args.sample_starts_from_three_fixed and args.sample_pickups_from_three_fixed:
        raise ValueError("Cannot use both --sample_starts_from_three_fixed and --sample_pickups_from_three_fixed at the same time.")
    
    # Select fixed nodes if requested (must use same seed and method as training!)
    fixed_start_nodes = None
    fixed_pickup_nodes = None
    seed = args.seed if args.seed is not None else 42
    
    if args.sample_starts_from_three_fixed:
        print(f"\nSelecting 3 fixed START nodes (method: {args.three_fixed_selection_method}, seed: {seed})...")
        fixed_start_nodes = select_three_fixed_nodes(
            G=sim.G,
            node_to_idx=sim.node_to_idx,
            seed=seed,
            selection_method=args.three_fixed_selection_method,
            node_to_zone=sim.node_to_zone,
            target="starts"
        )
        print(f"Fixed start nodes: {fixed_start_nodes}")
    
    if args.sample_pickups_from_three_fixed:
        print(f"\nSelecting 3 fixed PICKUP nodes (method: {args.three_fixed_selection_method}, seed: {seed})...")
        fixed_pickup_nodes = select_three_fixed_nodes(
            G=sim.G,
            node_to_idx=sim.node_to_idx,
            seed=seed,
            selection_method=args.three_fixed_selection_method,
            node_to_zone=sim.node_to_zone,
            target="pickups"
        )
        print(f"Fixed pickup nodes: {fixed_pickup_nodes}")
    
    print(f"\n--- Multi-Agent Simulation Mode ---")
    print(f"  Agents: {args.num_agents}")
    print(f"  Pickups: {num_pickups}")
    if fixed_start_nodes:
        print(f"  Start nodes: 3 fixed ({args.three_fixed_selection_method})")
    if fixed_pickup_nodes:
        print(f"  Pickup nodes: 3 fixed ({args.three_fixed_selection_method})")
    print(f"  Max time: {args.max_time}s")
    print(f"  Time step: {args.dt}s")
    print(f"  Traffic cycle: {args.cycle_length}s")
    if args.q_table:
        print(f"  Q-table: {args.q_table}")
    else:
        print(f"  Policy: Shortest Path")
    if args.round_trip:
        print(f"  Round trip: ENABLED (agents return to start)")
    print(f"  Reassignment interval: {args.reassignment_interval}s")
    
    # Run simulation
    agents, assignments = sim.run_simulation(
        num_agents=args.num_agents,
        num_pickups=num_pickups,
        max_time=args.max_time,
        dt=args.dt,
        seed=args.seed,
        q_table_path=args.q_table,
        fixed_start_nodes=fixed_start_nodes,
        fixed_pickup_nodes=fixed_pickup_nodes,
        round_trip=args.round_trip,
        reassignment_interval=args.reassignment_interval
    )
    
    # Print statistics
    print(f"\n--- Simulation Results ---")
    total_travel_time = 0
    completed_agents = 0
    for agent_id, start, pickup in assignments:
        agent = agents[agent_id]
        
        # Skip agents that started at their destination
        if start == pickup:
            print(f"  Agent {agent_id}: Already at destination (0.0s)")
            continue
        
        # Calculate actual travel time (time spent on_ride)
        start_time = None
        end_time = None
        for t, _, _, state in agent.history:
            if state == "on_ride" and start_time is None:
                start_time = t
            if state == "free" and start_time is not None and end_time is None:
                end_time = t
                break
        
        if start_time is None:
            print(f"  Agent {agent_id}: Did not start ride")
            continue
        
        if end_time is None:
            # Still on ride at end of simulation
            end_time = agent.history[-1][0]
            status = " (incomplete)"
        else:
            status = ""
            completed_agents += 1
        
        trip_time = end_time - start_time
        total_travel_time += trip_time
        
        # Calculate base travel time (without traffic lights)
        path = sim.get_shortest_path(start, pickup)
        base_time = sum(sim.get_path_travel_times(path))
        wait_time = trip_time - base_time
        
        print(f"  Agent {agent_id}: {trip_time:.1f}s total (travel: {base_time:.1f}s, wait: {max(0, wait_time):.1f}s){status}")
    
    print("-" * 60)
    print(f"  Total travel time (all agents): {total_travel_time:.1f}s ({total_travel_time/60:.2f} min)")
    if completed_agents > 0:
        print(f"  Average travel time per completed agent: {total_travel_time / completed_agents:.1f}s")
        print(f"  Completed: {completed_agents}/{len(assignments)} agents")
    
    # Create animation
    print("\nCreating animation...")
    policy_name = "Q-table Greedy" if sim.use_q_policy else "Shortest Path"
    fig = sim.create_multi_agent_animation(
        agents=agents,
        assignments=assignments,
        fps=args.fps,
        show_paths=not args.no_paths,
        title=f"Manhattan Simulation ({args.num_agents} agents, {policy_name})"
    )
    
    # Save to file
    sim.save_html(fig, args.output)
    print(f"\nAnimation saved to: {args.output}")
    
    print("\nDone!")


def run_simple_simulation(sim, args):
    """Run simple simulation with predefined trips."""
    # Generate trips
    print(f"\nGenerating {args.num_trips} trips...")
    if args.start_zones and args.pickup_zones:
        trips = sim.get_trips_by_zone(
            start_zone_names=args.start_zones,
            pickup_zone_names=args.pickup_zones,
            num_trips=args.num_trips,
            seed=args.seed
        )
        print(f"  Start zones: {args.start_zones}")
        print(f"  Pickup zones: {args.pickup_zones}")
    else:
        trips = sim.get_random_trips(num_trips=args.num_trips, seed=args.seed)
        print("  Using random start/pickup locations")
    
    # Display trip info
    print("\nTrip details:")
    print("-" * 60)
    total_travel_time = 0
    for i, (start, pickup) in enumerate(trips):
        path = sim.get_shortest_path(start, pickup)
        times = sim.get_path_travel_times(path)
        trip_time = sum(times)
        total_travel_time += trip_time
        
        # Get coordinates for display
        start_coords = sim.node_coords[start]
        pickup_coords = sim.node_coords[pickup]
        
        print(f"  Agent {i+1}:")
        print(f"    Start:  Node {start} ({start_coords[1]:.5f}, {start_coords[0]:.5f})")
        print(f"    Pickup: Node {pickup} ({pickup_coords[1]:.5f}, {pickup_coords[0]:.5f})")
        print(f"    Path:   {len(path)} nodes, {trip_time:.1f}s ({trip_time/60:.2f} min)")
    
    print("-" * 60)
    print(f"  Total travel time: {total_travel_time:.1f}s ({total_travel_time/60:.2f} min)")
    
    # Create animation
    print("\nCreating animation...")
    fig = sim.create_animation(
        trips=trips,
        fps=args.fps,
        speed_factor=args.speed,
        show_paths=not args.no_paths,
        title=args.title
    )
    
    # Save to file
    sim.save_html(fig, args.output)
    print(f"\nAnimation saved to: {args.output}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
