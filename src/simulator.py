"""
Manhattan Ride-Sharing Simulator with Plotly Animation

This module provides visualization of agents moving around Manhattan
following either the shortest path policy or a learned Q-table policy.
It reuses the same traffic parameters and preprocessing from the main 
training environment.

Features:
- M agents with state tracking (free/on_a_ride)
- Optimal transport matching using Q-table values (if loaded) or shortest path costs
- Traffic light waiting times
- Discrete event simulation where agents decide at node arrivals
- Support for Q-table greedy policy
"""

import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import linear_sum_assignment
import osmnx as ox
import pickle
from pathlib import Path

from utils import load_graph, apply_congestion_model


class AgentState(Enum):
    """Agent states in the simulation."""
    FREE = "free"
    ON_A_RIDE = "on_a_ride"


class TripDirection(Enum):
    """Trip direction for round-trip mode."""
    TO_PICKUP = "to_pickup"      # Going from start to pickup
    RETURNING = "returning"      # Going from pickup back to start


@dataclass
class Agent:
    """
    Represents an agent in the simulation.
    
    Attributes:
        agent_id: Unique identifier for the agent
        state: Current state (FREE or ON_A_RIDE)
        current_node: Current node ID (OSM node ID)
        current_node_idx: Current node index (for Q-table lookup)
        target_node: Target pickup node ID (None if FREE)
        target_node_idx: Target pickup node index
        path: Remaining path to target (for shortest path policy)
        path_index: Current position in path
        next_node_idx: Next node index (for Q-table policy)
        current_edge_start_time: Time when agent started current edge
        current_edge_travel_time: Travel time of current edge
        current_edge_wait_time: Wait time at traffic light after edge
        time_on_edge: Time spent on current edge (for interpolation)
        arrival_time: Time when agent will arrive at next node
        position: Current (lon, lat) position
        sim_time: Current simulation time for this agent
        history: List of (time, lon, lat, state) for animation
    """
    agent_id: int
    state: AgentState = AgentState.FREE
    current_node: int = None
    current_node_idx: int = None
    target_node: int = None
    target_node_idx: int = None
    path: List[int] = field(default_factory=list)
    path_index: int = 0
    next_node_idx: int = None
    current_edge_start_time: float = 0.0
    current_edge_travel_time: float = 0.0
    current_edge_wait_time: float = 0.0
    time_on_edge: float = 0.0
    arrival_time: float = float('inf')
    position: Tuple[float, float] = (0.0, 0.0)
    sim_time: float = 0.0
    history: List[Tuple[float, float, float, str]] = field(default_factory=list)
    # Round-trip tracking
    original_start_node: int = None      # Original starting node for round-trip
    original_pickup_node: int = None     # Original pickup/destination node
    trip_direction: str = "to_pickup"    # "to_pickup" or "returning"
    num_trips_completed: int = 0         # Number of round trips completed
    
    def is_free(self) -> bool:
        return self.state == AgentState.FREE
    
    def is_on_ride(self) -> bool:
        return self.state == AgentState.ON_A_RIDE


class ManhattanSimulator:
    """
    Simulator for visualizing agent movement on Manhattan road network.
    
    Uses Plotly for interactive animations showing agents moving from
    starting locations to pickup destinations following shortest paths.
    
    Features:
    - Multi-agent simulation with state tracking
    - Optimal transport matching for agent-pickup assignment (Q-table or shortest path costs)
    - Traffic light waiting times at intersections
    - Event-driven simulation with edge-level decisions
    """
    
    def __init__(
        self,
        place_name: str = "Manhattan, New York City, New York, USA",
        zone_shp: str = "../data/processed/taxi_zones.shp",
        zone_names: Optional[List[str]] = None,
        no_congestion: bool = False,
        cycle_length: float = 90.0,
        offset: float = 0.0,
        random_offsets: bool = False,
        seed: int = 42,
    ):
        """
        Initialize the simulator by loading the Manhattan graph.
        
        Args:
            place_name: OSMnx place name for downloading the road network
            zone_shp: Path to the taxi zones shapefile
            zone_names: List of zone names to include (None for default zones)
            no_congestion: If True, disable congestion multipliers
            cycle_length: Traffic light cycle length in seconds
            offset: Global offset for traffic lights
            random_offsets: If True, use random offsets per traffic light
            seed: Random seed for reproducibility
        """
        self.place_name = place_name
        self.zone_shp = zone_shp
        self.no_congestion = no_congestion
        self.cycle_length = cycle_length
        self.seed = seed
        
        # Load graph with preprocessing (reuses utils.py functions)
        print("Loading Manhattan road network...")
        self.G, self.nodes_gdf, self.node_to_zone, self.zone_to_nodes, self.zone_name_to_locationID = load_graph(
            place_name=place_name,
            zone_shp=zone_shp,
            no_congestion=no_congestion
        )
        
        # Build node index mappings
        self.all_nodes = list(self.G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.all_nodes)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}
        self.num_nodes = len(self.all_nodes)
        
        # Extract node coordinates
        self.node_coords = self._extract_coordinates()
        
        # Precompute shortest path distances
        print("Computing shortest path distances...")
        self.distances = self._compute_distances()
        
        # Build traffic light parameters (reuses same logic as taxi_env_utils.py)
        print("Building traffic light parameters...")
        self.traffic_params = self._build_traffic_params(cycle_length, offset, random_offsets, seed)
        
        # Store edge geometries for visualization
        self.edge_traces = self._build_edge_traces()
        
        # Agent management
        self.agents: List[Agent] = []
        
        # Q-table for learned policy (None = use shortest path)
        self.q_table: Optional[np.ndarray] = None
        self.q_dt: float = 1.0  # Time discretization for Q-table
        self.q_max_time_slices: int = 90
        self.adj_list: Optional[np.ndarray] = None  # Adjacency list for Q-table actions
        self.use_q_policy: bool = False
        
        # Build adjacency list for action mapping
        self._build_adjacency_list()
        
        print(f"Loaded graph with {len(self.all_nodes)} nodes and {self.G.number_of_edges()} edges")
        
    def _extract_coordinates(self) -> Dict[int, Tuple[float, float]]:
        """Extract lat/lon coordinates for all nodes."""
        coords = {}
        for node in self.all_nodes:
            coords[node] = (
                self.G.nodes[node]['x'],  # longitude
                self.G.nodes[node]['y']   # latitude
            )
        return coords
    
    def _build_adjacency_list(self):
        """Build adjacency list mapping node indices to neighbor indices."""
        # Find max degree
        max_deg = max(dict(self.G.out_degree()).values())
        self.max_deg = max_deg
        
        # Build adjacency list: adj_list[node_idx] = [neighbor_idx, ...]
        self.adj_list = -np.ones((self.num_nodes, max_deg), dtype=np.int32)
        self.travel_times_matrix = np.zeros((self.num_nodes, max_deg), dtype=np.float32)
        self.neighbor_mask = np.zeros((self.num_nodes, max_deg), dtype=bool)
        
        for i, node in enumerate(self.all_nodes):
            neighbors = list(self.G.successors(node))
            for j, nbr in enumerate(neighbors[:max_deg]):
                nbr_idx = self.node_to_idx[nbr]
                self.adj_list[i, j] = nbr_idx
                
                # Get travel time for this edge
                edge_data = self.G.get_edge_data(node, nbr)
                if edge_data:
                    if isinstance(list(edge_data.values())[0], dict):
                        best_key = min(edge_data, key=lambda k: edge_data[k].get('travel_time_congested', float('inf')))
                        time_min = edge_data[best_key].get('travel_time_congested', 0)
                    else:
                        time_min = edge_data.get('travel_time_congested', 0)
                    self.travel_times_matrix[i, j] = time_min * 60.0  # Convert to seconds
                
                self.neighbor_mask[i, j] = True
            
            # Pad with first neighbor if fewer neighbors than max_deg
            n_neighbors = len(neighbors)
            if 0 < n_neighbors < max_deg:
                self.adj_list[i, n_neighbors:] = self.adj_list[i, 0]
                self.travel_times_matrix[i, n_neighbors:] = self.travel_times_matrix[i, 0]
    
    def load_q_table(self, q_table_path: str) -> bool:
        """
        Load a Q-table from a pickle file for greedy policy.
        
        Args:
            q_table_path: Path to the Q-table pickle file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        path = Path(q_table_path)
        if not path.exists():
            print(f"Q-table file not found: {q_table_path}")
            return False
        
        try:
            with open(path, 'rb') as f:
                state_dict = pickle.load(f)
            
            self.q_table = np.array(state_dict['q_table'])
            self.q_dt = state_dict.get('dt', 1.0)
            self.q_max_time_slices = state_dict.get('max_time_slices', 90)
            self.use_q_policy = True
            
            print(f"Loaded Q-table from {q_table_path}")
            print(f"  Shape: {self.q_table.shape}")
            print(f"  dt: {self.q_dt}")
            print(f"  Max time slices: {self.q_max_time_slices}")
            
            # Verify Q-table dimensions match our graph
            if self.q_table.shape[0] != self.num_nodes:
                print(f"  WARNING: Q-table num_nodes ({self.q_table.shape[0]}) != graph nodes ({self.num_nodes})")
                print(f"  Q-table policy may not work correctly!")
            
            return True
            
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            return False
    
    def get_greedy_action(self, current_node_idx: int, pickup_node_idx: int, time: float) -> int:
        """
        Get the greedy action from the Q-table.
        
        Args:
            current_node_idx: Current node index
            pickup_node_idx: Pickup/target node index
            time: Current time in seconds
            
        Returns:
            Action index (which neighbor to move to)
        """
        if self.q_table is None:
            raise ValueError("Q-table not loaded. Call load_q_table() first.")
        
        # Discretize time to get time slice index
        time_idx = int(time / self.q_dt) % self.q_max_time_slices
        
        # Get Q-values for all actions at this state
        q_values = self.q_table[current_node_idx, pickup_node_idx, time_idx, :]
        
        # Mask invalid actions (neighbors that don't exist)
        valid_mask = self.neighbor_mask[current_node_idx]
        masked_q = np.where(valid_mask, q_values, -np.inf)
        
        # Return greedy action (argmax)
        return int(np.argmax(masked_q))
    
    def get_next_node_greedy(self, current_node_idx: int, pickup_node_idx: int, time: float) -> int:
        """
        Get the next node using greedy Q-table policy.
        
        Args:
            current_node_idx: Current node index
            pickup_node_idx: Target pickup node index
            time: Current simulation time
            
        Returns:
            Next node index
        """
        action = self.get_greedy_action(current_node_idx, pickup_node_idx, time)
        next_node_idx = self.adj_list[current_node_idx, action]
        return int(next_node_idx)
    
    def _compute_distances(self) -> np.ndarray:
        """Compute pairwise shortest path distances using travel_time_congested."""
        N = len(self.all_nodes)
        dist_mat = np.full((N, N), np.inf, dtype=np.float32)
        
        for u in self.all_nodes:
            ui = self.node_to_idx[u]
            lengths = nx.single_source_dijkstra_path_length(
                self.G, u, weight="travel_time_congested"
            )
            for v, d in lengths.items():
                vi = self.node_to_idx[v]
                dist_mat[ui, vi] = d * 60.0  # Convert minutes to seconds
        
        return dist_mat
    
    def _build_traffic_params(
        self, 
        cycle_length: float, 
        offset: float, 
        random_offsets: bool,
        seed: int
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Build traffic light parameters for each edge.
        
        Uses the same logic as taxi_env_utils.py:
        - Motorway/Trunk: always green (no light)
        - Primary: 50% green
        - Secondary: 30% green
        - Tertiary: always green
        - Residential/Living street: 30% green
        
        Returns:
            Dict mapping (u, v) edge to traffic params dict with keys:
            'period', 'green_duration', 'offset'
        """
        traffic_params = {}
        rng = np.random.RandomState(seed) if random_offsets else None
        
        for u, v, data in self.G.edges(data=True):
            hw = data.get("highway", "unclassified")
            if isinstance(hw, list):
                hw = hw[0] if hw else "unclassified"
            
            # Determine cycle and green duration based on road type
            if hw in ("motorway", "trunk"):
                cycle, green = cycle_length, cycle_length  # Always green
            elif hw == "primary":
                cycle, green = cycle_length, cycle_length * 0.5
            elif hw == "secondary":
                cycle, green = cycle_length, cycle_length * 0.3
            elif hw == "tertiary":
                cycle, green = cycle_length, cycle_length  # Always green
            elif hw in ("residential", "living_street"):
                cycle, green = cycle_length, cycle_length * 0.3
            else:
                cycle, green = cycle_length, cycle_length  # Default: always green
            
            edge_offset = float(rng.randint(0, int(cycle_length))) if random_offsets else offset
            
            traffic_params[(u, v)] = {
                'period': cycle,
                'green_duration': green,
                'offset': edge_offset
            }
        
        return traffic_params
    
    def compute_wait_time(self, u: int, v: int, arrival_time: float) -> float:
        """
        Compute the wait time at the traffic light when arriving at edge end.
        
        Args:
            u: Source node of the edge
            v: Destination node of the edge  
            arrival_time: Time when agent arrives at the end of edge (v)
            
        Returns:
            Wait time in seconds (0 if green light)
        """
        params = self.traffic_params.get((u, v))
        if params is None:
            return 0.0
        
        period = params['period']
        green = params['green_duration']
        offset = params['offset']
        
        # Compute position in the traffic cycle
        cycle_pos = (arrival_time + offset) % period
        
        # If within green phase, no wait; otherwise wait until green
        if cycle_pos < green:
            return 0.0
        else:
            return period - cycle_pos
    
    def _calculate_map_bounds(self) -> Tuple[float, float, float, float, float]:
        """
        Calculate map center and appropriate zoom level for all nodes.
        
        Returns:
            Tuple of (center_lon, center_lat, zoom, min_lon, max_lon)
        """
        all_lons = [self.node_coords[n][0] for n in self.all_nodes]
        all_lats = [self.node_coords[n][1] for n in self.all_nodes]
        
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)
        
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        
        # Calculate zoom level based on bounding box
        # Approximate degrees per pixel at different zoom levels
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        
        # Use the larger range to determine zoom
        max_range = max(lon_range, lat_range)
        
        # Approximate zoom calculation (empirical formula)
        # At zoom 10, roughly 0.1 degrees visible
        # Each zoom level doubles the detail
        if max_range > 0:
            import math
            zoom = 10 - math.log2(max_range / 0.05)
            zoom = max(10, min(16, zoom))  # Clamp between 10 and 16
        else:
            zoom = 14
        
        return center_lon, center_lat, zoom, min_lon, max_lon
    
    def _build_edge_traces(self) -> go.Scatter:
        """Build Plotly traces for all edges in the graph."""
        edge_x = []
        edge_y = []
        
        for u, v in self.G.edges():
            x0, y0 = self.node_coords[u]
            x1, y1 = self.node_coords[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        return go.Scattermapbox(
            lon=edge_x,
            lat=edge_y,
            mode='lines',
            line=dict(width=1, color='#4a5568'),
            hoverinfo='none',
            name='Roads'
        )
    
    def get_shortest_path(self, start_node: int, target_node: int) -> List[int]:
        """
        Compute the shortest path between two nodes using congested travel times.
        
        Args:
            start_node: Starting node ID (OSM node ID)
            target_node: Target node ID (OSM node ID)
            
        Returns:
            List of node IDs forming the shortest path
        """
        try:
            path = nx.shortest_path(
                self.G, start_node, target_node, 
                weight="travel_time_congested"
            )
            return path
        except nx.NetworkXNoPath:
            print(f"No path found between {start_node} and {target_node}")
            return []
    
    def get_path_travel_times(self, path: List[int]) -> List[float]:
        """
        Get the travel time for each edge in the path.
        
        Args:
            path: List of node IDs forming the path
            
        Returns:
            List of travel times (in seconds) for each edge
        """
        times = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Get the edge with minimum travel time (handles MultiDiGraph)
            edge_data = self.G.get_edge_data(u, v)
            if edge_data:
                # MultiDiGraph returns dict of dicts, regular DiGraph returns dict
                if isinstance(list(edge_data.values())[0], dict):
                    # MultiDiGraph
                    best_key = min(edge_data, key=lambda k: edge_data[k].get('travel_time_congested', float('inf')))
                    time_min = edge_data[best_key].get('travel_time_congested', 0)
                else:
                    time_min = edge_data.get('travel_time_congested', 0)
                times.append(time_min * 60.0)  # Convert to seconds
            else:
                times.append(0)
        return times
    
    def get_edge_travel_time(self, u: int, v: int) -> float:
        """Get travel time for a single edge in seconds."""
        edge_data = self.G.get_edge_data(u, v)
        if edge_data:
            if isinstance(list(edge_data.values())[0], dict):
                best_key = min(edge_data, key=lambda k: edge_data[k].get('travel_time_congested', float('inf')))
                time_min = edge_data[best_key].get('travel_time_congested', 0)
            else:
                time_min = edge_data.get('travel_time_congested', 0)
            return time_min * 60.0
        return 0.0
    
    # =========================================================================
    # Multi-Agent Simulation Methods
    # =========================================================================
    
    def initialize_agents(self, num_agents: int, initial_positions: Optional[List[int]] = None) -> List[Agent]:
        """
        Initialize M agents at specified or random positions.
        
        Args:
            num_agents: Number of agents to create
            initial_positions: Optional list of starting node IDs
            
        Returns:
            List of initialized Agent objects
        """
        self.agents = []
        
        if initial_positions is None:
            # Random initial positions
            np.random.seed(self.seed)
            initial_positions = np.random.choice(self.all_nodes, size=num_agents, replace=True).tolist()
        
        for i in range(num_agents):
            node = initial_positions[i] if i < len(initial_positions) else self.all_nodes[0]
            node_idx = self.node_to_idx[node]
            pos = self.node_coords[node]
            agent = Agent(
                agent_id=i,
                state=AgentState.FREE,
                current_node=node,
                current_node_idx=node_idx,
                position=pos
            )
            agent.history.append((0.0, pos[0], pos[1], "free"))
            self.agents.append(agent)
        
        return self.agents
    
    def sample_pickups(self, num_pickups: int, seed: Optional[int] = None) -> List[int]:
        """
        Sample pickup locations from the graph.
        
        Args:
            num_pickups: Number of pickups to sample
            seed: Random seed
            
        Returns:
            List of pickup node IDs
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.choice(self.all_nodes, size=num_pickups, replace=True).tolist()
    
    def optimal_transport_matching(
        self, 
        agent_positions: List[int], 
        pickup_positions: List[int],
        current_time: float = 0.0
    ) -> List[Tuple[int, int]]:
        """
        Perform optimal transport matching between agents and pickups.
        
        Uses the Hungarian algorithm to minimize total cost.
        - When a Q-table is loaded: cost = -max_a Q(agent, pickup, time, a)
          (negated because Hungarian minimizes, and higher Q = better assignment)
        - Otherwise: cost = shortest path distance
        
        Args:
            agent_positions: List of agent current node IDs
            pickup_positions: List of pickup node IDs
            current_time: Current simulation time (used for Q-table time discretization)
            
        Returns:
            List of (agent_index, pickup_index) assignments
        """
        num_agents = len(agent_positions)
        num_pickups = len(pickup_positions)
        
        cost_matrix = np.zeros((num_agents, num_pickups))
        
        if self.use_q_policy and self.q_table is not None:
            # Build cost matrix using Q-table values: cost = -V(s) = -max_a Q(s, a)
            time_idx = int(current_time / self.q_dt) % self.q_max_time_slices
            
            for i, agent_node in enumerate(agent_positions):
                agent_idx = self.node_to_idx[agent_node]
                valid_mask = self.neighbor_mask[agent_idx]
                
                for j, pickup_node in enumerate(pickup_positions):
                    pickup_idx = self.node_to_idx[pickup_node]
                    
                    if agent_idx == pickup_idx:
                        # Already at destination — zero cost
                        cost_matrix[i, j] = 0.0
                        continue
                    
                    # Q-values for all actions at state (agent, pickup, time)
                    q_values = self.q_table[agent_idx, pickup_idx, time_idx, :]
                    masked_q = np.where(valid_mask, q_values, -np.inf)
                    
                    # Cost = -V(s) = -max_a Q(s, a)
                    # Higher Q-value → lower cost → preferred assignment
                    cost_matrix[i, j] = -np.max(masked_q)
        else:
            # Build cost matrix using shortest path distances
            for i, agent_node in enumerate(agent_positions):
                agent_idx = self.node_to_idx[agent_node]
                for j, pickup_node in enumerate(pickup_positions):
                    pickup_idx = self.node_to_idx[pickup_node]
                    cost_matrix[i, j] = self.distances[agent_idx, pickup_idx]
        
        # Solve assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Return assignments as list of tuples
        assignments = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
        return assignments
    
    def assign_pickups_to_agents(
        self, 
        pickups: List[int],
        current_time: float = 0.0
    ) -> List[Tuple[int, int, int]]:
        """
        Assign pickups to free agents using optimal transport.
        
        Args:
            pickups: List of pickup node IDs
            current_time: Current simulation time
            
        Returns:
            List of (agent_id, start_node, pickup_node) assignments
        """
        # Get free agents
        free_agents = [a for a in self.agents if a.is_free()]
        
        if not free_agents or not pickups:
            return []
        
        # Match number of pickups to free agents
        num_to_assign = min(len(free_agents), len(pickups))
        free_agents = free_agents[:num_to_assign]
        pickups_to_assign = pickups[:num_to_assign]
        
        # Get agent positions
        agent_positions = [a.current_node for a in free_agents]
        
        # Perform optimal transport matching (uses Q-table costs if loaded, else shortest path)
        assignments = self.optimal_transport_matching(agent_positions, pickups_to_assign, current_time)
        
        result = []
        for agent_local_idx, pickup_idx in assignments:
            agent = free_agents[agent_local_idx]
            pickup_node = pickups_to_assign[pickup_idx]
            start_node = agent.current_node
            
            # Check if already at destination
            if start_node == pickup_node:
                continue
            
            # Set up agent state
            agent.state = AgentState.ON_A_RIDE
            agent.target_node = pickup_node
            agent.target_node_idx = self.node_to_idx[pickup_node]
            agent.current_node_idx = self.node_to_idx[start_node]
            agent.sim_time = current_time
            
            # Store original start/pickup for round-trip mode
            agent.original_start_node = start_node
            agent.original_pickup_node = pickup_node
            agent.trip_direction = "to_pickup"
            
            if self.use_q_policy:
                # Q-table policy: determine next node from Q-table
                next_node_idx = self.get_next_node_greedy(
                    agent.current_node_idx, 
                    agent.target_node_idx, 
                    current_time
                )
                agent.next_node_idx = next_node_idx
                agent.path = []  # Not using path for Q-policy
                agent.path_index = 0
                
                # Get travel time to next node
                next_node = self.idx_to_node[next_node_idx]
                travel_time = self.get_edge_travel_time(start_node, next_node)
            else:
                # Shortest path policy: compute full path
                path = self.get_shortest_path(start_node, pickup_node)
                if not path or len(path) < 2:
                    agent.state = AgentState.FREE
                    agent.target_node = None
                    continue
                
                agent.path = path
                agent.path_index = 0
                agent.next_node_idx = self.node_to_idx[path[1]]
                
                # Get travel time for first edge
                travel_time = self.get_edge_travel_time(path[0], path[1])
            
            agent.current_edge_start_time = current_time
            agent.current_edge_travel_time = travel_time
            agent.current_edge_wait_time = 0.0
            agent.arrival_time = current_time + travel_time
            
            result.append((agent.agent_id, start_node, pickup_node))
        
        return result
    
    def step_simulation(self, current_time: float, dt: float) -> float:
        """
        Advance simulation by processing events up to current_time + dt.
        
        This is an event-driven simulation where agents make decisions
        only when they complete an edge (travel + traffic light wait).
        
        Args:
            current_time: Current simulation time
            dt: Time step for animation sampling
            
        Returns:
            New current time after step
        """
        new_time = current_time + dt
        
        for agent in self.agents:
            if agent.state == AgentState.FREE:
                # Free agents don't move - record position
                agent.history.append((new_time, agent.position[0], agent.position[1], "free"))
                continue
            
            # Check if agent should still be moving
            if self.use_q_policy:
                # Q-policy: check if we have a valid next node
                if agent.next_node_idx is None or agent.target_node is None:
                    agent.state = AgentState.FREE
                    agent.target_node = None
                    agent.history.append((new_time, agent.position[0], agent.position[1], "free"))
                    continue
            else:
                # Shortest path: check if path is exhausted
                if not agent.path or agent.path_index >= len(agent.path) - 1:
                    agent.state = AgentState.FREE
                    agent.target_node = None
                    agent.history.append((new_time, agent.position[0], agent.position[1], "free"))
                    continue
            
            # Process agent movement and update position
            self._process_agent_movement(agent, new_time)
        
        return new_time
    
    def _process_agent_movement(self, agent: Agent, current_time: float):
        """
        Process an agent's movement up to current_time.
        
        Updates agent position based on edge traversal and traffic light waits.
        The agent makes a decision (moves to next edge) only when completing
        the current edge travel + traffic light wait.
        
        Supports both shortest path policy and Q-table greedy policy.
        """
        max_iterations = 1000  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get current and next node
            u = agent.current_node
            u_idx = agent.current_node_idx
            
            if self.use_q_policy:
                # Q-table policy: next node determined by Q-table
                v_idx = agent.next_node_idx
                if v_idx is None or v_idx < 0:
                    agent.state = AgentState.FREE
                    agent.target_node = None
                    break
                v = self.idx_to_node[v_idx]
            else:
                # Shortest path policy: next node from precomputed path
                if agent.path_index >= len(agent.path) - 1:
                    agent.state = AgentState.FREE
                    agent.target_node = None
                    agent.path = []
                    break
                v = agent.path[agent.path_index + 1]
                v_idx = self.node_to_idx[v]
            
            # Calculate when agent finishes traveling this edge
            edge_travel_end = agent.current_edge_start_time + agent.current_edge_travel_time
            
            # Calculate traffic light wait time
            wait_time = self.compute_wait_time(u, v, edge_travel_end)
            
            # Total time to complete this edge (travel + wait)
            edge_complete_time = edge_travel_end + wait_time
            
            if current_time < edge_travel_end:
                # Still traveling on edge - interpolate position
                if agent.current_edge_travel_time > 0:
                    progress = (current_time - agent.current_edge_start_time) / agent.current_edge_travel_time
                    progress = min(1.0, max(0.0, progress))
                else:
                    progress = 1.0
                
                x0, y0 = self.node_coords[u]
                x1, y1 = self.node_coords[v]
                agent.position = (
                    x0 + progress * (x1 - x0),
                    y0 + progress * (y1 - y0)
                )
                break
                
            elif current_time < edge_complete_time:
                # Finished traveling but waiting at traffic light
                agent.position = self.node_coords[v]
                break
                
            else:
                # Edge complete - move to next node
                agent.current_node = v
                agent.current_node_idx = v_idx
                agent.position = self.node_coords[v]
                agent.sim_time = edge_complete_time
                
                if not self.use_q_policy:
                    agent.path_index += 1
                
                # Check if reached destination
                if agent.current_node == agent.target_node:
                    # Check if round-trip mode and should return
                    if getattr(self, 'round_trip_mode', False) and agent.trip_direction == "to_pickup":
                        # Start return trip: swap start and pickup
                        agent.trip_direction = "returning"
                        new_target = agent.original_start_node
                        agent.target_node = new_target
                        agent.target_node_idx = self.node_to_idx[new_target]
                        
                        # Set up path for return trip
                        if self.use_q_policy:
                            next_node_idx = self.get_next_node_greedy(
                                agent.current_node_idx,
                                agent.target_node_idx,
                                edge_complete_time
                            )
                            agent.next_node_idx = next_node_idx
                            agent.path = []
                            agent.path_index = 0
                            next_node = self.idx_to_node[next_node_idx]
                        else:
                            path = self.get_shortest_path(agent.current_node, new_target)
                            if path and len(path) >= 2:
                                agent.path = path
                                agent.path_index = 0
                                agent.next_node_idx = self.node_to_idx[path[1]]
                                next_node = path[1]
                            else:
                                # Can't find return path, mark as complete
                                agent.state = AgentState.FREE
                                agent.target_node = None
                                agent.target_node_idx = None
                                agent.path = []
                                agent.next_node_idx = None
                                agent.num_trips_completed += 1
                                break
                        
                        # Start first edge of return trip
                        agent.current_edge_start_time = edge_complete_time
                        agent.current_edge_travel_time = self.get_edge_travel_time(agent.current_node, next_node)
                        # Continue to process next edge
                    else:
                        # Trip complete (either no round-trip or already returned)
                        if agent.trip_direction == "returning":
                            agent.num_trips_completed += 1
                        agent.state = AgentState.FREE
                        agent.target_node = None
                        agent.target_node_idx = None
                        agent.path = []
                        agent.next_node_idx = None
                        break
                
                # Determine next node for next edge
                if self.use_q_policy:
                    # Query Q-table for next action
                    next_node_idx = self.get_next_node_greedy(
                        agent.current_node_idx,
                        agent.target_node_idx,
                        edge_complete_time
                    )
                    agent.next_node_idx = next_node_idx
                    next_node = self.idx_to_node[next_node_idx]
                else:
                    # Get next node from path
                    if agent.path_index >= len(agent.path) - 1:
                        agent.state = AgentState.FREE
                        agent.target_node = None
                        agent.path = []
                        break
                    next_node = agent.path[agent.path_index + 1]
                    agent.next_node_idx = self.node_to_idx[next_node]
                
                # Start next edge
                agent.current_edge_start_time = edge_complete_time
                agent.current_edge_travel_time = self.get_edge_travel_time(v, next_node)
                
                # Continue loop to check if we've also completed the next edge
        
        # Record history
        state_str = "on_ride" if agent.state == AgentState.ON_A_RIDE else "free"
        agent.history.append((current_time, agent.position[0], agent.position[1], state_str))
    
    def run_simulation(
        self,
        num_agents: int,
        num_pickups: int,
        max_time: float = 600.0,
        dt: float = 1.0,
        seed: Optional[int] = None,
        q_table_path: Optional[str] = None,
        fixed_start_nodes: Optional[List[int]] = None,
        fixed_pickup_nodes: Optional[List[int]] = None,
        round_trip: bool = False,
        reassignment_interval: float = 120.0
    ) -> Tuple[List[Agent], List[Tuple[int, int, int]]]:
        """
        Run a complete multi-agent simulation.
        
        Args:
            num_agents: Number of agents to initialize
            num_pickups: Number of pickups to sample
            max_time: Maximum simulation time in seconds
            dt: Time step for simulation
            seed: Random seed
            q_table_path: Path to Q-table file for greedy policy (None = shortest path)
            fixed_start_nodes: Optional list of node IDs to sample starts from (e.g., 3 fixed nodes)
            fixed_pickup_nodes: Optional list of node IDs to sample pickups from (e.g., 3 fixed nodes)
            round_trip: If True, agents travel back to start after reaching pickup
            reassignment_interval: Seconds between reassignment batches (default 120s).
                Free agents accumulate and are reassigned together every this many seconds.
            
        Returns:
            Tuple of (agents list, assignments list)
        """
        self.round_trip_mode = round_trip
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        
        # Load Q-table if specified
        if q_table_path is not None:
            if self.load_q_table(q_table_path):
                print(f"Using Q-table greedy policy")
            else:
                print(f"Failed to load Q-table, falling back to shortest path policy")
                self.use_q_policy = False
        else:
            self.use_q_policy = False
            print(f"Using shortest path policy")
        
        # Initialize agents at random positions from fixed_start_nodes or all nodes
        start_pool = fixed_start_nodes if fixed_start_nodes is not None else self.all_nodes
        initial_positions = np.random.choice(start_pool, size=num_agents, replace=True).tolist()
        self.initialize_agents(num_agents, initial_positions)
        
        # Sample pickups from fixed_pickup_nodes or all nodes
        pickup_pool = fixed_pickup_nodes if fixed_pickup_nodes is not None else self.all_nodes
        pickups = np.random.choice(pickup_pool, size=num_pickups, replace=True).tolist()
        
        # Perform optimal transport matching and assign
        assignments = self.assign_pickups_to_agents(pickups, current_time=0.0)
        
        policy_name = "Q-table Greedy" if self.use_q_policy else "Shortest Path"
        print(f"\nAssignments (Optimal Transport, {policy_name} Policy):")
        for agent_id, start, pickup in assignments:
            path = self.get_shortest_path(start, pickup)
            times = self.get_path_travel_times(path)
            total_time = sum(times)
            print(f"  Agent {agent_id}: Node {start} → Node {pickup} (SP: {len(path)} nodes, {total_time:.1f}s)")
        
        # Store pickup pool for continuous reassignment
        self._pickup_pool = pickup_pool
        
        # Run simulation
        current_time = 0.0
        reassignment_count = 0
        next_reassignment_time = 0.0  # First reassignment happens immediately (initial is already done)
        # The initial assignment already happened above, so schedule the first periodic
        # reassignment after one full interval
        next_reassignment_time = reassignment_interval
        
        print(f"\nReassignment interval: {reassignment_interval:.0f}s (free agents accumulate between batches)")
        
        while current_time < max_time:
            # Check if it's time to reassign accumulated free agents
            if current_time >= next_reassignment_time:
                free_agents = [a for a in self.agents if a.is_free()]
                if free_agents:
                    # Sample new pickups for free agents
                    num_free = len(free_agents)
                    new_pickups = np.random.choice(self._pickup_pool, size=num_free, replace=True).tolist()
                    
                    # Perform optimal transport matching and assign
                    new_assignments = self.assign_pickups_to_agents(new_pickups, current_time=current_time)
                    
                    if new_assignments:
                        reassignment_count += 1
                        print(f"\n[t={current_time:.1f}s] Reassignment #{reassignment_count} ({len(new_assignments)} agents, {num_free - len(new_assignments)} still free):")
                        for agent_id, start, pickup in new_assignments:
                            path = self.get_shortest_path(start, pickup)
                            total_time = sum(self.get_path_travel_times(path)) if path else 0
                            print(f"  Agent {agent_id}: Node {start} → Node {pickup} ({len(path)} nodes, {total_time:.1f}s)")
                        
                        # Add to total assignments
                        assignments.extend(new_assignments)
                
                # Schedule next reassignment
                next_reassignment_time += reassignment_interval
            
            current_time = self.step_simulation(current_time, dt)
        
        # Final state for any remaining agents
        for agent in self.agents:
            if agent.history and agent.history[-1][0] < max_time:
                state_str = "on_ride" if agent.state == AgentState.ON_A_RIDE else "free"
                agent.history.append((max_time, agent.position[0], agent.position[1], state_str))
        
        print(f"\nSimulation complete: {reassignment_count} total reassignments")
        return self.agents, assignments
    
    def interpolate_position(
        self, 
        path: List[int], 
        travel_times: List[float],
        current_time: float
    ) -> Tuple[float, float, int]:
        """
        Interpolate the agent's position along the path at a given time.
        
        Args:
            path: List of node IDs forming the path
            travel_times: Travel time for each edge (in seconds)
            current_time: Current simulation time (in seconds)
            
        Returns:
            Tuple of (longitude, latitude, current_edge_index)
        """
        if len(path) < 2:
            x, y = self.node_coords[path[0]]
            return x, y, 0
        
        cumulative_time = 0.0
        for i, edge_time in enumerate(travel_times):
            if cumulative_time + edge_time >= current_time:
                # Interpolate within this edge
                fraction = (current_time - cumulative_time) / edge_time if edge_time > 0 else 1.0
                fraction = min(1.0, max(0.0, fraction))
                
                x0, y0 = self.node_coords[path[i]]
                x1, y1 = self.node_coords[path[i + 1]]
                
                x = x0 + fraction * (x1 - x0)
                y = y0 + fraction * (y1 - y0)
                return x, y, i
            cumulative_time += edge_time
        
        # Past the end - return final position
        x, y = self.node_coords[path[-1]]
        return x, y, len(travel_times) - 1
    
    def create_animation(
        self,
        trips: List[Tuple[int, int]],
        fps: int = 30,
        speed_factor: float = 10.0,
        agent_colors: Optional[List[str]] = None,
        show_paths: bool = True,
        title: str = "Manhattan Ride-Sharing Simulation"
    ) -> go.Figure:
        """
        Create a Plotly animation showing agents moving along their routes.
        
        Args:
            trips: List of (start_node, pickup_node) tuples
            fps: Frames per second for the animation
            speed_factor: Speed up factor for the simulation
            agent_colors: List of colors for each agent (None for default)
            show_paths: Whether to show the full path for each agent
            title: Title for the animation
            
        Returns:
            Plotly Figure with animation
        """
        if agent_colors is None:
            default_colors = [
                '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                '#1abc9c', '#e91e63', '#00bcd4', '#ff5722', '#607d8b'
            ]
            agent_colors = [default_colors[i % len(default_colors)] for i in range(len(trips))]
        
        # Compute paths and travel times for each trip
        paths = []
        travel_times_list = []
        total_times = []
        
        for start_node, pickup_node in trips:
            path = self.get_shortest_path(start_node, pickup_node)
            if not path:
                path = [start_node]
            paths.append(path)
            times = self.get_path_travel_times(path)
            travel_times_list.append(times)
            total_times.append(sum(times))
        
        # Find the maximum travel time to determine animation duration
        max_time = max(total_times) if total_times else 0
        
        # Generate animation frames
        num_frames = int(max_time / speed_factor * fps)
        num_frames = max(10, min(num_frames, 500))  # Limit frames for performance
        
        frames = []
        time_step = max_time / num_frames
        
        # Calculate map center and zoom
        center_lon, center_lat, map_zoom, _, _ = self._calculate_map_bounds()
        
        # Create figure with map
        fig = go.Figure()
        
        # Add road network
        fig.add_trace(self.edge_traces)
        
        # Add path traces for each agent (if show_paths)
        if show_paths:
            for i, (path, color) in enumerate(zip(paths, agent_colors)):
                path_x = [self.node_coords[n][0] for n in path]
                path_y = [self.node_coords[n][1] for n in path]
                fig.add_trace(go.Scattermapbox(
                    lon=path_x,
                    lat=path_y,
                    mode='lines',
                    line=dict(width=3, color=color),
                    opacity=0.6,
                    name=f'Agent {i+1} Route',
                    hoverinfo='name'
                ))
        
        # Add start markers
        start_lons = [self.node_coords[trips[i][0]][0] for i in range(len(trips))]
        start_lats = [self.node_coords[trips[i][0]][1] for i in range(len(trips))]
        fig.add_trace(go.Scattermapbox(
            lon=start_lons,
            lat=start_lats,
            mode='markers',
            marker=dict(size=12, color='green', symbol='circle'),
            name='Start Points',
            hovertemplate='Start %{customdata}<extra></extra>',
            customdata=[f'Agent {i+1}' for i in range(len(trips))]
        ))
        
        # Add pickup markers
        pickup_lons = [self.node_coords[trips[i][1]][0] for i in range(len(trips))]
        pickup_lats = [self.node_coords[trips[i][1]][1] for i in range(len(trips))]
        fig.add_trace(go.Scattermapbox(
            lon=pickup_lons,
            lat=pickup_lats,
            mode='markers',
            marker=dict(size=12, color='red', symbol='circle'),
            name='Pickup Points',
            hovertemplate='Pickup %{customdata}<extra></extra>',
            customdata=[f'Agent {i+1}' for i in range(len(trips))]
        ))
        
        # Build all frame data first
        all_frame_data = []
        for frame_idx in range(num_frames + 1):
            current_time = frame_idx * time_step
            frame_data = {'lons': [], 'lats': [], 'texts': []}
            
            for i, (path, times) in enumerate(zip(paths, travel_times_list)):
                lon, lat, edge_idx = self.interpolate_position(path, times, current_time)
                frame_data['lons'].append(lon)
                frame_data['lats'].append(lat)
                
                total_trip_time = sum(times)
                status = "🏁 Arrived!" if current_time >= total_trip_time else "En route"
                frame_data['texts'].append(f'Agent {i+1}<br>{status}<br>Time: {current_time:.1f}s')
            
            all_frame_data.append(frame_data)
        
        # Add initial agent positions (this trace will be animated)
        agent_trace_idx = len(fig.data)
        fig.add_trace(go.Scattermapbox(
            lon=all_frame_data[0]['lons'],
            lat=all_frame_data[0]['lats'],
            mode='markers',
            marker=dict(
                size=20,
                color=agent_colors,
            ),
            name='Agents',
            hovertemplate='%{text}<extra></extra>',
            text=all_frame_data[0]['texts']
        ))
        
        # Generate frames for animation
        for frame_idx, fd in enumerate(all_frame_data):
            frames.append(go.Frame(
                data=[go.Scattermapbox(
                    lon=fd['lons'],
                    lat=fd['lats'],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color=agent_colors,
                    ),
                    hovertemplate='%{text}<extra></extra>',
                    text=fd['texts']
                )],
                name=str(frame_idx),
                traces=[agent_trace_idx]
            ))
        
        fig.frames = frames
        
        # Create slider steps
        slider_steps = []
        step_interval = max(1, num_frames // 30)
        for k in range(0, num_frames + 1, step_interval):
            slider_steps.append({
                "args": [[str(k)], {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }],
                "label": f"{k * time_step:.1f}",
                "method": "animate"
            })
        
        # Animation controls
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            mapbox=dict(
                style='carto-positron',
                center=dict(lon=center_lon, lat=center_lat),
                zoom=map_zoom
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.9)"
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=True,
                    y=0.02,
                    x=0.02,
                    xanchor="left",
                    yanchor="bottom",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                                "mode": "immediate"
                            }]
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        )
                    ]
                )
            ],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 14},
                    "prefix": "Time: ",
                    "suffix": "s",
                    "visible": True,
                    "xanchor": "left"
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.05,
                "y": 0,
                "steps": slider_steps
            }],
            margin=dict(l=0, r=0, t=50, b=100),
            height=750
        )
        
        return fig
    
    def create_multi_agent_animation(
        self,
        agents: Optional[List[Agent]] = None,
        assignments: Optional[List[Tuple[int, int, int]]] = None,
        fps: int = 30,
        show_paths: bool = True,
        title: str = "Manhattan Multi-Agent Simulation"
    ) -> go.Figure:
        """
        Create a Plotly animation from the multi-agent simulation results.
        
        Args:
            agents: List of Agent objects with history (uses self.agents if None)
            assignments: List of (agent_id, start, pickup) assignments
            fps: Frames per second for animation
            show_paths: Whether to show agent paths
            title: Animation title
            
        Returns:
            Plotly Figure with animation
        """
        if agents is None:
            agents = self.agents
        
        if not agents:
            raise ValueError("No agents to animate. Run simulation first.")
        
        # Define colors for agents
        default_colors = [
            '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e91e63', '#00bcd4', '#ff5722', '#607d8b'
        ]
        agent_colors = [default_colors[i % len(default_colors)] for i in range(len(agents))]
        
        # Calculate map center and zoom
        center_lon, center_lat, map_zoom, _, _ = self._calculate_map_bounds()
        
        # Get max time from agent histories
        max_time = 0
        for agent in agents:
            if agent.history:
                max_time = max(max_time, agent.history[-1][0])
        
        if max_time == 0:
            raise ValueError("No history data in agents")
        
        # Sample frames at regular intervals (more frames = smoother animation)
        num_frames = min(300, int(max_time * 2))  # ~2 frames per second of simulation
        num_frames = max(50, num_frames)
        frame_times = np.linspace(0, max_time, num_frames)
        
        # Count number of static traces (roads + paths + start/pickup markers)
        num_static_traces = 1  # road network
        if show_paths and assignments:
            num_static_traces += len(assignments)  # path traces
        if assignments:
            num_static_traces += 2  # start and pickup markers
        
        # Build all frame data first to get positions
        all_frame_data = []
        for t in frame_times:
            frame_data = {
                'lons': [],
                'lats': [],
                'colors': [],
                'texts': []
            }
            for agent in agents:
                lon, lat, state = self._interpolate_agent_position(agent, t)
                frame_data['lons'].append(lon)
                frame_data['lats'].append(lat)
                color = '#2ecc71' if state == 'free' else agent_colors[agent.agent_id % len(agent_colors)]
                frame_data['colors'].append(color)
                frame_data['texts'].append(f"Agent {agent.agent_id}<br>State: {state}<br>Time: {t:.1f}s")
            all_frame_data.append(frame_data)
        
        # Create figure with initial frame data
        fig = go.Figure()
        
        # Add road network (trace 0)
        fig.add_trace(self.edge_traces)
        
        # Add paths for each agent (traces 1 to len(assignments))
        if show_paths and assignments:
            for agent_id, start, pickup in assignments:
                path = self.get_shortest_path(start, pickup)
                if path:
                    path_x = [self.node_coords[n][0] for n in path]
                    path_y = [self.node_coords[n][1] for n in path]
                    color = agent_colors[agent_id % len(agent_colors)]
                    fig.add_trace(go.Scattermapbox(
                        lon=path_x,
                        lat=path_y,
                        mode='lines',
                        line=dict(width=4, color=color),
                        opacity=0.7,
                        name=f'Agent {agent_id} Route',
                        hoverinfo='name'
                    ))
        
        # Add start markers
        if assignments:
            start_lons = [self.node_coords[a[1]][0] for a in assignments]
            start_lats = [self.node_coords[a[1]][1] for a in assignments]
            fig.add_trace(go.Scattermapbox(
                lon=start_lons,
                lat=start_lats,
                mode='markers',
                marker=dict(size=14, color='#27ae60', symbol='circle'),
                name='Start Points',
                hovertemplate='Start Agent %{customdata}<extra></extra>',
                customdata=[a[0] for a in assignments]
            ))
            
            pickup_lons = [self.node_coords[a[2]][0] for a in assignments]
            pickup_lats = [self.node_coords[a[2]][1] for a in assignments]
            fig.add_trace(go.Scattermapbox(
                lon=pickup_lons,
                lat=pickup_lats,
                mode='markers',
                marker=dict(size=14, color='#c0392b', symbol='circle'),
                name='Pickup Points',
                hovertemplate='Pickup Agent %{customdata}<extra></extra>',
                customdata=[a[0] for a in assignments]
            ))
        
        # Add agent markers (this is the trace that will be animated)
        # This is the last trace (index = num_static_traces)
        agent_trace_idx = len(fig.data)
        fig.add_trace(go.Scattermapbox(
            lon=all_frame_data[0]['lons'],
            lat=all_frame_data[0]['lats'],
            mode='markers',
            marker=dict(
                size=20,
                color=all_frame_data[0]['colors'],
            ),
            name='Agents',
            hovertemplate='%{text}<extra></extra>',
            text=all_frame_data[0]['texts']
        ))
        
        # Generate frames - each frame updates ONLY the agent trace
        frames = []
        for frame_idx, (t, fd) in enumerate(zip(frame_times, all_frame_data)):
            frames.append(go.Frame(
                data=[go.Scattermapbox(
                    lon=fd['lons'],
                    lat=fd['lats'],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color=fd['colors'],
                    ),
                    hovertemplate='%{text}<extra></extra>',
                    text=fd['texts']
                )],
                name=str(frame_idx),
                traces=[agent_trace_idx]  # Only update the agent trace
            ))
        
        fig.frames = frames
        
        # Create slider steps
        slider_steps = []
        step_interval = max(1, len(frame_times) // 30)  # ~30 slider steps
        for k in range(0, len(frame_times), step_interval):
            slider_steps.append({
                "args": [[str(k)], {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }],
                "label": f"{frame_times[k]:.0f}",
                "method": "animate"
            })
        
        # Layout with controls
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            mapbox=dict(
                style='carto-positron',
                center=dict(lon=center_lon, lat=center_lat),
                zoom=map_zoom
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.9)"
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=True,
                    y=0.02,
                    x=0.02,
                    xanchor="left",
                    yanchor="bottom",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                                "mode": "immediate"
                            }]
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        )
                    ]
                )
            ],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 14},
                    "prefix": "Time: ",
                    "suffix": "s",
                    "visible": True,
                    "xanchor": "left"
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.05,
                "y": 0,
                "steps": slider_steps
            }],
            margin=dict(l=0, r=0, t=50, b=100),
            height=750
        )
        
        return fig
    
    def _interpolate_agent_position(
        self, 
        agent: Agent, 
        t: float
    ) -> Tuple[float, float, str]:
        """
        Interpolate agent position at time t from history.
        
        Args:
            agent: Agent object with history
            t: Time to interpolate at
            
        Returns:
            Tuple of (lon, lat, state)
        """
        if not agent.history:
            return agent.position[0], agent.position[1], "free"
        
        # Handle edge cases
        if t <= agent.history[0][0]:
            return agent.history[0][1], agent.history[0][2], agent.history[0][3]
        if t >= agent.history[-1][0]:
            return agent.history[-1][1], agent.history[-1][2], agent.history[-1][3]
        
        # Binary search for the bracketing entries
        left, right = 0, len(agent.history) - 1
        while left < right - 1:
            mid = (left + right) // 2
            if agent.history[mid][0] <= t:
                left = mid
            else:
                right = mid
        
        t0, x0, y0, s0 = agent.history[left]
        t1, x1, y1, s1 = agent.history[right]
        
        if t1 == t0:
            return x0, y0, s0
        
        # Linear interpolation
        alpha = (t - t0) / (t1 - t0)
        alpha = min(1.0, max(0.0, alpha))
        
        lon = x0 + alpha * (x1 - x0)
        lat = y0 + alpha * (y1 - y0)
        
        # Use state from the earlier time point
        return lon, lat, s0

    def get_random_trips(self, num_trips: int, seed: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Generate random trips (start, pickup) pairs.
        
        Args:
            num_trips: Number of trips to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of (start_node, pickup_node) tuples
        """
        if seed is not None:
            np.random.seed(seed)
        
        trips = []
        for _ in range(num_trips):
            start_idx = np.random.randint(len(self.all_nodes))
            pickup_idx = np.random.randint(len(self.all_nodes))
            # Ensure start and pickup are different
            while pickup_idx == start_idx:
                pickup_idx = np.random.randint(len(self.all_nodes))
            
            start_node = self.all_nodes[start_idx]
            pickup_node = self.all_nodes[pickup_idx]
            trips.append((start_node, pickup_node))
        
        return trips
    
    def get_trips_by_zone(
        self, 
        start_zone_names: List[str], 
        pickup_zone_names: List[str],
        num_trips: int = 1,
        seed: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Generate trips between specified zones.
        
        Args:
            start_zone_names: List of zone names for start locations
            pickup_zone_names: List of zone names for pickup locations
            num_trips: Number of trips to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of (start_node, pickup_node) tuples
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Get nodes in start zones
        start_nodes = []
        for zone_name in start_zone_names:
            if zone_name in self.zone_name_to_locationID:
                loc_id = self.zone_name_to_locationID[zone_name]
                if loc_id in self.zone_to_nodes:
                    nodes = [n for n in self.zone_to_nodes[loc_id] if n in self.node_to_idx]
                    start_nodes.extend(nodes)
        
        # Get nodes in pickup zones
        pickup_nodes = []
        for zone_name in pickup_zone_names:
            if zone_name in self.zone_name_to_locationID:
                loc_id = self.zone_name_to_locationID[zone_name]
                if loc_id in self.zone_to_nodes:
                    nodes = [n for n in self.zone_to_nodes[loc_id] if n in self.node_to_idx]
                    pickup_nodes.extend(nodes)
        
        if not start_nodes or not pickup_nodes:
            print(f"Warning: Could not find nodes in specified zones")
            return self.get_random_trips(num_trips, seed)
        
        trips = []
        for _ in range(num_trips):
            start_node = start_nodes[np.random.randint(len(start_nodes))]
            pickup_node = pickup_nodes[np.random.randint(len(pickup_nodes))]
            trips.append((start_node, pickup_node))
        
        return trips
    
    def show(self, fig: go.Figure):
        """Display the figure in a browser or notebook."""
        fig.show()
    
    def save_html(self, fig: go.Figure, filename: str = "simulation.html"):
        """Save the animation as an interactive HTML file."""
        fig.write_html(filename)
        print(f"Animation saved to {filename}")


def main():
    """Example usage of the simulator with multi-agent simulation."""
    print("=" * 60)
    print("  Manhattan Multi-Agent Ride-Sharing Simulator")
    print("=" * 60)
    
    # Initialize simulator with default Manhattan zones
    sim = ManhattanSimulator(
        place_name="Manhattan, New York City, New York, USA",
        zone_shp="../data/processed/taxi_zones.shp",
        no_congestion=False,
        cycle_length=90.0,  # 90 second traffic light cycle
        random_offsets=True,  # Random traffic light offsets
        seed=42
    )
    
    # Run multi-agent simulation
    print("\n--- Running Multi-Agent Simulation ---")
    num_agents = 5
    num_pickups = 5
    
    agents, assignments = sim.run_simulation(
        num_agents=num_agents,
        num_pickups=num_pickups,
        max_time=600.0,  # 10 minutes max
        dt=0.5,  # 0.5 second time steps
        seed=42
    )
    
    # Print final statistics
    print(f"\n--- Simulation Complete ---")
    total_travel_time = 0
    for agent_id, start, pickup in assignments:
        # Find agent's total travel time from history
        agent = agents[agent_id]
        start_time = agent.history[0][0]
        # Find when agent became free
        end_time = start_time
        for t, _, _, state in agent.history:
            if state == "on_ride":
                end_time = t
        trip_time = end_time - start_time
        total_travel_time += trip_time
        print(f"  Agent {agent_id}: Travel time = {trip_time:.1f}s")
    
    print(f"\nTotal travel time (all agents): {total_travel_time:.1f}s")
    print(f"Average travel time per agent: {total_travel_time / len(assignments):.1f}s")
    
    # Create animation
    print("\n--- Creating Animation ---")
    fig = sim.create_multi_agent_animation(
        agents=agents,
        assignments=assignments,
        fps=30,
        show_paths=True,
        title=f"Manhattan Multi-Agent Simulation ({num_agents} agents, Optimal Transport Matching)"
    )
    
    # Save and show
    sim.save_html(fig, "multi_agent_simulation.html")
    print("\nOpening animation in browser...")
    sim.show(fig)


def demo_simple():
    """Simple demo with predefined trips (original interface)."""
    sim = ManhattanSimulator(
        place_name="Manhattan, New York City, New York, USA",
        zone_shp="../data/processed/taxi_zones.shp",
        no_congestion=False
    )
    
    # Generate random trips
    trips = sim.get_random_trips(num_trips=3, seed=42)
    print(f"\nGenerated trips:")
    for i, (start, pickup) in enumerate(trips):
        path = sim.get_shortest_path(start, pickup)
        times = sim.get_path_travel_times(path)
        total_time = sum(times)
        print(f"  Agent {i+1}: {start} → {pickup} ({len(path)} nodes, {total_time:.1f}s)")
    
    # Create and display animation
    fig = sim.create_animation(
        trips=trips,
        fps=30,
        speed_factor=10.0,
        show_paths=True,
        title="Manhattan Ride-Sharing Simulation (Shortest Path Policy)"
    )
    
    # Save and show
    sim.save_html(fig, "manhattan_simulation.html")
    sim.show(fig)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        demo_simple()
    else:
        main()
