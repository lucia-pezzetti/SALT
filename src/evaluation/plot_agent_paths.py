#!/usr/bin/env python3
"""
Integration example showing how to plot agent paths from your ride-sharing simulator.
This script demonstrates how to use the path plotting functions with your existing codebase.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plot_path import plot_path_in_grid, plot_multiple_paths
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_agent_path_from_trajectory(
    trajectory: list,
    G: nx.DiGraph,
    node_to_idx: dict,
    idx_to_node: list,
    agent_id: int = 0,
    title: str = None,
    save_path: str = None
):
    """
    Plot the path taken by an agent from a trajectory.
    
    Args:
        trajectory: List of (state, action, reward, next_state) tuples or just node indices
        G: NetworkX graph
        node_to_idx: Dictionary mapping node IDs to indices
        idx_to_node: List mapping indices to node IDs
        agent_id: ID of the agent for labeling
        title: Optional title for the plot
        save_path: Optional path to save the plot
    """
    
    # Extract node indices from trajectory
    if isinstance(trajectory[0], tuple):
        # If trajectory contains (state, action, reward, next_state) tuples
        node_indices = [state.current_node for state, _, _, _ in trajectory]
    else:
        # If trajectory is just a list of node indices
        node_indices = trajectory
    
    if title is None:
        title = f"Agent {agent_id} Path"
    
    fig = plot_path_in_grid(
        node_indices,
        G,
        node_to_idx,
        idx_to_node,
        title=title,
        path_color='blue',
        save_path=save_path,
        show_plot=True
    )
    
    return fig


def plot_multiple_agent_paths(
    agent_trajectories: dict,
    G: nx.DiGraph,
    node_to_idx: dict,
    idx_to_node: list,
    title: str = "Multiple Agent Paths",
    save_path: str = None
):
    """
    Plot paths for multiple agents on the same grid.
    
    Args:
        agent_trajectories: Dictionary mapping agent_id to trajectory
        G: NetworkX graph
        node_to_idx: Dictionary mapping node IDs to indices
        idx_to_node: List mapping indices to node IDs
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    
    paths = []
    labels = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for agent_id, trajectory in agent_trajectories.items():
        # Extract node indices from trajectory
        if isinstance(trajectory[0], tuple):
            node_indices = [state.current_node for state, _, _, _ in trajectory]
        else:
            node_indices = trajectory
        
        paths.append(node_indices)
        labels.append(f"Agent {agent_id}")
    
    fig = plot_multiple_paths(
        paths,
        G,
        node_to_idx,
        idx_to_node,
        path_labels=labels,
        colors=colors[:len(paths)],
        title=title,
        save_path=save_path,
        show_plot=True
    )
    
    return fig


def plot_rl_vs_shortest_path(
    rl_trajectory: list,
    sp_trajectory: list,
    G: nx.DiGraph,
    node_to_idx: dict,
    idx_to_node: list,
    start_node: int,
    pickup_node: int,
    save_path: str = None
):
    """
    Compare RL agent path vs shortest path.
    
    Args:
        rl_trajectory: RL agent's trajectory
        sp_trajectory: Shortest path trajectory
        G: NetworkX graph
        node_to_idx: Dictionary mapping node IDs to indices
        idx_to_node: List mapping indices to node IDs
        start_node: Starting node index
        pickup_node: Pickup node index
        save_path: Optional path to save the plot
    """
    
    # Support both single path (list[int]) and multiple paths (list[list[int]]) inputs
    if rl_trajectory and isinstance(rl_trajectory[0], list):
        # Multiple trajectories provided
        rl_paths = rl_trajectory
        sp_paths = sp_trajectory if (sp_trajectory and isinstance(sp_trajectory[0], list)) else [sp_trajectory]
        paths = rl_paths + sp_paths
        labels = [f"RL Agent {i+1}" for i in range(len(rl_paths))] + [f"Shortest Path {i+1}" for i in range(len(sp_paths))]
        base_colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
        colors = [base_colors[i % len(base_colors)] for i in range(len(paths))]
    else:
        # Single trajectory compare
        paths = [rl_trajectory, sp_trajectory]
        labels = ["RL Agent", "Shortest Path"]
        colors = ["red", "blue"]
    
    fig = plot_multiple_paths(
        paths,
        G,
        node_to_idx,
        idx_to_node,
        path_labels=labels,
        colors=colors,
        title=f"RL vs Shortest Path (Start: {start_node}, Pickup: {pickup_node})",
        save_path=save_path,
        show_plot=True
    )
    
    return fig


def create_path_analysis_plot(
    agent_trajectories: dict,
    G: nx.DiGraph,
    node_to_idx: dict,
    idx_to_node: list,
    analysis_type: str = "efficiency",
    save_path: str = None
):
    """
    Create a comprehensive path analysis plot.
    
    Args:
        agent_trajectories: Dictionary mapping agent_id to trajectory
        G: NetworkX graph
        node_to_idx: Dictionary mapping node IDs to indices
        idx_to_node: List mapping indices to node IDs
        analysis_type: Type of analysis ("efficiency", "coverage", "congestion")
        save_path: Optional path to save the plot
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: All agent paths
    paths = []
    labels = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for agent_id, trajectory in agent_trajectories.items():
        if isinstance(trajectory[0], tuple):
            node_indices = [state.current_node for state, _, _, _ in trajectory]
        else:
            node_indices = trajectory
        
        paths.append(node_indices)
        labels.append(f"Agent {agent_id}")
    
    # Plot paths on the first subplot
    ax1.set_title("All Agent Paths")
    for i, (path, label, color) in enumerate(zip(paths, labels, colors[:len(paths)])):
        if len(path) > 1:
            # Get coordinates for the path
            path_coords = []
            for node_idx in path:
                if node_idx < len(idx_to_node):
                    node_id = idx_to_node[node_idx]
                    if node_id in G.nodes():
                        x = G.nodes[node_id].get('x', 0)
                        y = G.nodes[node_id].get('y', 0)
                        path_coords.append((x, y))
            
            if len(path_coords) > 1:
                path_x = [coord[0] for coord in path_coords]
                path_y = [coord[1] for coord in path_coords]
                ax1.plot(path_x, path_y, color=color, linewidth=2, label=label, alpha=0.8)
                
                # Add arrows
                for j in range(len(path_coords) - 1):
                    dx = path_x[j+1] - path_x[j]
                    dy = path_y[j+1] - path_y[j]
                    ax1.annotate('', xy=(path_x[j+1], path_y[j+1]), 
                               xytext=(path_x[j], path_y[j]),
                               arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Path statistics
    ax2.set_title("Path Statistics")
    
    path_lengths = [len(path) for path in paths]
    path_efficiencies = []
    
    for path in paths:
        if len(path) > 1:
            # Calculate efficiency as straight-line distance / actual path length
            start_coords = (G.nodes[idx_to_node[path[0]]].get('x', 0), 
                           G.nodes[idx_to_node[path[0]]].get('y', 0))
            end_coords = (G.nodes[idx_to_node[path[-1]]].get('x', 0), 
                         G.nodes[idx_to_node[path[-1]]].get('y', 0))
            
            straight_line_dist = np.sqrt((end_coords[0] - start_coords[0])**2 + 
                                        (end_coords[1] - start_coords[1])**2)
            actual_path_length = len(path) - 1  # Number of steps
            efficiency = straight_line_dist / max(actual_path_length, 1)
            path_efficiencies.append(efficiency)
        else:
            path_efficiencies.append(0)
    
    # Create bar chart of path lengths
    x_pos = range(len(labels))
    bars = ax2.bar(x_pos, path_lengths, color=colors[:len(labels)], alpha=0.7)
    ax2.set_xlabel('Agent')
    ax2.set_ylabel('Path Length (steps)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    
    # Add efficiency values on top of bars
    for i, (bar, efficiency) in enumerate(zip(bars, path_efficiencies)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'Eff: {efficiency:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Path analysis plot saved to: {save_path}")
    
    plt.show()
    
    return fig


# Example usage with your existing codebase structure
def example_integration():
    """
    Example showing how to integrate with your existing codebase.
    """
    print("This is an example of how to use the path plotting functions")
    print("with your existing ride-sharing simulator codebase.")
    print("\nTo use these functions in your code:")
    print("1. Import the plotting functions")
    print("2. Pass your graph G, node_to_idx, and idx_to_node from your environment")
    print("3. Extract node indices from your agent trajectories")
    print("4. Call the appropriate plotting function")
    print("\nExample code snippet:")
    print("""
    from plot_agent_paths import plot_agent_path_from_trajectory
    
    # Assuming you have your environment set up
    # G, node_to_idx, idx_to_node = your_environment_setup()
    
    # Extract trajectory from your agent
    trajectory = your_agent.get_trajectory()
    
    # Plot the path
    plot_agent_path_from_trajectory(
        trajectory, G, node_to_idx, idx_to_node,
        agent_id=0, title="Agent 0 Path"
    )
    """)


if __name__ == "__main__":
    example_integration()
