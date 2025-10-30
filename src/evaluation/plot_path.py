#!/usr/bin/env python3
"""
Path plotting utilities for Manhattan grid visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
from typing import List, Optional, Tuple, Union
import jax.numpy as jnp


def plot_path_in_grid(
    node_indices: List[int],
    G: nx.DiGraph,
    node_to_idx: dict,
    idx_to_node: List[int],
    title: str = "Path Visualization",
    figsize: Tuple[int, int] = (12, 8),
    show_grid: bool = True,
    show_nodes: bool = True,
    show_edges: bool = False,
    start_marker: str = 'o',
    end_marker: str = 's',
    path_color: str = 'red',
    path_width: float = 3.0,
    node_size: int = 50,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot a path through the Manhattan grid given a series of node indices.
    
    Args:
        node_indices: List of node indices representing the path
        G: NetworkX graph containing the grid
        node_to_idx: Dictionary mapping node IDs to indices
        idx_to_node: List mapping indices to node IDs
        title: Title for the plot
        figsize: Figure size (width, height)
        show_grid: Whether to show grid lines
        show_nodes: Whether to show all nodes in the graph
        show_edges: Whether to show all edges in the graph
        start_marker: Marker style for start node
        end_marker: Marker style for end node
        path_color: Color for the path line
        path_width: Width of the path line
        node_size: Size of nodes
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    
    # Extract coordinates for all nodes
    node_coords = {}
    for node in G.nodes():
        if 'x' in G.nodes[node] and 'y' in G.nodes[node]:
            node_coords[node] = (G.nodes[node]['x'], G.nodes[node]['y'])
        else:
            # Fallback: use node ID as coordinates if x,y not available
            node_coords[node] = (node, 0)
    
    # Convert node indices to actual node IDs
    node_ids = [idx_to_node[idx] for idx in node_indices if idx < len(idx_to_node)]
    
    # Get coordinates for the path
    path_coords = [node_coords[node_id] for node_id in node_ids if node_id in node_coords]
    
    if not path_coords:
        raise ValueError("No valid coordinates found for the given node indices")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all nodes if requested
    if show_nodes:
        all_x = [coord[0] for coord in node_coords.values()]
        all_y = [coord[1] for coord in node_coords.values()]
        ax.scatter(all_x, all_y, c='lightgray', s=node_size//2, alpha=0.5, zorder=1)
    
    # Plot all edges if requested
    if show_edges:
        for edge in G.edges():
            if edge[0] in node_coords and edge[1] in node_coords:
                x_coords = [node_coords[edge[0]][0], node_coords[edge[1]][0]]
                y_coords = [node_coords[edge[0]][1], node_coords[edge[1]][1]]
                ax.plot(x_coords, y_coords, 'lightblue', alpha=0.3, linewidth=0.5, zorder=1)
    
    # Plot the path
    if len(path_coords) > 1:
        path_x = [coord[0] for coord in path_coords]
        path_y = [coord[1] for coord in path_coords]
        
        # Draw the path line
        ax.plot(path_x, path_y, color=path_color, linewidth=path_width, 
                alpha=0.8, zorder=3, label='Path')
        
        # Add arrows to show direction
        for i in range(len(path_coords) - 1):
            dx = path_x[i+1] - path_x[i]
            dy = path_y[i+1] - path_y[i]
            ax.annotate('', xy=(path_x[i+1], path_y[i+1]), 
                       xytext=(path_x[i], path_y[i]),
                       arrowprops=dict(arrowstyle='->', color=path_color, 
                                     lw=path_width*0.7, alpha=0.8))
    
    # Mark start and end points
    if len(path_coords) > 0:
        # Start point
        ax.scatter(path_coords[0][0], path_coords[0][1], 
                  c='green', s=node_size*2, marker=start_marker, 
                  label='Start', zorder=4, edgecolors='black', linewidth=2)
        
        # End point
        if len(path_coords) > 1:
            ax.scatter(path_coords[-1][0], path_coords[-1][1], 
                      c='red', s=node_size*2, marker=end_marker, 
                      label='End', zorder=4, edgecolors='black', linewidth=2)
    
    # Add grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3, zorder=0)
    
    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio for better visualization
    ax.set_aspect('equal', adjustable='box')
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Path plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_multiple_paths(
    paths: List[List[int]],
    G: nx.DiGraph,
    node_to_idx: dict,
    idx_to_node: List[int],
    path_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    title: str = "Multiple Paths Visualization",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot multiple paths on the same grid for comparison.
    
    Args:
        paths: List of paths, where each path is a list of node indices
        G: NetworkX graph containing the grid
        node_to_idx: Dictionary mapping node IDs to indices
        idx_to_node: List mapping indices to node IDs
        path_labels: Optional labels for each path
        colors: Optional colors for each path
        title: Title for the plot
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    
    # Default colors
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Default labels
    if path_labels is None:
        path_labels = [f'Path {i+1}' for i in range(len(paths))]
    
    # Extract coordinates for all nodes
    node_coords = {}
    for node in G.nodes():
        if 'x' in G.nodes[node] and 'y' in G.nodes[node]:
            node_coords[node] = (G.nodes[node]['x'], G.nodes[node]['y'])
        else:
            node_coords[node] = (node, 0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all nodes
    all_x = [coord[0] for coord in node_coords.values()]
    all_y = [coord[1] for coord in node_coords.values()]
    ax.scatter(all_x, all_y, c='lightgray', s=30, alpha=0.5, zorder=1)
    
    # Plot each path
    for i, path_indices in enumerate(paths):
        # Convert node indices to actual node IDs
        node_ids = [idx_to_node[idx] for idx in path_indices if idx < len(idx_to_node)]
        
        # Get coordinates for the path
        path_coords = [node_coords[node_id] for node_id in node_ids if node_id in node_coords]
        
        if len(path_coords) > 1:
            path_x = [coord[0] for coord in path_coords]
            path_y = [coord[1] for coord in path_coords]
            
            color = colors[i % len(colors)]
            label = path_labels[i] if i < len(path_labels) else f'Path {i+1}'
            
            # Draw the path line
            ax.plot(path_x, path_y, color=color, linewidth=2, 
                    alpha=0.8, zorder=3, label=label)
            
            # Add arrows to show direction
            for j in range(len(path_coords) - 1):
                dx = path_x[j+1] - path_x[j]
                dy = path_y[j+1] - path_y[j]
                ax.annotate('', xy=(path_x[j+1], path_y[j+1]), 
                           xytext=(path_x[j], path_y[j]),
                           arrowprops=dict(arrowstyle='->', color=color, 
                                         lw=1.5, alpha=0.8))
            
            # Mark start and end points
            ax.scatter(path_coords[0][0], path_coords[0][1], 
                      c=color, s=100, marker='o', zorder=4, 
                      edgecolors='black', linewidth=1)
            ax.scatter(path_coords[-1][0], path_coords[-1][1], 
                      c=color, s=100, marker='s', zorder=4, 
                      edgecolors='black', linewidth=1)
    
    # Add grid
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multiple paths plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def create_simple_grid_plot(
    node_indices: List[int],
    grid_width: int = 4,
    grid_height: int = 4,
    title: str = "Simple Grid Path",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create a simple grid plot for testing purposes when no graph is available.
    
    Args:
        node_indices: List of node indices representing the path
        grid_width: Width of the grid
        grid_height: Height of the grid
        title: Title for the plot
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    
    # Create a simple grid
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw grid lines
    for i in range(grid_width + 1):
        ax.axvline(i, color='lightgray', linewidth=1)
    for i in range(grid_height + 1):
        ax.axhline(i, color='lightgray', linewidth=1)
    
    # Convert node indices to coordinates
    coords = []
    for idx in node_indices:
        if idx < grid_width * grid_height:
            x = idx % grid_width
            y = idx // grid_width
            coords.append((x, y))
    
    if len(coords) > 1:
        # Plot the path
        path_x = [coord[0] for coord in coords]
        path_y = [coord[1] for coord in coords]
        
        ax.plot(path_x, path_y, 'ro-', linewidth=3, markersize=8, 
                label='Path', zorder=3)
        
        # Add arrows
        for i in range(len(coords) - 1):
            dx = path_x[i+1] - path_x[i]
            dy = path_y[i+1] - path_y[i]
            ax.annotate('', xy=(path_x[i+1], path_y[i+1]), 
                       xytext=(path_x[i], path_y[i]),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Mark start and end
        ax.scatter(coords[0][0], coords[0][1], c='green', s=200, 
                  marker='o', label='Start', zorder=4, edgecolors='black')
        ax.scatter(coords[-1][0], coords[-1][1], c='red', s=200, 
                  marker='s', label='End', zorder=4, edgecolors='black')
    
    # Set limits and labels
    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(-0.5, grid_height - 0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Simple grid plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


# Example usage function
def example_usage():
    """
    Example of how to use the path plotting functions.
    """
    # Example 1: Simple grid path
    print("Creating simple grid path example...")
    simple_path = [0, 1, 5, 9, 13, 14, 15]  # Example path in 4x4 grid
    create_simple_grid_plot(simple_path, title="Example Simple Grid Path")
    
    # Example 2: Multiple paths comparison
    print("Creating multiple paths example...")
    paths = [
        [0, 1, 2, 3, 7, 11, 15],  # Path 1
        [0, 4, 8, 12, 13, 14, 15]  # Path 2
    ]
    labels = ["Shortest Path", "Alternative Path"]
    colors = ["red", "blue"]
    
    # Note: This would need actual graph data to work
    # plot_multiple_paths(paths, G, node_to_idx, idx_to_node, 
    #                     path_labels=labels, colors=colors)


if __name__ == "__main__":
    example_usage()
