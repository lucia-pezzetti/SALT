"""
Graph Neural Network architectures for PPO policy and value networks.

These networks use graph structure (adjacency list, travel times) to learn
spatial patterns in the road network, which should improve BC accuracy and
generalization in Manhattan environments.

The key idea: Instead of processing the entire graph, we use a "local GNN"
that processes the current node's neighborhood using the adjacency structure.
This is more efficient and directly matches how the expert makes decisions.
"""

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional

# Map activation names
activation_dict = {"relu": jax.nn.relu, "silu": jax.nn.silu, "elu": jax.nn.elu}

# Base observation layout matches PPO networks
BASE_OBS_DIM = 5


class GNPPolicyNetwork(hk.Module):
    """
    Graph Neural Network Policy for PPO.
    
    Uses local graph convolution: processes neighbor features using message passing.
    Works with observations that include neighbor information (travel times, distances).
    This allows the network to learn spatial patterns in the graph structure.
    """
    
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.hidden_dim = config['num_hidden_units']
        self.num_layers = config['num_hidden_layers']
        self.activation = activation_dict[config['activation']]
        self.max_deg = config.get('max_deg', 8)
        self.cycle_length = config.get('cycle_length', 200)
        self.use_edge_features = config.get('use_edge_features', True)
        self.gnn_layers = config.get('gnn_layers', 2)  # Number of GNN message passing layers
    
    def __call__(self, obs, neighbor_mask=None):
        """
        GNN Policy network with local graph convolution.
        
        Uses neighbor information from observations to perform message passing.
        This allows learning spatial patterns in the graph structure.
        
        Args:
            obs: [batch_size, D] where D = 5 (base) or 5+2*max_deg (with neighbor info)
              Base: [current_pos(2), pickup_pos(2), time(1)]
              With neighbors: [current_pos(2), pickup_pos(2), time(1),
                               neighbor_travel_times(max_deg), neighbor_distances(max_deg)]
            neighbor_mask: [batch_size, max_deg] - mask for valid neighbors (optional)
        
        Returns:
            [batch_size, max_deg] - action logits
        """
        batch_size = obs.shape[0]
        obs_dim = obs.shape[-1]
        has_neighbor_info = obs_dim > BASE_OBS_DIM
        
        # Extract base features
        current_pos = obs[:, 0:2]  # [batch_size, 2]
        pickup_pos = obs[:, 2:4]  # [batch_size, 2]
        time = obs[:, 4:5] / self.cycle_length  # [batch_size, 1]
        
        # Get neighbor information if available
        if has_neighbor_info:
            start_idx = BASE_OBS_DIM
            neighbor_travel_times = obs[:, start_idx:start_idx+self.max_deg]  # [batch_size, max_deg]
            neighbor_distances = obs[:, start_idx+self.max_deg:start_idx+2*self.max_deg]  # [batch_size, max_deg]
            # Use provided mask or create default
            if neighbor_mask is None:
                # Assume all are valid if no mask provided
                neighbor_mask = jnp.ones((batch_size, self.max_deg), dtype=bool)
        else:
            # No neighbor info - fallback to MLP
            neighbor_travel_times = jnp.zeros((batch_size, self.max_deg))
            neighbor_distances = jnp.zeros((batch_size, self.max_deg))
            neighbor_mask = jnp.zeros((batch_size, self.max_deg), dtype=bool)
        
        # Create initial node features
        x = jnp.concatenate([current_pos, pickup_pos, time], axis=-1)  # [batch_size, 5]
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        
        # Project to hidden dimension
        x = hk.Linear(
            self.hidden_dim,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
            b_init=hk.initializers.Constant(0.0)
        )(x)
        x = self.activation(x)
        
        # Graph convolution layers: aggregate information from neighbors
        if has_neighbor_info:
            for gnn_layer in range(self.gnn_layers):
                # Normalize neighbor features for stability (per-sample normalization)
                # Use max per batch item to avoid division issues
                max_travel = jnp.max(neighbor_travel_times, axis=-1, keepdims=True) + 1e-8
                max_dist = jnp.max(neighbor_distances, axis=-1, keepdims=True) + 1e-8
                neighbor_travel_norm = neighbor_travel_times / max_travel
                neighbor_dist_norm = neighbor_distances / max_dist
                
                # Create neighbor feature vectors: [batch_size, max_deg, 2]
                neighbor_feats = jnp.stack([
                    neighbor_travel_norm,
                    neighbor_dist_norm
                ], axis=-1)  # [batch_size, max_deg, 2]
                
                # Project neighbor features to hidden dimension
                neighbor_feats_proj = hk.Linear(
                    self.hidden_dim,
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                    b_init=hk.initializers.Constant(0.0)
                )(neighbor_feats)  # [batch_size, max_deg, hidden_dim]
                
                # Apply edge weights (use travel times as attention weights)
                if self.use_edge_features:
                    # Inverse travel time = stronger connection (shorter = better)
                    edge_weights = 1.0 / (neighbor_travel_times + 1e-8)  # [batch_size, max_deg]
                    edge_weights = edge_weights * neighbor_mask  # Mask invalid neighbors
                    # Normalize weights to sum to 1
                    weights_sum = jnp.sum(edge_weights, axis=-1, keepdims=True) + 1e-8
                    edge_weights = edge_weights / weights_sum
                    edge_weights = edge_weights[..., None]  # [batch_size, max_deg, 1]
                else:
                    # Uniform weights for valid neighbors
                    num_valid = jnp.sum(neighbor_mask, axis=-1, keepdims=True) + 1e-8
                    edge_weights = (neighbor_mask / num_valid)[..., None]  # [batch_size, max_deg, 1]
                
                # Weighted aggregation of neighbor features (message passing)
                aggregated = jnp.sum(neighbor_feats_proj * edge_weights, axis=1)  # [batch_size, hidden_dim]
                
                # Combine with current node features (residual connection)
                x = x + aggregated  # [batch_size, hidden_dim]
                
                # Transform through MLP layer
                residual = x
                x = hk.Linear(
                    self.hidden_dim,
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                    b_init=hk.initializers.Constant(0.0)
                )(x)
                x = self.activation(x)
                x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
                x = x + residual  # Residual connection
        
        # Additional MLP layers (if more layers than GNN layers)
        remaining_layers = max(0, self.num_layers - (self.gnn_layers if has_neighbor_info else 0))
        for i in range(remaining_layers):
            residual = x if x.shape[-1] == self.hidden_dim else None
            x = hk.Linear(
                self.hidden_dim,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                b_init=hk.initializers.Constant(0.0)
            )(x)
            x = self.activation(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            if residual is not None:
                x = x + residual
        
        # Output action logits
        return hk.Linear(
            self.max_deg,
            w_init=hk.initializers.VarianceScaling(0.1, "fan_in", "truncated_normal"),
            b_init=hk.initializers.Constant(0.0)
        )(x)


class GNNValueNetwork(hk.Module):
    """Graph Neural Network Value function for PPO."""
    
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.hidden_dim = config['num_hidden_units']
        self.num_layers = config['num_hidden_layers']
        self.activation = activation_dict[config['activation']]
        self.max_deg = config.get('max_deg', 8)
        self.cycle_length = config.get('cycle_length', 200)
        self.use_edge_features = config.get('use_edge_features', True)
        self.gnn_layers = config.get('gnn_layers', 2)
        self.init_bias = config.get('V_init_bias', 0.0)
    
    def __call__(self, obs, neighbor_mask=None):
        """
        GNN Value network.
        
        Args:
            obs: [batch_size, D] where D = 5 (base) or 5+2*max_deg (with neighbor info)
            neighbor_mask: [batch_size, max_deg] - mask for valid neighbors (optional)
        
        Returns:
            [batch_size] - value estimates
        """
        batch_size = obs.shape[0]
        obs_dim = obs.shape[-1]
        has_neighbor_info = obs_dim > BASE_OBS_DIM
        
        # Extract features
        current_pos = obs[:, 0:2]
        pickup_pos = obs[:, 2:4]
        time = obs[:, 4:5] / self.cycle_length
        
        # Get neighbor info if available
        if has_neighbor_info:
            start_idx = BASE_OBS_DIM
            neighbor_travel_times = obs[:, start_idx:start_idx+self.max_deg]
            neighbor_distances = obs[:, start_idx+self.max_deg:start_idx+2*self.max_deg]
            if neighbor_mask is None:
                neighbor_mask = jnp.ones((batch_size, self.max_deg), dtype=bool)
        else:
            neighbor_travel_times = jnp.zeros((batch_size, self.max_deg))
            neighbor_distances = jnp.zeros((batch_size, self.max_deg))
            neighbor_mask = jnp.zeros((batch_size, self.max_deg), dtype=bool)
        
        # Initial node features
        x = jnp.concatenate([current_pos, pickup_pos, time], axis=-1)  # [batch_size, 5]
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        
        x = hk.Linear(
            self.hidden_dim,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
            b_init=hk.initializers.Constant(0.0)
        )(x)
        x = self.activation(x)
        
        # Graph convolution layers
        if has_neighbor_info:
            for gnn_layer in range(self.gnn_layers):
                max_travel = jnp.max(neighbor_travel_times, axis=-1, keepdims=True) + 1e-8
                max_dist = jnp.max(neighbor_distances, axis=-1, keepdims=True) + 1e-8
                neighbor_travel_norm = neighbor_travel_times / max_travel
                neighbor_dist_norm = neighbor_distances / max_dist
                
                neighbor_feats = jnp.stack([neighbor_travel_norm, neighbor_dist_norm], axis=-1)
                neighbor_feats_proj = hk.Linear(
                    self.hidden_dim,
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                    b_init=hk.initializers.Constant(0.0)
                )(neighbor_feats)
                
                if self.use_edge_features:
                    edge_weights = 1.0 / (neighbor_travel_times + 1e-8)
                    edge_weights = edge_weights * neighbor_mask
                    weights_sum = jnp.sum(edge_weights, axis=-1, keepdims=True) + 1e-8
                    edge_weights = edge_weights / weights_sum
                    edge_weights = edge_weights[..., None]
                else:
                    num_valid = jnp.sum(neighbor_mask, axis=-1, keepdims=True) + 1e-8
                    edge_weights = (neighbor_mask / num_valid)[..., None]
                
                aggregated = jnp.sum(neighbor_feats_proj * edge_weights, axis=1)
                x = x + aggregated
                
                residual = x
                x = hk.Linear(
                    self.hidden_dim,
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                    b_init=hk.initializers.Constant(0.0)
                )(x)
                x = self.activation(x)
                x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
                x = x + residual
        
        # Additional MLP layers
        remaining_layers = max(0, self.num_layers - (self.gnn_layers if has_neighbor_info else 0))
        for i in range(remaining_layers):
            residual = x if x.shape[-1] == self.hidden_dim else None
            x = hk.Linear(
                self.hidden_dim,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                b_init=hk.initializers.Constant(0.0)
            )(x)
            x = self.activation(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            if residual is not None:
                x = x + residual
        
        return hk.Linear(
            1,
            w_init=hk.initializers.VarianceScaling(0.1, "fan_in", "truncated_normal"),
            b_init=hk.initializers.Constant(self.init_bias)
        )(x).squeeze(-1)

