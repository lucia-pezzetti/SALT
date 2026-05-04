from dataclasses import dataclass
from typing import Any, Dict
import jax.numpy as jnp


@dataclass
class RunContext:
    """Shared references needed by mode handlers to avoid duplicate setup."""
    # Environment and graph
    env: Any
    G: Any
    node_to_idx: Dict[Any, int]
    idx_to_node: Dict[int, Any]

    # Observation builders
    obs_fn_single: Any
    obs_fn_batch: Any

    # Fixed sets and matrices
    fixed_starts_idx: jnp.ndarray
    fixed_pickups_idx: jnp.ndarray
    distances: jnp.ndarray
    hop_distances: jnp.ndarray
    neighbor_mask_static: jnp.ndarray

    # Precomputed helpers
    estimate_state: Any
    max_length: int
    paths_dict: Dict[Any, Any]


