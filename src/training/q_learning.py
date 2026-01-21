"""
Tabular Q-Learning implementation for discrete state-action spaces.
Used when args.discrete=True to discretize time with dt=1.

This implementation uses dense JAX arrays for efficient computation and JIT compilation.
"""
import jax
import jax.numpy as jnp
from jax import random as jax_random
from typing import Dict, Tuple, Optional
import numpy as np
import pickle
from pathlib import Path

from taxi_env import TaxiState, TaxiEnv, init_env


@jax.jit
def _jitted_step_with_discretization(
    env: TaxiEnv,
    state: TaxiState,
    action: jnp.int32,  # Changed from int to jnp.int32 to avoid type conversion
    discrete_travel_times: jnp.ndarray,
    dt: float,
):
    """Compiled environment step matching TabularQLearning.step_with_discretization."""
    already_done = state.done

    curr = state.current_node
    nxt = env.adj_list[curr, action]

    travel = discrete_travel_times[curr, action]

    reach = (curr == state.pickup_node) | (nxt == state.pickup_node)
    step_n = state.step_count + 1
    done = reach

    t1 = state.time + travel

    period_edge = env.periods[curr, action]
    offset_edge = env.offsets[curr, action]
    green_edge = env.green_durations[curr, action]

    cycle = (t1 + offset_edge) % period_edge
    wait = jnp.where(cycle < green_edge, 0.0, period_edge - cycle)
    t2 = t1 + wait

    discrete_t2 = jnp.round(t2 / dt) * dt
    norm_time = jnp.mod(discrete_t2, period_edge)

    total_delay = travel + wait
    dist_curr = env.distances[curr, state.pickup_node]
    dist_next = env.distances[nxt, state.pickup_node]
    dist_diff = dist_curr - dist_next
    dist_shaping = dist_diff / 60.0

    pickup_bonus = jnp.where(reach, env.pickup_bonus, 0.0)
    reward = -total_delay / 60.0 + pickup_bonus + dist_shaping
    reward = jnp.where(already_done, 0.0, reward)

    nm = env.neighbor_mask_static[nxt]

    at_pickup_before = (curr == state.pickup_node)
    reached_pickup_this_step = reach & ~already_done
    final_node = jnp.where(
        already_done,
        curr,
        jnp.where(
            reached_pickup_this_step,
            jnp.where(at_pickup_before, curr, nxt),
            nxt,
        ),
    )
    final_neighbor_mask = jnp.where(already_done, state.neighbor_mask, nm)
    final_time = jnp.where(already_done, state.time, norm_time)
    final_step_count = jnp.where(already_done, state.step_count, jnp.where(done, 0, step_n))
    final_done = jnp.where(already_done, True, done)

    final_wait = jnp.where(already_done, 0.0, wait)
    final_travel = jnp.where(already_done, 0.0, travel)

    new_state = TaxiState(
        current_node=final_node,
        pickup_node=state.pickup_node,
        done=final_done,
        step_count=final_step_count,
        neighbor_mask=final_neighbor_mask,
        time=final_time,
    )

    return new_state, reward, final_done, {"wait": final_wait, "travel": final_travel}


def discretize_time(time: float, dt: float = 1.0) -> int:
    """Discretize time to the nearest multiple of dt."""
    return int(round(time / dt))


def discretize_travel_time(travel_time: float, dt: float = 1.0) -> float:
    """Discretize travel time to the nearest multiple of dt."""
    return dt * round(travel_time / dt)


def state_to_discrete_key(state: TaxiState, dt: float = 1.0) -> Tuple[int, int, int]:
    """
    Convert continuous state to discrete state key.
    Returns: (current_node, pickup_node, discrete_time)
    """
    discrete_time = discretize_time(float(state.time), dt)
    return (int(state.current_node), int(state.pickup_node), discrete_time)


def _state_index_jax(state: TaxiState, dt: float, max_time_slices: int, num_nodes: int) -> Tuple[int, int, int]:
    """
    JAX-compatible state indexing function.
    Returns: (current_node, pickup_node, time_idx)
    """
    curr = int(state.current_node)
    pickup = int(state.pickup_node)
    t_idx = discretize_time(float(state.time), dt)
    # Clamp indices to valid ranges
    t_idx = max(0, min(max_time_slices - 1, t_idx))
    curr = max(0, min(num_nodes - 1, curr))
    pickup = max(0, min(num_nodes - 1, pickup))
    return curr, pickup, t_idx


@jax.jit
def _get_state_indices_jit(state: TaxiState, dt: float, max_time_slices: int, num_nodes: int) -> Tuple[jnp.int32, jnp.int32, jnp.int32]:
    """
    JIT-compiled state indexing function.
    Returns: (current_node, pickup_node, time_idx) as JAX arrays
    """
    curr = jnp.int32(state.current_node)
    pickup = jnp.int32(state.pickup_node)
    t_idx = jnp.int32(jnp.round(state.time / dt))
    # Clamp indices to valid ranges
    t_idx = jnp.clip(t_idx, 0, max_time_slices - 1)
    curr = jnp.clip(curr, 0, num_nodes - 1)
    pickup = jnp.clip(pickup, 0, num_nodes - 1)
    return curr, pickup, t_idx


@jax.jit
def _get_epsilon_jit(step: jnp.int32, epsilon_start: float, epsilon_end: float, epsilon_decay_steps: int) -> jnp.float32:
    """JIT-compiled epsilon calculation."""
    step_f = jnp.float32(step)
    decay_steps_f = jnp.float32(epsilon_decay_steps)
    progress = jnp.clip(step_f / decay_steps_f, 0.0, 1.0)
    return jnp.float32(epsilon_start * (1.0 - progress) + epsilon_end * progress)


@jax.jit
def _select_action_jit(
    q_table: jnp.ndarray,
    state: TaxiState,
    step: jnp.int32,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    dt: float,
    max_time_slices: int,
    num_nodes: int,
    key: jnp.ndarray,
) -> Tuple[jnp.int32, jnp.ndarray]:
    """
    JIT-compiled epsilon-greedy action selection.
    Returns: (action, new_key)
    """
    curr, pickup, t_idx = _get_state_indices_jit(state, dt, max_time_slices, num_nodes)
    
    # Get Q-values for all actions
    q_row = q_table[curr, pickup, t_idx, :]  # [max_deg]
    
    # Mask invalid actions
    # state.neighbor_mask is already jnp.ndarray with dtype jnp.bool_, so use directly
    valid_mask = state.neighbor_mask
    q_masked = jnp.where(valid_mask, q_row, -1e9)
    
    # Calculate epsilon
    epsilon = _get_epsilon_jit(step, epsilon_start, epsilon_end, epsilon_decay_steps)
    
    # Epsilon-greedy
    key, subkey = jax_random.split(key)
    rand_val = jax_random.uniform(subkey)
    
    # Greedy action
    best_action = jnp.argmax(q_masked)
    
    # For random action: sample uniformly from all actions, then mask
    # This is simpler than finding valid actions, and we'll fallback to greedy if invalid
    key, subkey = jax_random.split(key)
    max_deg = q_table.shape[-1]
    random_action = jax_random.randint(subkey, (), 0, max_deg, dtype=jnp.int32)
    
    # Choose between greedy and random based on epsilon
    # If random action is invalid, fallback to greedy
    action = jnp.where(
        rand_val < epsilon,
        jnp.where(valid_mask[random_action], random_action, best_action),
        best_action
    )
    
    return action, key


@jax.jit
def _update_q_value_jit(
    q_table: jnp.ndarray,
    state: TaxiState,
    action: jnp.int32,
    reward: jnp.float32,
    next_state: TaxiState,
    done: jnp.bool_,
    learning_rate: float,
    gamma: float,
    dt: float,
    max_time_slices: int,
    num_nodes: int,
) -> jnp.ndarray:
    """
    JIT-compiled Q-value update.
    Returns: updated Q-table
    """
    curr, pickup, t_idx = _get_state_indices_jit(state, dt, max_time_slices, num_nodes)
    current_q = q_table[curr, pickup, t_idx, action]
    
    # Compute target
    n_curr, n_pickup, n_t_idx = _get_state_indices_jit(next_state, dt, max_time_slices, num_nodes)
    q_next_row = q_table[n_curr, n_pickup, n_t_idx, :]  # [max_deg]
    
    # Mask invalid actions for next state
    # next_state.neighbor_mask is already jnp.ndarray with dtype jnp.bool_, so use directly
    valid_mask = next_state.neighbor_mask
    q_next_masked = jnp.where(valid_mask, q_next_row, -1e9)
    max_next_q = jnp.max(q_next_masked)
    
    target = jnp.where(
        done,
        reward,  # Terminal state
        reward + gamma * max_next_q
    )
    
    # Q-learning update
    new_q = current_q + learning_rate * (target - current_q)
    
    # Update Q-table
    return q_table.at[curr, pickup, t_idx, action].set(new_q)


@jax.jit
def _estimate_return_q_table_direct(
    q_table: jnp.ndarray,
    start: jnp.int32,
    pickup: jnp.int32,
    num_nodes: int,
    env: TaxiEnv,
) -> jnp.float32:
    """
    JIT-compiled function to estimate return for a start-pickup pair using Q-table directly.
    Much faster than rollouts - just uses max Q-value at initial state.
    Returns: estimated return (negative cost for matching)
    """
    # Get state indices for initial state (time=0)
    curr = jnp.clip(jnp.int32(start), 0, num_nodes - 1)
    pickup_idx = jnp.clip(jnp.int32(pickup), 0, num_nodes - 1)
    t_idx = jnp.int32(0)  # Start at time 0
    
    # Get Q-values for all actions at this state
    q_row = q_table[curr, pickup_idx, t_idx, :]  # [max_deg]
    
    # Mask invalid actions
    valid_mask = env.neighbor_mask_static[start]
    q_masked = jnp.where(valid_mask, q_row, -jnp.inf)
    
    # Return max Q-value (expected return from best action)
    return jnp.max(q_masked)


# Batched version for multiple start-pickup pairs
_estimate_returns_batch_q_table_direct = jax.vmap(
    _estimate_return_q_table_direct,
    in_axes=(None, 0, 0, None, None),
    out_axes=0
)


# Batched versions for multi-agent training
_batched_select_action = jax.vmap(
    _select_action_jit,
    in_axes=(None, 0, 0, None, None, None, None, None, None, 0),
    out_axes=(0, 0)
)


@jax.jit
def _batched_update_q_values_vectorized(
    q_table: jnp.ndarray,
    states_current_node: jnp.ndarray,
    states_pickup_node: jnp.ndarray,
    states_time: jnp.ndarray,
    states_neighbor_mask: jnp.ndarray,
    states_done: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_states_current_node: jnp.ndarray,
    next_states_pickup_node: jnp.ndarray,
    next_states_time: jnp.ndarray,
    next_states_neighbor_mask: jnp.ndarray,
    next_states_done: jnp.ndarray,
    dones: jnp.ndarray,
    learning_rate: float,
    gamma: float,
    dt: float,
    max_time_slices: int,
    num_nodes: int,
) -> jnp.ndarray:
    """
    Update Q-table for multiple agents
    """    
    # Compute state indices for all agents
    def get_indices(curr, pickup, time):
        curr_clipped = jnp.clip(curr, 0, num_nodes - 1)
        pickup_clipped = jnp.clip(pickup, 0, num_nodes - 1)
        t_idx = jnp.clip(jnp.int32(jnp.round(time / dt)), 0, max_time_slices - 1)
        return curr_clipped, pickup_clipped, t_idx
    
    # Vectorized index computation
    curr_indices, pickup_indices, t_indices = jax.vmap(get_indices)(
        states_current_node, states_pickup_node, states_time
    )
    next_curr_indices, next_pickup_indices, next_t_indices = jax.vmap(get_indices)(
        next_states_current_node, next_states_pickup_node, next_states_time
    )
    
    # Get current Q-values for all agents [num_agents]
    current_q = q_table[curr_indices, pickup_indices, t_indices, actions]
    
    # Get next state Q-values for all agents [num_agents, max_deg]
    q_next_rows = q_table[next_curr_indices, next_pickup_indices, next_t_indices, :]
    
    # Mask invalid actions for next states [num_agents, max_deg]
    # next_states_neighbor_mask is already jnp.ndarray with dtype jnp.bool_, so use directly
    valid_masks = next_states_neighbor_mask  # [num_agents, max_deg]
    q_next_masked = jnp.where(valid_masks, q_next_rows, -1e9)
    max_next_q = jnp.max(q_next_masked, axis=-1)  # [num_agents]
    
    # Compute targets [num_agents]
    targets = jnp.where(
        dones,
        rewards,  # Terminal state
        rewards + gamma * max_next_q
    )
    
    # Q-learning updates [num_agents]
    new_q_values = current_q + learning_rate * (targets - current_q)
    
    # Mask updates for done agents (keep old values)
    should_update = ~states_done  # [num_agents]
    final_q_values = jnp.where(should_update, new_q_values, current_q)
    
    # Apply all updates at once using vectorized indexing
    return q_table.at[curr_indices, pickup_indices, t_indices, actions].set(final_q_values)




class TabularQLearning:
    """
    Tabular Q-Learning agent for discrete state-action spaces.
    
    Uses dense JAX arrays for efficient computation and JIT compilation.
    Q-table shape: [num_nodes, num_nodes, max_time_slices, max_deg]
    """
    
    def __init__(
        self,
        env: TaxiEnv,
        dt: float = 1.0,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10000,
        initial_q_value: float = 0.0,
        max_time_slices: int = 100,
    ):
        self.env = env
        self.dt = dt
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.initial_q_value = initial_q_value
        self.max_time_slices = max_time_slices
        
        # Dense Q-table: [num_nodes, num_nodes, max_time_slices, max_deg]
        q_shape = (env.num_nodes, env.num_nodes, max_time_slices, env.max_deg)
        self.q_table = jnp.full(q_shape, initial_q_value, dtype=jnp.float32)
        
        # Track visits for statistics (sparse dict for compatibility)
        # Disabled by default for performance - set track_visits=True to enable
        self.track_visits = False  # Set to True only if statistics are needed
        self.visit_counts: Dict[Tuple[int, int, int, int], int] = {}
        
        # Discretize travel times in environment
        self._discretize_travel_times()
    
    def _discretize_travel_times(self):
        """Discretize all travel times in the environment to multiples of dt."""
        # Create a discretized copy of travel_times
        self.discrete_travel_times = jnp.round(self.env.travel_times / self.dt) * self.dt
    
    def _get_state_indices(self, state: TaxiState) -> Tuple[int, int, int]:
        """Get Q-table indices for a state."""
        return _state_index_jax(state, self.dt, self.max_time_slices, self.env.num_nodes)
    
    def initialize_q_values_from_shortest_paths(
        self,
        use_all_time_slices: bool = False,
        max_time_slices: int = 100,
    ):
        """
        Initialize Q-table values based on shortest path travel times.
        
        This provides a good initialization by computing Q-values based on:
        - Travel time for the action (negative cost)
        - Distance shaping (progress toward pickup)
        - Estimated future value (remaining distance to pickup)
        - Pickup bonus if action reaches pickup
        
        Note: This initialization ignores time discretization and uses travel times
        directly from the environment. It initializes Q-values for time=0 (or all
        time slices if use_all_time_slices=True).
        
        Args:
            use_all_time_slices: If True, initialize for all time slices up to max_time_slices.
                                 If False, only initialize for time=0.
            max_time_slices: Maximum number of time slices to initialize (if use_all_time_slices=True).
        """
        print(f"\nInitializing Q-table from shortest path travel times...")
        print(f"  Using all time slices: {use_all_time_slices}")
        if use_all_time_slices:
            print(f"  Max time slices: {max_time_slices}")
        
        reward_scaling = 1.0  # Match the reward scaling in step_with_discretization
        
        num_nodes = self.env.num_nodes
        max_deg = self.env.max_deg
        
        # Vectorized Q-value computation using JAX operations
        # Note: use_all_time_slices must be static for JIT compilation
        def compute_q_values_vectorized(
            q_table: jnp.ndarray,
            adj_list: jnp.ndarray,
            travel_times: jnp.ndarray,
            distances: jnp.ndarray,
            neighbor_mask: jnp.ndarray,
            reward_scaling: float,
            gamma: float,
            pickup_bonus: float,
            initial_q_value: float,
            use_all_time_slices: bool,
            max_time_slices: int,
        ) -> jnp.ndarray:
            """
            Vectorized computation of Q-values for all state-action pairs.
            Returns updated Q-table.
            """
            # Get dimensions from q_table shape: [num_nodes, num_nodes, max_time_slices, max_deg]
            num_nodes = q_table.shape[0]
            max_deg = q_table.shape[3]
            
            # Create meshgrids for all combinations
            # Shape: [num_nodes, num_nodes, max_deg]
            curr_indices = jnp.arange(num_nodes)[:, None, None]  # [num_nodes, 1, 1]
            pickup_indices = jnp.arange(num_nodes)[None, :, None]  # [1, num_nodes, 1]
            action_indices = jnp.arange(max_deg)[None, None, :]  # [1, 1, max_deg]
            
            # Broadcast to [num_nodes, num_nodes, max_deg]
            curr_broadcast = jnp.broadcast_to(curr_indices, (num_nodes, num_nodes, max_deg))
            pickup_broadcast = jnp.broadcast_to(pickup_indices, (num_nodes, num_nodes, max_deg))
            action_broadcast = jnp.broadcast_to(action_indices, (num_nodes, num_nodes, max_deg))
            
            # Get next nodes for all (curr, action) pairs
            # adj_list[curr, action] -> shape [num_nodes, num_nodes, max_deg]
            next_nodes = adj_list[curr_broadcast, action_broadcast]  # [num_nodes, num_nodes, max_deg]
            
            # Get travel times
            travel_times_broadcast = travel_times[curr_broadcast, action_broadcast]  # [num_nodes, num_nodes, max_deg]
            
            # Get distances
            dist_curr = distances[curr_broadcast, pickup_broadcast]  # [num_nodes, num_nodes, max_deg]
            dist_next = distances[next_nodes, pickup_broadcast]  # [num_nodes, num_nodes, max_deg]
            
            # Compute reward components (vectorized)
            travel_cost = -reward_scaling * travel_times_broadcast / 60.0
            dist_diff = dist_curr - dist_next
            dist_shaping = dist_diff / 60.0
            reaches_pickup = (next_nodes == pickup_broadcast)
            pickup_bonus_broadcast = jnp.where(reaches_pickup, pickup_bonus, 0.0)
            
            # Estimated future value
            remaining_dist = distances[next_nodes, pickup_broadcast]
            future_value = jnp.where(
                reaches_pickup,
                0.0,
                -gamma * remaining_dist / 60.0
            )
            
            # Compute Q-values
            q_values = travel_cost + dist_shaping + pickup_bonus_broadcast + future_value
            
            # Mask invalid entries:
            # 1. Skip if curr == pickup
            same_node_mask = (curr_broadcast != pickup_broadcast)
            
            # 2. Skip if action is invalid
            valid_action_mask = neighbor_mask[curr_broadcast, action_broadcast]
            
            # 3. Skip if next_node is invalid (adj_list returns -1 for invalid)
            valid_next_node_mask = (next_nodes >= 0)
            
            # Combined mask
            valid_mask = same_node_mask & valid_action_mask & valid_next_node_mask
            
            # Update Q-table - only update valid entries, leave invalid entries unchanged (they remain at initial_q_value)
            if use_all_time_slices:
                # Broadcast q_values to all time slices: [num_nodes, num_nodes, max_time_slices, max_deg]
                q_values_expanded = jnp.broadcast_to(
                    q_values[:, :, None, :],  # [num_nodes, num_nodes, 1, max_deg]
                    (num_nodes, num_nodes, max_time_slices, max_deg)
                )
                # Update all time slices at once - only update valid entries
                valid_mask_expanded = jnp.broadcast_to(
                    valid_mask[:, :, None, :], 
                    (num_nodes, num_nodes, max_time_slices, max_deg)
                )
                q_table = q_table.at[:, :, :, :].set(
                    jnp.where(valid_mask_expanded, q_values_expanded, q_table)
                )
            else:
                # Only update time slice 0 - only update valid entries
                q_table = q_table.at[:, :, 0, :].set(
                    jnp.where(valid_mask, q_values, q_table[:, :, 0, :])
                )
            
            return q_table
        
        # JIT-compile with use_all_time_slices as static argument
        compute_q_values_vectorized_jit = jax.jit(
            compute_q_values_vectorized,
            static_argnums=(9, 10)  # use_all_time_slices and max_time_slices are static
        )
        
        # Compute Q-values vectorized
        self.q_table = compute_q_values_vectorized_jit(
            self.q_table,
            self.env.adj_list,
            self.env.travel_times,
            self.env.distances,
            self.env.neighbor_mask_static,
            reward_scaling,
            self.gamma,
            self.env.pickup_bonus,
            self.initial_q_value,
            use_all_time_slices,
            max_time_slices,
        )
        
        # Count initialized values (only valid actions) - vectorized
        # Count valid (curr, pickup, action) triplets where curr != pickup and action is valid
        same_node_mask = jnp.arange(num_nodes)[:, None] != jnp.arange(num_nodes)[None, :]  # [num_nodes, num_nodes]
        num_valid_curr_pickup_pairs = int(same_node_mask.sum())
        
        # Count valid actions per node
        valid_actions_per_node = self.env.neighbor_mask_static.sum(axis=1)  # [num_nodes]
        # For each (curr, pickup) pair, count valid actions for curr
        # We need to sum over all curr nodes, but only for valid (curr, pickup) pairs
        # This is approximately: num_valid_pairs * avg_valid_actions_per_node
        # More accurately: sum over curr of (num_valid_pickups_for_curr * num_valid_actions_for_curr)
        num_valid_pickups_per_curr = same_node_mask.sum(axis=1)  # [num_nodes] - number of valid pickups for each curr
        num_valid_actions = int((num_valid_pickups_per_curr * valid_actions_per_node).sum())
        
        if use_all_time_slices:
            num_initialized = num_valid_actions * max_time_slices
        else:
            num_initialized = num_valid_actions
        
        print(f"  Initialized Q-values for {num_initialized} state-action pairs")
        
        # Update statistics
        stats = self.get_statistics()
        print(f"  Avg Q-value: {stats['avg_q_value']:.4f}")
        print(f"  Min Q-value: {stats['min_q_value']:.4f}")
        print(f"  Max Q-value: {stats['max_q_value']:.4f}")
        print("Q-table initialization completed!\n")
    
    def get_epsilon(self, step: int) -> float:
        """Get epsilon value for epsilon-greedy policy at current step."""
        if step >= self.epsilon_decay_steps:
            return self.epsilon_end
        progress = step / self.epsilon_decay_steps
        return self.epsilon_start * (1.0 - progress) + self.epsilon_end * progress
    
    def get_q_value(self, state: TaxiState, action: int) -> float:
        """Get Q-value for a state-action pair."""
        curr, pickup, t_idx = self._get_state_indices(state)
        return float(self.q_table[curr, pickup, t_idx, action])
    
    def set_q_value(self, state: TaxiState, action: int, value: float):
        """Set Q-value for a state-action pair."""
        curr, pickup, t_idx = self._get_state_indices(state)
        # Use in-place update with .at[].set() - creates new array but JAX optimizes this
        self.q_table = self.q_table.at[curr, pickup, t_idx, action].set(float(value))
        # if self.track_visits:
        #     q_key = (curr, pickup, t_idx, action)
        #     self.visit_counts[q_key] = self.visit_counts.get(q_key, 0) + 1
    
    def get_action(self, state: TaxiState, step: int, key: jnp.ndarray) -> int:
        """
        Epsilon-greedy action selection (uses JIT-compiled version).
        
        Args:
            state: Current state
            step: Current training step (for epsilon decay)
            key: JAX random key
            
        Returns:
            Selected action
        """
        action, new_key = _select_action_jit(
            self.q_table,
            state,
            jnp.int32(step),
            self.epsilon_start,
            self.epsilon_end,
            self.epsilon_decay_steps,
            self.dt,
            self.max_time_slices,
            self.env.num_nodes,
            key,
        )
        # Note: new_key is returned but we don't use it here since key is passed by value
        # For proper key management, the caller should handle this
        return int(action)
    
    def update_q_value(
        self,
        state: TaxiState,
        action: int,
        reward: float,
        next_state: TaxiState,
        done: bool
    ):
        """
        Update Q-value using Q-learning update rule (uses JIT-compiled version).
        
        Q(s, a) = Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        # Use JIT-compiled update function
        self.q_table = _update_q_value_jit(
            self.q_table,
            state,
            jnp.int32(action),
            jnp.float32(reward),
            next_state,
            jnp.bool_(done),
            self.learning_rate,
            self.gamma,
            self.dt,
            self.max_time_slices,
            self.env.num_nodes,
        )
        
        # Update visit counts (not JIT-compiled, but fast)
        # Skip if tracking disabled for performance
        # if self.track_visits:
        #     curr, pickup, t_idx = self._get_state_indices(state)
        #     q_key = (curr, pickup, t_idx, action)
        #     self.visit_counts[q_key] = self.visit_counts.get(q_key, 0) + 1
    
    def estimate_returns_for_matching(
        self,
        starts: jnp.ndarray,
        pickups: jnp.ndarray,
        rollout_steps: int = 10,  # Ignored - kept for API compatibility
        key: jnp.ndarray = None,  # Ignored - kept for API compatibility
    ) -> jnp.ndarray:
        """
        Estimate returns for start-pickup pairs using Q-table directly (no rollouts needed!).
        Much faster - just uses max Q-value at initial state.
        
        Args:
            starts: Array of start node indices [num_agents]
            pickups: Array of pickup node indices [num_agents]
            rollout_steps: Ignored (kept for API compatibility)
            key: Ignored (kept for API compatibility)
            
        Returns:
            Returns matrix [num_agents, num_agents] where [i, j] is return for start[i] -> pickup[j]
        """
        num_starts = len(starts)
        num_pickups = len(pickups)
        
        # Create all combinations: [num_starts * num_pickups]
        starts_expanded = jnp.repeat(starts, num_pickups)
        pickups_expanded = jnp.tile(pickups, num_starts)
        
        # Estimate returns directly from Q-table (no rollouts!)
        returns_flat = _estimate_returns_batch_q_table_direct(
            self.q_table,
            starts_expanded,
            pickups_expanded,
            self.env.num_nodes,
            self.env,
        )
        
        # Reshape to [num_starts, num_pickups]
        returns_matrix = returns_flat.reshape(num_starts, num_pickups)
        return returns_matrix
    
    def step_with_discretization(self, state: TaxiState, action: int, key: jnp.ndarray) -> Tuple[TaxiState, float, bool, dict]:
        """
        Environment step with discretized travel times.
        This replaces the normal env.step() when using discrete mode.
        """
        next_state, reward, done_flag, info = _jitted_step_with_discretization(
            self.env,
            state,
            action,
            self.discrete_travel_times,
            self.dt,
        )

        return (
            next_state,
            float(reward),
            bool(done_flag),
            {"wait": float(info["wait"]), "travel": float(info["travel"])},
        )
    
    def get_statistics(self) -> Dict:
        """Get statistics about the Q-table."""
        # Compute statistics on the entire Q-table
        q_values = self.q_table.flatten()
        
        # Compute Q-value statistics (on entire table, including initialized values)
        avg_q = float(jnp.mean(q_values))
        min_q = float(jnp.min(q_values))
        max_q = float(jnp.max(q_values))

        # TEMPORARY: for signature matching
        num_visited_states = 0
        avg_visits = 0.0
        max_visits = 0
        
        # # Only compute visit statistics if tracking is enabled (expensive for large dicts)
        # if self.track_visits and self.visit_counts:
        #     visit_counts_list = list(self.visit_counts.values())
        #     num_visited_states = len(self.visit_counts)
        #     avg_visits = float(np.mean(visit_counts_list)) if visit_counts_list else 0.0
        #     max_visits = int(np.max(visit_counts_list)) if visit_counts_list else 0
        # else:
        #     # Estimate visited states by counting non-initial values (much faster)
        #     # This is approximate but avoids expensive dict operations
        #     non_initial_mask = (q_values != self.initial_q_value)
        #     num_visited_states = int(jnp.sum(non_initial_mask))
        #     avg_visits = 0.0
        #     max_visits = 0
        
        return {
            'num_states': num_visited_states,
            'avg_q_value': avg_q,
            'min_q_value': min_q,
            'max_q_value': max_q,
            'avg_visits': avg_visits,
            'max_visits': max_visits,
        }

    def state_dict(self) -> Dict:
        """Return a serializable snapshot of the Q-table and metadata."""
        return {
            "dt": self.dt,
            "learning_rate": self.learning_rate,
            "discount_factor": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "initial_q_value": self.initial_q_value,
            "max_time_slices": self.max_time_slices,
            "q_table": np.array(self.q_table),  # Convert JAX array to numpy for serialization
            "visit_counts": dict(self.visit_counts),
        }

    def save(self, path: str) -> None:
        """Serialize the agent to disk."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("wb") as f:
            pickle.dump(self.state_dict(), f)

    @classmethod
    def load(cls, env: TaxiEnv, path: str):
        """Load agent state from disk."""
        with Path(path).open("rb") as f:
            state = pickle.load(f)

        max_time_slices = state.get("max_time_slices", 100)
        agent = cls(
            env=env,
            dt=state.get("dt", 1.0),
            learning_rate=state.get("learning_rate", 0.1),
            discount_factor=state.get("discount_factor", 0.99),
            epsilon_start=state.get("epsilon_start", 1.0),
            epsilon_end=state.get("epsilon_end", 0.01),
            epsilon_decay_steps=state.get("epsilon_decay_steps", 10000),
            initial_q_value=state.get("initial_q_value", 0.0),
            max_time_slices=max_time_slices,
        )

        # Load Q-table (convert numpy back to JAX array)
        if "q_table" in state:
            if isinstance(state["q_table"], dict):
                # Legacy format: convert dict to dense array
                for (curr, pickup, t_idx, action), value in state["q_table"].items():
                    agent.q_table = agent.q_table.at[curr, pickup, t_idx, action].set(value)
            else:
                # New format: numpy array
                agent.q_table = jnp.array(state["q_table"])
        
        # Load visit counts
        if "visit_counts" in state:
            agent.visit_counts.update(state["visit_counts"])
        
        return agent
    
    def load_q_table_from_file(self, path: str) -> None:
        """
        Load Q-table from a saved file for initialization.
        This allows initializing the Q-table from a previously saved Q-table
        while keeping the current agent's hyperparameters.
        
        Args:
            path: Path to the saved Q-table pickle file
        """
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        
        # Load Q-table (convert numpy back to JAX array)
        if "q_table" in state:
            loaded_q_table = state["q_table"]
            
            # Handle different formats
            if isinstance(loaded_q_table, dict):
                # Legacy format: convert dict to dense array efficiently using vectorized operations
                print(f"Converting dictionary format to dense array (this may take a moment for large Q-tables)...")
                print(f"  Dictionary has {len(loaded_q_table)} entries")
                
                # Get the shape of the current Q-table
                q_shape = self.q_table.shape
                
                # Convert current JAX array to numpy for efficient indexing
                q_table_np = np.array(self.q_table)
                
                # Extract keys and values from dictionary
                # Use vectorized numpy operations for much faster updates
                # Unpack keys and values in one pass for efficiency
                items = list(loaded_q_table.items())
                
                if len(items) > 0:
                    # Unpack keys and values directly
                    # Keys are tuples of (curr, pickup, t_idx, action)
                    keys_list, values_list = zip(*items)
                    
                    # Convert to numpy arrays for vectorized indexing
                    # Unpack keys into separate arrays for each dimension
                    curr_indices = np.array([k[0] for k in keys_list], dtype=np.int32)
                    pickup_indices = np.array([k[1] for k in keys_list], dtype=np.int32)
                    t_idx_indices = np.array([k[2] for k in keys_list], dtype=np.int32)
                    action_indices = np.array([k[3] for k in keys_list], dtype=np.int32)
                    q_values = np.array(values_list, dtype=np.float32)
                    
                    # Filter out indices that are out of bounds
                    valid_mask = (
                        (curr_indices >= 0) & (curr_indices < q_shape[0]) &
                        (pickup_indices >= 0) & (pickup_indices < q_shape[1]) &
                        (t_idx_indices >= 0) & (t_idx_indices < q_shape[2]) &
                        (action_indices >= 0) & (action_indices < q_shape[3])
                    )
                    
                    if np.any(~valid_mask):
                        num_invalid = np.sum(~valid_mask)
                        print(f"  Warning: {num_invalid:,} entries out of bounds, skipping them...")
                    
                    # Use advanced indexing to update all valid entries at once
                    # This is much faster than updating one by one
                    valid_curr = curr_indices[valid_mask]
                    valid_pickup = pickup_indices[valid_mask]
                    valid_t_idx = t_idx_indices[valid_mask]
                    valid_action = action_indices[valid_mask]
                    valid_values = q_values[valid_mask]
                    
                    # Vectorized update: update all entries in one operation
                    q_table_np[valid_curr, valid_pickup, valid_t_idx, valid_action] = valid_values
                    
                    print(f"  Updated {np.sum(valid_mask):,} / {len(items):,} entries using vectorized operations")
                
                # Convert back to JAX array in one operation
                self.q_table = jnp.array(q_table_np)
                print(f"  Conversion complete!")
            else:
                # New format: numpy array
                # Convert to numpy first if needed (more efficient for operations)
                if not isinstance(loaded_q_table, np.ndarray):
                    loaded_q_table = np.array(loaded_q_table)
                
                # Check if shapes match
                if loaded_q_table.shape != self.q_table.shape:
                    print(f"Warning: Q-table shape mismatch!")
                    print(f"  Loaded Q-table shape: {loaded_q_table.shape}")
                    print(f"  Current Q-table shape: {self.q_table.shape}")
                    
                    # Critical: max_deg (last dimension) must match environment's max_deg
                    loaded_max_deg = loaded_q_table.shape[3]
                    env_max_deg = self.q_table.shape[3]
                    
                    if loaded_max_deg != env_max_deg:
                        raise ValueError(
                            f"Q-table max_deg mismatch: loaded Q-table has max_deg={loaded_max_deg}, "
                            f"but current environment has max_deg={env_max_deg}. "
                            f"This mismatch will cause broadcasting errors. "
                            f"Please use a Q-table created with the same graph structure."
                        )
                    
                    print(f"  Attempting to copy compatible entries...")
                    
                    # Convert current Q-table to numpy for efficient slice operations
                    q_table_np = np.array(self.q_table)
                    
                    # Copy compatible entries (up to minimum dimensions) using numpy
                    min_shape = tuple(min(s1, s2) for s1, s2 in zip(loaded_q_table.shape, self.q_table.shape))
                    q_table_np[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]] = \
                        loaded_q_table[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
                    
                    # Convert back to JAX array in one operation
                    self.q_table = jnp.array(q_table_np)
                else:
                    # Shapes match, convert to JAX array directly
                    self.q_table = jnp.array(loaded_q_table)
            
            print(f"Successfully loaded Q-table from {path}")
            print(f"  Q-table shape: {self.q_table.shape}")
            stats = self.get_statistics()
            print(f"  Avg Q-value: {stats['avg_q_value']:.4f}")
            print(f"  Min Q-value: {stats['min_q_value']:.4f}")
            print(f"  Max Q-value: {stats['max_q_value']:.4f}")
        else:
            raise ValueError(f"No Q-table found in saved file: {path}")

