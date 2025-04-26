import jax
import jax.numpy as jnp
import jax.random as jrandom 
import equinox as eqx
import numpy as np
import random
from jax import debug
from typing import NamedTuple, Sequence, Tuple

class TaxiState(NamedTuple):
    current_node: int
    pickup_node: int
    # ride_phase: int  # 0 = to pickup, 1 = to dropoff
    done: bool
    step_count: int = 0
    neighbor_mask: jnp.ndarray = None  # [batch, max_degree]

@jax.jit
def init_env(rng_key,
             fixed_starts: Sequence[int],
             fixed_pickups: Sequence[int],
             neighbor_mask_static: jnp.ndarray,
             ) -> Tuple[TaxiState, jnp.ndarray]:
    """
    Environment reset: returns (initial_state, new_key), including neighbor_mask.
    """
    starts = jnp.asarray(fixed_starts, dtype=jnp.int32)
    pickups = jnp.asarray(fixed_pickups, dtype=jnp.int32)

    k1, k2, new_key = jrandom.split(rng_key, 3)
    taxi_idx = jrandom.randint(k1, (), 0, starts.shape[0])
    pickup_idx = jrandom.randint(k2, (), 0, pickups.shape[0])

    current = starts[taxi_idx]
    nm = neighbor_mask_static[current]

    state = TaxiState(
        current_node=current,
        pickup_node=pickups[pickup_idx],
        done=False,
        step_count=jnp.int32(0),
        neighbor_mask=nm
    )
    return state, new_key


class JAXRideEnv(eqx.Module):
    adj_list: jnp.ndarray         # shape [num_nodes, max_deg]
    travel_times: jnp.ndarray     # shape [num_nodes, max_deg]
    neighbor_mask_static: jnp.ndarray  # [num_nodes, max_deg]
    max_deg: int
    num_nodes: int
    distances: jnp.ndarray        # shape [num_nodes, num_nodes], shortest-path lengths
    max_steps: int                     # maximum allowed steps per episode
    fixed_starts: jnp.ndarray     # 1D array of valid start-node indices
    fixed_pickups: jnp.ndarray  
    timeout_penalty: float = -50.0     # penalty if timeout before pickup

    def __init__(
        self,
        adj_list: np.ndarray,
        travel_times: np.ndarray,
        neighbor_mask_static: np.ndarray,
        fixed_starts: Sequence[int],
        fixed_pickups: Sequence[int],
        distances: np.ndarray,
        max_steps: int,
        timeout_penalty: float = -50.0
    ):
        # Initialize environment arrays
        self.adj_list = jnp.array(adj_list)
        self.travel_times = jnp.array(travel_times)
        # Use provided static neighbor mask (1 for real neighbors, 0 for padding)
        self.neighbor_mask_static = jnp.array(neighbor_mask_static)
        # Shapes
        self.num_nodes, self.max_deg = self.adj_list.shape
        self.distances = jnp.array(distances)
        self.max_steps = max_steps
        self.fixed_starts = fixed_starts
        self.fixed_pickups = fixed_pickups
        self.timeout_penalty = timeout_penalty

    @jax.jit
    def step(self, state: TaxiState, action: int) -> tuple[TaxiState, float]:
        current = state.current_node
        next_node = self.adj_list[current, action].astype(state.current_node.dtype)
        time_cost = self.travel_times[current, action]

        # debug.print("current node: {}, action: {}, next node: {}, time cost: {}, pickup node: {}", current[0], action[0], next_node[0], time_cost[0], state.pickup_node[0])

        # invalid move if no neighbor
        invalid = (next_node == -1)
        # ride terminates on invalid or on reaching the pickup node
        reach_pickup = (next_node == state.pickup_node)
        # next_phase = jnp.where(reach_pickup, 1, state.ride_phase)

        # increment step count
        next_step = state.step_count + 1
        # check timeout
        timeout = (next_step >= self.max_steps) & (~reach_pickup)
        done = invalid | reach_pickup | timeout
        # debug.print("done: {}, step: {}, reached pickup: {}, invalid: {}, timeout: {}", done[0], state.step_count[0], reach_pickup[0], invalid[0], timeout[0])

        # base reward: penalize time, heavy penalty for invalid
        base_reward = jnp.where(invalid, -1000.0, -time_cost/60)

        # reward shaping: encourage moving towards pickup
        # using precomputed shortest-path distances
        dist_current = self.distances[state.current_node, state.pickup_node]
        dist_next = self.distances[next_node, state.pickup_node]
        shaping = dist_current - dist_next

        # bonus for actually reaching the pickup
        pickup_bonus = jnp.where(reach_pickup, 1000.0, 0.0)
        # penalty for timeout
        timeout_penalty = jnp.where(timeout, self.timeout_penalty, 0.0)

        reward = base_reward + shaping*10 + pickup_bonus + timeout_penalty
        # debug.print("reward: {}, base: {}, shaping: {}, pickup_bonus: {}, timeout_penalty: {}", reward[0], base_reward[0], shaping[0], pickup_bonus[0], timeout_penalty[0])

        nm = self.neighbor_mask_static[next_node]

        new_state = TaxiState(
            current_node=next_node,
            pickup_node=state.pickup_node,
            # ride_phase=next_phase,
            step_count=jnp.where(done, 0, next_step),
            done=done,
            neighbor_mask=nm
        )
        return new_state, reward, reach_pickup

    def reset(self, rng_key) -> Tuple[TaxiState, jnp.ndarray]:
        # Reset environment and sample initial state with mask
        state, new_key = init_env(
            rng_key,
            self.fixed_starts,
            self.fixed_pickups,
            self.neighbor_mask_static,
            self.distances
        )
        return state, new_key