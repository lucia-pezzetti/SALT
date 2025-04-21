import jax
import jax.numpy as jnp
import jax.random as jrandom 
import equinox as eqx
import numpy as np
import random
from jax import debug
from typing import NamedTuple

class TaxiState(NamedTuple):
    current_node: int
    pickup_node: int
    ride_phase: int  # 0 = to pickup, 1 = to dropoff
    done: bool
    step_count: int = 0

def init_env(rng_key, fixed_starts, fixed_pickups, distances):
    """
    Initializes a taxi environment state using JAX randomness.
    Returns a tuple of (TaxiState, new_rng_key).
    """
    def single_reset(key):
        def body_fun(val):
            key, _, _, _ = val  # correctly unpack the full 4-tuple
            key, subkey1, subkey2 = jrandom.split(key, 3)

            starts = jnp.array(fixed_starts, dtype=jnp.int64)
            pickups = jnp.array(fixed_pickups, dtype=jnp.int64)
            taxi_idx = jrandom.randint(subkey1, (), 0, starts.shape[0])
            pickup_idx = jrandom.randint(subkey2, (), 0, pickups.shape[0])
            taxi_node = starts[taxi_idx]
            pickup_node = pickups[pickup_idx]

            valid = (taxi_node != pickup_node) & (distances[taxi_node, pickup_node] > 0)
            return key, taxi_node, pickup_node, valid

        def cond_fun(val):
            _, _, _, valid = val
            return ~valid

        # initialize dummy values for taxi_node and pickup_node
        dummy_node = jnp.array(0, dtype=jnp.int64)
        init_valid = jnp.array(False)

        key, taxi_node, pickup_node, _ = jax.lax.while_loop(
            cond_fun,
            body_fun,
            (key, dummy_node, dummy_node, init_valid)
        )

        return TaxiState(
            current_node=taxi_node,
            pickup_node=pickup_node,
            ride_phase=0,
            done=False,
            step_count=0
        ), key

    return single_reset(rng_key)

class JAXRideEnv(eqx.Module):
    adj_list: jnp.ndarray         # shape [num_nodes, max_deg]
    travel_times: jnp.ndarray     # shape [num_nodes, max_deg]
    max_deg: int
    num_nodes: int
    distances: jnp.ndarray        # shape [num_nodes, num_nodes], shortest-path lengths
    max_steps: int                     # maximum allowed steps per episode
    fixed_starts: jnp.ndarray     # 1D array of valid start-node indices
    fixed_pickups: jnp.ndarray  
    timeout_penalty: float = -500.0     # penalty if timeout before pickup

    def step(self, state: TaxiState, action: int) -> tuple[TaxiState, float]:
        current = state.current_node
        next_node = self.adj_list[current, action]
        time_cost = self.travel_times[current, action]

        # debug.print("current node: {}, next node: {}, time cost: {}", current[0], next_node[0], time_cost[0])

        # invalid move if no neighbor
        invalid = (next_node == -1)
        # ride terminates on invalid or on reaching the pickup node
        reach_pickup = (next_node == state.pickup_node)
        next_phase = jnp.where(reach_pickup, 1, state.ride_phase)

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

        new_state = TaxiState(
            current_node=next_node,
            pickup_node=state.pickup_node,
            ride_phase=next_phase,
            step_count=jnp.where(done, 0, next_step),
            done=done
        )
        return new_state, reward

    def reset(self, rng_key) -> TaxiState:
        return init_env(rng_key, self.fixed_starts, self.fixed_pickups, self.distances)