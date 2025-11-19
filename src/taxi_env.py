import jax
import jax.numpy as jnp
import jax.random as jrandom  
import equinox as eqx
from typing import NamedTuple, Sequence, Tuple, Dict, Optional
import gymnasium as gym
import numpy as np

# State
class TaxiState(NamedTuple):
    current_node: jnp.ndarray  # shape () - scalar int32
    pickup_node: jnp.ndarray   # shape () - scalar int32  
    done: jnp.ndarray          # shape () - scalar bool
    step_count: jnp.ndarray    # shape () - scalar int32
    neighbor_mask: jnp.ndarray # shape [max_deg]
    time: jnp.ndarray          # shape () - scalar float32


@jax.jit
def init_env(
    rng_key, 
    start_idx: int, 
    pickup_idx: int,
    neighbor_mask_static: jnp.ndarray 
    ) -> Tuple[TaxiState, jnp.ndarray]:
    """Initialize the Taxi environment state. using ot-matched start and pickup indices."""

    nm = neighbor_mask_static[start_idx]
    done = (start_idx == pickup_idx)
    state = TaxiState(
       current_node=jnp.int32(start_idx),
       pickup_node=jnp.int32(pickup_idx),
       done=done,
       step_count=jnp.int32(0),
       neighbor_mask=nm,
       time=jnp.array(0.0, dtype=jnp.float32)
    )
    return state, rng_key

# Environment
class TaxiEnv(eqx.Module):
    # Graph and base dynamics
    adj_list: jnp.ndarray         # [num_nodes, max_deg]
    travel_times: jnp.ndarray     # [num_nodes, max_deg]
    max_travel_time: float
    neighbor_mask_static: jnp.ndarray
    max_deg: int
    num_nodes: int
    distances: jnp.ndarray        # [num_nodes, num_nodes]
    hop_distances: jnp.ndarray    # [num_nodes, num_nodes]
    node_coordinates: Optional[jnp.ndarray] = None  # [num_nodes, 2] - normalized lat/lon coordinates
    paths_dict: Optional[dict] = None  # Dict of (source, target) -> path for precomputed shortest paths
    max_steps: int
    fixed_starts: jnp.ndarray     # [num_starts]
    fixed_pickups: jnp.ndarray    # [num_pickups]
    # Traffic-signal parameters per edge (at the end of each edge)
    periods: jnp.ndarray          # [num_nodes, max_deg]
    max_wait_time: float
    green_durations: jnp.ndarray  # [num_nodes, max_deg]
    offsets: jnp.ndarray          # [num_nodes, max_deg]
    # Congestion weight & timeout
    # alpha: float                  # per-vehicle delay
    pickup_bonus: float = 50.0  # bonus for reaching the pickup
    timeout_penalty: float
    global_state_dim: int
    gamma: float

    def __init__(
        self,
        adj_list: jnp.ndarray,
        travel_times: jnp.ndarray,
        neighbor_mask_static: jnp.ndarray,
        fixed_starts: Sequence[int],
        fixed_pickups: Sequence[int],
        distances: jnp.ndarray,
        hop_distances: jnp.ndarray,
        max_steps: int,
        traffic_params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # (periods, green_durations, offsets) each [num_nodes, max_deg]
        paths_dict: Optional[dict] = None,
        node_coordinates: Optional[jnp.ndarray] = None,  # [num_nodes, 2] - normalized lat/lon coordinates
        # alpha: float = 1.0,
        pickup_bonus: float = 5.0,
        timeout_penalty: float = -50.0,
        gamma: float = 0.99,
    ):
        # static graph data - ensure all arrays are on GPU with proper dtypes
        travel_times_host = np.asarray(travel_times, dtype=np.float32)
        valid_times = travel_times_host[travel_times_host > 0.0]
        # Equinox modules are frozen dataclasses; use object.__setattr__ for new fields
        object.__setattr__(
            self,
            "avg_travel_time_per_hop",
            float(valid_times.mean()) if valid_times.size else 0.0,
        )

        self.adj_list = jax.device_put(jnp.array(adj_list, dtype=jnp.int32))
        self.travel_times = jax.device_put(jnp.array(travel_times, dtype=jnp.float32))
        self.max_travel_time = float(travel_times.max())
        self.neighbor_mask_static = jax.device_put(jnp.array(neighbor_mask_static, dtype=bool))
        self.distances = jax.device_put(jnp.array(distances, dtype=jnp.float32))  # shape [num_nodes, num_nodes]
        self.hop_distances = jax.device_put(jnp.array(hop_distances, dtype=jnp.float32))  # shape [num_nodes, num_nodes]
        self.paths_dict = paths_dict
        # Store normalized node coordinates for Euclidean distance computation
        if node_coordinates is not None:
            self.node_coordinates = jax.device_put(jnp.array(node_coordinates, dtype=jnp.float32))  # [num_nodes, 2]
        else:
            self.node_coordinates = None

        # sizes
        self.num_nodes, self.max_deg = adj_list.shape
        self.max_steps = max_steps

        # fixed starts/pickups
        self.fixed_starts = jax.device_put(jnp.array(fixed_starts, dtype=jnp.int32))
        self.fixed_pickups = jax.device_put(jnp.array(fixed_pickups, dtype=jnp.int32))

        # congestion params
        self.pickup_bonus = pickup_bonus
        self.timeout_penalty = timeout_penalty
        # self.alpha = alpha
        self.gamma = gamma

        # Traffic params
        periods_arr, green_durations_arr, offsets_arr = traffic_params
        self.periods = jax.device_put(jnp.array(periods_arr, dtype=jnp.float32))  # [num_nodes, max_deg]
        self.green_durations = jax.device_put(jnp.array(green_durations_arr, dtype=jnp.float32))  # [num_nodes, max_deg]
        self.offsets = jax.device_put(jnp.array(offsets_arr, dtype=jnp.float32))  # [num_nodes, max_deg]
        self.max_wait_time = float(self.periods.max())  # max cycle length - convert to Python float
        
        # Align arrays
        self.periods = jnp.asarray(self.periods, dtype=jnp.float32)
        self.green_durations = jnp.asarray(self.green_durations, dtype=jnp.float32)
        self.offsets = jnp.asarray(self.offsets, dtype=jnp.float32)

        # Global traffic params
        self.global_state_dim = 3 * self.num_nodes  # [N,3] -> [3*N]

    @jax.jit
    def reset(self, rng_key) -> Tuple[TaxiState, jnp.ndarray]:
        # Reset
        key1, subkey = jrandom.split(rng_key)
        key2, rng_key = jrandom.split(subkey)
        start = jrandom.choice(key1, self.fixed_starts)
        pickup = jrandom.choice(key2, self.fixed_pickups)
        return init_env(rng_key,
                        start,
                        pickup,
                        self.neighbor_mask_static)[0]

    @jax.jit
    def step(self, state: TaxiState, action: int) -> Tuple[TaxiState, float, bool, dict]:
        """Environment step
        
        If the episode is already done (state.done == True), the agent stays in place
        and receives zero reward. This prevents agents from continuing to move and
        accumulate rewards after reaching the pickup point.
        """
        # Already done
        already_done = state.done
        
        # Base move
        curr = state.current_node
        nxt = self.adj_list[curr, action]
        travel = self.travel_times[curr, action]

        # Check for invalid moves - should not happen if action masking is correct
        # Raise an error if invalid move is attempted
        invalid = (nxt == -1)
        reach = (curr == state.pickup_node) | (nxt == state.pickup_node)
        step_n = state.step_count + 1
        done = reach

        # Time calculations
        t1 = state.time + travel
        
        # Signal phase & wait - use edge-based traffic params (at the end of the edge)
        # DEBUG: Set all traffic lights to green (wait = 0) for time-invariant shortest path learning
        period_edge = self.periods[curr, action]
        offset_edge = self.offsets[curr, action]
        green_edge = self.green_durations[curr, action]
        
        cycle = (t1 + offset_edge) % period_edge
        # wait = jnp.where(cycle < green_edge, 0.0, period_edge - cycle)  # Original traffic light logic
        wait = 0.0  # DEBUG: All lights green - no waiting
        t2 = t1 + wait
        norm_time = t2 % period_edge

        # Reward calculation - optimized
        total_delay = travel + wait
        
        # EUCLIDEAN DISTANCE
        # Use Euclidean distance from normalized coordinates if available, otherwise fall back to travel time distance
        # Note: In JIT-compiled code, we can't check None with Python if, so we always compute both and select
        # Since node_coordinates is either None (not provided) or a valid array, we use jnp.where with a check
        # if self.node_coordinates is not None:
        #     # Compute Euclidean distance from normalized coordinates
        #     coord_nxt = self.node_coordinates[nxt]  # [2]
        #     coord_pickup = self.node_coordinates[state.pickup_node]  # [2]
        #     diff = coord_nxt - coord_pickup
        #     dist_n = jnp.sqrt(jnp.sum(diff * diff) + 1e-8)  # Euclidean distance in normalized coordinate space (add small epsilon for numerical stability)
        # else:
        #     # Fallback to shortest path travel time distance
        # dist_n = self.distances[nxt, state.pickup_node]
        # 
        # # Add pickup bonus when reaching the pickup location
        # pickup_bonus = jnp.where(reach, self.pickup_bonus, 0.0)
        # # Scale down delay penalty and increase pickup bonus to ensure positive rewards for completion
        # # Using Euclidean distance penalty instead of travel time distance
        # reward = - total_delay/60.0 - 0.5*dist_n/60.0
        
        # HOP DISTANCES
        # Compute hop distances from current and next nodes to pickup
        # hop_dist_curr = self.hop_distances[curr, state.pickup_node]
        # hop_dist_next = self.hop_distances[nxt, state.pickup_node]
        # hop_dist_diff = hop_dist_curr - hop_dist_next
        # Average travel time per hop
        # valid_travel_times = jnp.where(self.travel_times > 0.0, self.travel_times, 0.0)
        # sum_valid = jnp.sum(valid_travel_times)
        # count_valid = jnp.sum(self.travel_times > 0.0, dtype=jnp.float32)
        # avg_travel_time_per_hop = jnp.where(
        #     count_valid > 0.0, sum_valid / count_valid, 0.0
        # )
        # Scale the hop-based shaping term by the average travel time per hop
        # hop_shaping = (avg_travel_time_per_hop / 60.0) * hop_dist_diff
        dist_curr = self.distances[curr, state.pickup_node]
        dist_next = self.distances[nxt, state.pickup_node]
        dist_diff = dist_curr - dist_next
        dist_shaping = dist_diff / 60.0
        
        # Pickup bonus
        pickup_bonus = jnp.where(reach, self.pickup_bonus, 0.0)

        # Reward scaling (we want to make the reward between -1 and 1)
        reward_scaling = 1.0 #5e-3

        # Include distance shaping to guide agent toward pickup
        # Positive reward for moving closer (dist_diff > 0), negative for moving away
        reward = - reward_scaling * total_delay/60.0 + pickup_bonus
        
        # Already done
        reward = jnp.where(already_done, 0.0, reward)

        # Debug
        # jax.debug.print("Reward: delay={delay:.2f}, shaping={shaping:.2f}, pickup_bonus={bonus:.2f}, reward={reward:.2f}, reach={reach}", 
        #                 delay=total_delay, shaping=shaping, bonus=pickup_bonus, reward=reward, reach=reach)

        # Neighbor mask
        nm = self.neighbor_mask_static[nxt]
        
        # Already done
        at_pickup_before = (curr == state.pickup_node)
        reached_pickup_this_step = reach & ~already_done
        final_node = jnp.where(
            already_done,
            curr,
            jnp.where(
                reached_pickup_this_step,
                jnp.where(at_pickup_before, curr, nxt),  # At pickup: stay at curr if already there, otherwise use nxt
                nxt  # Not at pickup, move to next node
            )
        )
        final_neighbor_mask = jnp.where(already_done, state.neighbor_mask, nm)
        final_time = jnp.where(already_done, state.time, norm_time)
        final_step_count = jnp.where(already_done, state.step_count, jnp.where(done, 0, step_n))
        final_done = jnp.where(already_done, True, done)
        
        # Already done
        final_wait = jnp.where(already_done, 0.0, wait)
        final_travel = jnp.where(already_done, 0.0, travel)

        # State
        new_state = TaxiState(
            current_node=final_node,
            pickup_node=state.pickup_node,
            done=final_done,
            step_count=final_step_count,
            neighbor_mask=final_neighbor_mask,
            time=final_time
        )
        return new_state, reward, final_done, {"wait": final_wait, "travel": final_travel}