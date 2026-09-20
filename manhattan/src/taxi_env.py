import jax
import jax.numpy as jnp
import jax.random as jrandom  
import equinox as eqx
from typing import NamedTuple, Sequence, Tuple, Dict, Optional
import gymnasium as gym
import numpy as np


def pickup_bonus_reward_from_seconds(seconds: float) -> float:
    """Convert a travel-time-equivalent bonus in seconds to reward units."""
    if seconds < 0:
        raise ValueError("pickup bonus seconds must be non-negative")
    return seconds / 60.0


# State
class TaxiState(NamedTuple):
    current_node: jnp.ndarray  
    pickup_node: jnp.ndarray    
    done: jnp.ndarray          
    step_count: jnp.ndarray    
    neighbor_mask: jnp.ndarray 
    time: jnp.ndarray          


@jax.jit
def mask_immediate_reverse_actions(
    adj_list: jnp.ndarray,
    base_mask: jnp.ndarray,
    current_node: jnp.ndarray,
    previous_node: jnp.ndarray,
) -> jnp.ndarray:
    """Disallow an immediate return when another valid successor exists."""
    neighbors = adj_list[current_node]
    without_reverse = base_mask & (neighbors != previous_node)
    return jnp.where(jnp.any(without_reverse), without_reverse, base_mask)


@jax.jit
def effective_action_mask(
    adj_list: jnp.ndarray,
    forced_return_actions: jnp.ndarray,
    current_node: jnp.ndarray,
    pickup_node: jnp.ndarray,
    base_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Also avoid entering a forced-return spur unless it is the target."""
    enters_target = adj_list[current_node] == pickup_node
    legal_mask = base_mask & (~forced_return_actions[current_node] | enters_target)
    return jnp.where(jnp.any(legal_mask), legal_mask, base_mask)


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
    forced_return_actions: jnp.ndarray
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
    # Reward and discount parameters
    pickup_bonus: float = 50.0  # bonus for reaching the pickup
    global_state_dim: int
    gamma: float
    # Per-step congestion noise: noise_mask[i,j] is the max extra fraction for edge (i,j).
    # 0.0 means deterministic; >0 means travel *= (1 + U(0, noise_mask[i,j])) each step.
    noise_mask: jnp.ndarray       # [num_nodes, max_deg]

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
        pickup_bonus: float = 50.0,
        gamma: float = 0.99,
        noise_mask: Optional[jnp.ndarray] = None,  # [num_nodes, max_deg] - per-edge noise ceiling
    ):
        travel_times_host = np.asarray(travel_times, dtype=np.float32)
        valid_times = travel_times_host[travel_times_host > 0.0]
        object.__setattr__(
            self,
            "avg_travel_time_per_hop",
            float(valid_times.mean()) if valid_times.size else 0.0,
        )

        self.adj_list = jax.device_put(jnp.array(adj_list, dtype=jnp.int32))
        self.travel_times = jax.device_put(jnp.array(travel_times, dtype=jnp.float32))
        self.max_travel_time = float(travel_times.max())
        self.neighbor_mask_static = jax.device_put(jnp.array(neighbor_mask_static, dtype=bool))
        adj_host = np.asarray(adj_list, dtype=np.int32)
        neighbor_mask_host = np.asarray(neighbor_mask_static, dtype=bool)
        forced_return_actions = np.zeros_like(neighbor_mask_host)
        for current in range(adj_host.shape[0]):
            for action in np.flatnonzero(neighbor_mask_host[current]):
                next_node = adj_host[current, action]
                next_successors = set(adj_host[next_node, neighbor_mask_host[next_node]].tolist())
                forced_return_actions[current, action] = next_successors == {current}
        self.forced_return_actions = jax.device_put(forced_return_actions)
        self.distances = jax.device_put(jnp.array(distances, dtype=jnp.float32))  # shape [num_nodes, num_nodes]
        self.hop_distances = jax.device_put(jnp.array(hop_distances, dtype=jnp.float32))  # shape [num_nodes, num_nodes]
        self.paths_dict = paths_dict
        if node_coordinates is not None:
            self.node_coordinates = jax.device_put(jnp.array(node_coordinates, dtype=jnp.float32))  # [num_nodes, 2]
        else:
            self.node_coordinates = None

        self.num_nodes, self.max_deg = adj_list.shape
        self.max_steps = max_steps

        # fixed starts/pickups
        self.fixed_starts = jax.device_put(jnp.array(fixed_starts, dtype=jnp.int32))
        self.fixed_pickups = jax.device_put(jnp.array(fixed_pickups, dtype=jnp.int32))

        # congestion params
        self.pickup_bonus = pickup_bonus
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

        # Per-step congestion noise mask
        if noise_mask is not None:
            self.noise_mask = jax.device_put(jnp.array(noise_mask, dtype=jnp.float32))
        else:
            # All zeros → no noise (fully deterministic travel times)
            self.noise_mask = jnp.zeros_like(self.travel_times)

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
    def step(self, state: TaxiState, action: int, noise_key: Optional[jnp.ndarray] = None) -> Tuple[TaxiState, float, bool, dict]:
        """Environment step
        
        If the episode is already done (state.done == True), the agent stays in place
        and receives zero reward. This prevents agents from continuing to move and
        accumulate rewards after reaching the pickup point.
        
        Args:
            noise_key: Optional JAX PRNG key. When provided and noise_mask > 0 for
                       the traversed edge, travel time is multiplied by
                       (1 + U(0, noise_mask[curr, action])).
        """
        # Already done
        already_done = state.done
        
        # Base move
        curr = state.current_node
        nxt = self.adj_list[curr, action]
        travel = self.travel_times[curr, action]
        
        # Per-step congestion noise (only when a key is provided)
        if noise_key is not None:
            ceil = self.noise_mask[curr, action]  # max extra fraction for this edge
            noise_factor = 1.0 + ceil * jrandom.uniform(noise_key)  # U(1, 1+ceil)
            travel = travel * noise_factor

        # Raise an error if invalid move is attempted
        invalid = (nxt == -1)
        reach = (curr == state.pickup_node) | (nxt == state.pickup_node)
        step_n = state.step_count + 1
        done = reach

        # Time calculations
        t1 = state.time + travel
        
        # Signal phase & wait - use edge-based traffic params
        period_edge = self.periods[curr, action]
        offset_edge = self.offsets[curr, action]
        green_edge = self.green_durations[curr, action]
        
        cycle = (t1 + offset_edge) % period_edge
        wait = jnp.where(cycle < green_edge, 0.0, period_edge - cycle)  # Original traffic light logic
        t2 = t1 + wait
        norm_time = t2 % period_edge

        # Reward calculation - optimized
        total_delay = travel + wait
        
        # Pickup bonus
        pickup_bonus = jnp.where(reach, self.pickup_bonus, 0.0)

        # Positive reward for moving closer (dist_diff > 0), negative for moving away
        reward = - total_delay/60.0 + pickup_bonus # + dist_shaping
        
        # Already done
        reward = jnp.where(already_done, 0.0, reward)

        # Neighbor mask
        nm = mask_immediate_reverse_actions(
            self.adj_list,
            self.neighbor_mask_static[nxt],
            nxt,
            curr,
        )
        
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
