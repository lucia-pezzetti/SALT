import jax
import jax.numpy as jnp
import jax.random as jrandom  
import equinox as eqx
from typing import NamedTuple, Sequence, Tuple, Dict, Optional
import gymnasium as gym

# ----- State -----
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

# ----- Environment -----
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
    # Traffic-signal parameters per node
    periods: jnp.ndarray          # [num_nodes]
    max_wait_time: float
    green_durations: jnp.ndarray  # [num_nodes]
    offsets: jnp.ndarray          # [num_nodes]
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
        traffic_params: Dict[int, Tuple[float, float, float]],
        paths_dict: Optional[dict] = None,
        node_coordinates: Optional[jnp.ndarray] = None,  # [num_nodes, 2] - normalized lat/lon coordinates
        # alpha: float = 1.0,
        pickup_bonus: float = 5.0,
        timeout_penalty: float = -50.0,
        gamma: float = 0.99,
    ):
        # static graph data - ensure all arrays are on GPU with proper dtypes
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

        # fixed starts/pickups - ensure on GPU
        self.fixed_starts = jax.device_put(jnp.array(fixed_starts, dtype=jnp.int32))
        self.fixed_pickups = jax.device_put(jnp.array(fixed_pickups, dtype=jnp.int32))

        # congestion params
        self.pickup_bonus = pickup_bonus
        self.timeout_penalty = timeout_penalty
        # self.alpha = alpha
        self.gamma = gamma

        # unpack traffic_params - ensure on GPU with optimal data types
        nodes = jax.device_put(jnp.array(list(traffic_params.keys()), dtype=jnp.int32))
        params = jax.device_put(jnp.array(list(traffic_params.values()), dtype=jnp.float32))  # shape [N,3]
        periods_vals = params[:, 0]
        green_vals  = params[:, 1]
        offset_vals = params[:, 2]

        # Pre-allocate with optimal memory layout
        zeros = jax.device_put(jnp.zeros((self.num_nodes,), dtype=jnp.float32))
        self.periods = zeros.at[nodes].set(periods_vals)
        self.max_wait_time = float(self.periods.max())  # max cycle length - convert to Python float
        self.green_durations = zeros.at[nodes].set(green_vals)
        self.offsets = zeros.at[nodes].set(offset_vals)
        
        # Ensure all arrays are properly aligned for GPU access
        self.periods = jnp.asarray(self.periods, dtype=jnp.float32)
        self.green_durations = jnp.asarray(self.green_durations, dtype=jnp.float32)
        self.offsets = jnp.asarray(self.offsets, dtype=jnp.float32)

        # global traffic params shape [num_nodes, 3] (green/period rate, is green, time to next switch)
        self.global_state_dim = 3 * self.num_nodes  # [N,3] -> [3*N]

    @jax.jit
    def reset(self, rng_key) -> Tuple[TaxiState, jnp.ndarray]:
        # resets global time via init_env
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
        """Optimized environment step function with reduced allocations and better GPU utilization"""
        # Base move - optimized memory access
        curr = state.current_node
        nxt = self.adj_list[curr, action]
        travel = self.travel_times[curr, action]

        # Terminal logic - combined conditions for efficiency
        invalid = (nxt == -1)
        reach = (curr == state.pickup_node) | (nxt == state.pickup_node)
        step_n = state.step_count + 1
        timeout = (step_n >= self.max_steps) & (~reach)
        done = invalid | reach | timeout

        # Time calculations - optimized
        t1 = state.time + travel
        
        # Signal phase & wait - pre-compute common values
        period_nxt = self.periods[nxt]
        offset_nxt = self.offsets[nxt]
        green_nxt = self.green_durations[nxt]
        
        cycle = (t1 + offset_nxt) % period_nxt
        wait = jnp.where(cycle < green_nxt, 0.0, period_nxt - cycle)
        t2 = t1 + wait
        norm_time = t2 % period_nxt

        # Reward calculation - optimized
        total_delay = travel + wait
        
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
            # Fallback to shortest path travel time distance
        dist_n = self.distances[nxt, state.pickup_node]
        
        # Add pickup bonus when reaching the pickup location
        pickup_bonus = jnp.where(reach, self.pickup_bonus, 0.0)
        # Scale down delay penalty and increase pickup bonus to ensure positive rewards for completion
        # Using Euclidean distance penalty instead of travel time distance
        reward = - total_delay / 60.0 - dist_n / 60.0 
        
        # Debug reward components to check if shaping is negligible
        # jax.debug.print("Reward: delay={delay:.2f}, shaping={shaping:.2f}, pickup_bonus={bonus:.2f}, reward={reward:.2f}, reach={reach}", 
        #                 delay=total_delay, shaping=shaping, bonus=pickup_bonus, reward=reward, reach=reach)

        # New neighbor mask - direct access
        nm = self.neighbor_mask_static[nxt]

        # New state - optimized construction
        new_state = TaxiState(
            current_node=nxt,  # already int32 from adj_list
            pickup_node=state.pickup_node,
            done=done,
            step_count=jnp.where(done, 0, step_n),  # simplified
            neighbor_mask=nm,
            time=norm_time  # already float32
        )
        return new_state, reward, done, {"wait": wait, "travel": travel}