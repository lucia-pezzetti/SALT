import jax
import jax.numpy as jnp
import jax.random as jrandom  
import equinox as eqx
from typing import NamedTuple, Sequence, Tuple, Dict
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
    pickup_bonus: float = 5.0  # bonus for reaching the pickup
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

        # unpack traffic_params - ensure on GPU
        nodes = jax.device_put(jnp.array(list(traffic_params.keys()), dtype=jnp.int32))
        params = jax.device_put(jnp.array(list(traffic_params.values()), dtype=jnp.float32))  # shape [N,3]
        periods_vals = params[:, 0]
        green_vals  = params[:, 1]
        offset_vals = params[:, 2]

        zeros = jax.device_put(jnp.zeros((self.num_nodes,), dtype=jnp.float32))
        self.periods = zeros.at[nodes].set(periods_vals)
        self.max_wait_time = float(self.periods.max())  # max cycle length - convert to Python float
        self.green_durations = zeros.at[nodes].set(green_vals)
        self.offsets = zeros.at[nodes].set(offset_vals)

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
        # Base move
        curr = state.current_node
        nxt = self.adj_list[curr, action]
        travel = self.travel_times[curr, action]

        # Terminal logic
        invalid = (nxt == -1)
        reach = (curr == state.pickup_node) | (nxt == state.pickup_node)
        step_n = state.step_count + 1
        timeout = (step_n >= self.max_steps) & (~reach)
        done = invalid | reach | timeout

        # Time after moving
        t1 = state.time + travel

        # Signal phase & wait
        cycle    = (t1 + self.offsets[nxt]) % self.periods[nxt]
        wait     = jnp.where(cycle < self.green_durations[nxt], 0.0, self.periods[nxt] - cycle)
        t2       = t1 + wait
        norm_time = t2 % self.periods[nxt]  # normalize to [0, period)

        total_delay = travel + wait
        dist_c = self.distances[curr, state.pickup_node]
        dist_n = self.distances[nxt, state.pickup_node]
        shaping = dist_c - self.gamma * dist_n
        bonus = jnp.where(reach, self.pickup_bonus, 0.0)
        reward      = - total_delay/60.0 #+ bonus + shaping

        # 7) New neighbor mask
        nm = self.neighbor_mask_static[nxt]

        # 8) New state
        new_state = TaxiState(
            current_node=jnp.int32(nxt),  # ensure it's a JAX array
            pickup_node=state.pickup_node,
            done=done,
            step_count=jnp.where(done, jnp.int32(0), step_n),
            neighbor_mask=nm,
            time=jnp.float32(norm_time)
        )
        return new_state, reward, done, {"wait": wait, "travel": travel}