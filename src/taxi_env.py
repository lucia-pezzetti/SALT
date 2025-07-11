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
    # no new random key needed (we’re driving it from outside)
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
        max_steps: int,
        traffic_params: Dict[int, Tuple[float, float, float]],
        # alpha: float = 1.0,
        pickup_bonus: float = 5.0,
        timeout_penalty: float = -50.0,
        gamma: float = 0.99,
    ):
        # static graph data
        self.adj_list = adj_list
        self.travel_times = travel_times
        self.max_travel_time = travel_times.max()
        self.neighbor_mask_static = neighbor_mask_static
        self.distances = distances

        # sizes
        self.num_nodes, self.max_deg = adj_list.shape
        self.max_steps = max_steps

        # fixed starts/pickups
        self.fixed_starts = jnp.array(fixed_starts, dtype=jnp.int32)
        self.fixed_pickups = jnp.array(fixed_pickups, dtype=jnp.int32)

        # congestion params
        self.pickup_bonus = pickup_bonus
        self.timeout_penalty = timeout_penalty
        # self.alpha = alpha
        self.gamma = gamma

        # unpack traffic_params
        nodes = jnp.array(list(traffic_params.keys()), dtype=jnp.int32)
        params = jnp.array(list(traffic_params.values()), dtype=jnp.float32)  # shape [N,3]
        periods_vals = params[:, 0]
        green_vals  = params[:, 1]
        offset_vals = params[:, 2]

        zeros = jnp.zeros((self.num_nodes,), dtype=jnp.float32)
        self.periods = zeros.at[nodes].set(periods_vals)
        self.max_wait_time = self.periods.max()  # max cycle length
        self.green_durations = zeros.at[nodes].set(green_vals)
        self.offsets = zeros.at[nodes].set(offset_vals)

        # global traffic params shape [num_nodes, 3] (green/period rate, is green, time to next switch)
        self.global_state_dim = 3 * self.num_nodes  # [N,3] -> [3*N]

        # runtime sanity checks
        assert self.periods.shape[0] == self.num_nodes, (
            f"Expected {self.num_nodes} periods, got {self.periods.shape[0]}"
        )
        assert jnp.all(self.periods > 0), (
            f"Cycle lengths must be >0, min found {self.periods.min()}"
        )
        assert jnp.all((self.green_durations >= 0) & (self.green_durations <= self.periods)), (
            "Green durations must satisfy 0 <= green <= period"
        )
        assert jnp.all((self.offsets >= 0) & (self.offsets < self.periods)), (
            "Offsets must satisfy 0 <= offset < period"
        )

    # @jax.jit
    # def reset(self, rng_key) -> Tuple[TaxiState, jnp.ndarray]:
    #     # resets global time via init_env
    #     key1, subkey = jrandom.split(rng_key)
    #     key2, rng_key = jrandom.split(subkey)
    #     start = jrandom.choice(key1, self.fixed_starts)
    #     pickup = jrandom.choice(key2, self.fixed_pickups)
    #     return init_env(rng_key,
    #                     start,
    #                     pickup,
    #                     self.neighbor_mask_static)

    @jax.jit
    def step(self, state: TaxiState, action: int) -> Tuple[TaxiState, float, bool]:
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
        reward      = - total_delay/60.0 + shaping + bonus

        # if step_n.ndim == 0:
        #     jax.debug.print("step: {}, done: {}, curr: {}, nxt: {}, pickup: {}, action: {}, travel: {:.2f}, wait: {:.2f}, shaping: {:.2f}, reward: {:.2f}",
        #                     step_n, done, curr, nxt, state.pickup_node, action,
        #                     travel/60.0, wait/60.0, shaping, reward)
        # else:
        #     jax.debug.print("\n printing batch debug info")
        #     jax.debug.print("0: step: {}, done: {}, curr: {}, nxt: {}, pickup: {}, action: {}, travel: {:.2f}, wait: {:.2f}, shaping: {:.2f}, reward: {:.2f}",
        #             step_n[0], done[0], curr[0], nxt[0], state.pickup_node[0], action[0],
        #             travel[0]/60.0, wait[0]/60.0, shaping[0], reward[0])
            
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
        return new_state, reward, reach, {"wait": wait, "travel": travel}