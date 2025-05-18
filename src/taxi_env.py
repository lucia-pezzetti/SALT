import jax
import jax.numpy as jnp
import jax.random as jrandom  
import equinox as eqx
from typing import NamedTuple, Sequence, Tuple, Dict

# ----- State -----
class TaxiState(NamedTuple):
    current_node:  int       # use explicit int32
    pickup_node:   int
    done:          bool       # this is fine
    step_count:    int
    neighbor_mask: jnp.ndarray     # shape [max_deg]
    time:          float     # global clock in seconds         # global clock

# ----- Reset / Init -----
@jax.jit
def init_env(
    rng_key,
    fixed_starts: Sequence[int],
    fixed_pickups: Sequence[int],
    neighbor_mask_static: jnp.ndarray,
) -> Tuple[TaxiState, jnp.ndarray]:
    # sample a start and pickup
    starts = jnp.asarray(fixed_starts, dtype=jnp.int32)
    pickups = jnp.asarray(fixed_pickups, dtype=jnp.int32)
    k1, k2, new_key = jrandom.split(rng_key, 3)
    taxi_idx = jrandom.randint(k1, (), 0, starts.shape[0])
    pickup_idx = jrandom.randint(k2, (), 0, pickups.shape[0])
    curr = starts[taxi_idx]
    drop = pickups[pickup_idx]
    nm = neighbor_mask_static[curr]

    # initial time & empty queues
    time0 = jnp.array(0.0, dtype=jnp.float32)

    state = TaxiState(
        current_node=curr,
        pickup_node=drop,
        done=False,
        step_count=jnp.int32(0),
        neighbor_mask=nm,
        time=time0
    )
    return state, new_key

# ----- Environment -----
class JAXRideEnv(eqx.Module):
    # Graph and base dynamics
    adj_list: jnp.ndarray         # [num_nodes, max_deg]
    travel_times: jnp.ndarray     # [num_nodes, max_deg]
    neighbor_mask_static: jnp.ndarray
    max_deg: int
    num_nodes: int
    distances: jnp.ndarray        # [num_nodes, num_nodes]
    max_steps: int
    fixed_starts: jnp.ndarray     # [num_starts]
    fixed_pickups: jnp.ndarray    # [num_pickups]
    # Traffic-signal parameters per node
    periods: jnp.ndarray          # [num_nodes]
    green_durations: jnp.ndarray  # [num_nodes]
    offsets: jnp.ndarray          # [num_nodes]
    # Congestion weight & timeout
    # alpha: float                  # per-vehicle delay
    timeout_penalty: float

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
        timeout_penalty: float = -50.0,
    ):
        # static graph data
        self.adj_list = adj_list
        self.travel_times = travel_times
        self.neighbor_mask_static = neighbor_mask_static
        self.distances = distances

        # sizes
        self.num_nodes, self.max_deg = adj_list.shape
        self.max_steps = max_steps

        # fixed starts/pickups
        self.fixed_starts = jnp.array(fixed_starts, dtype=jnp.int32)
        self.fixed_pickups = jnp.array(fixed_pickups, dtype=jnp.int32)

        # congestion params
        self.timeout_penalty = timeout_penalty
        # self.alpha = alpha

        # unpack traffic_params
        nodes = jnp.array(list(traffic_params.keys()), dtype=jnp.int32)
        params = jnp.array(list(traffic_params.values()), dtype=jnp.float32)  # shape [N,3]
        periods_vals = params[:, 0]
        green_vals  = params[:, 1]
        offset_vals = params[:, 2]

        zeros = jnp.zeros((self.num_nodes,), dtype=jnp.float32)
        self.periods = zeros.at[nodes].set(periods_vals)
        self.green_durations = zeros.at[nodes].set(green_vals)
        self.offsets = zeros.at[nodes].set(offset_vals)

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

    @jax.jit
    def reset(self, rng_key) -> Tuple[TaxiState, jnp.ndarray]:
        # resets global time via init_env
        return init_env(rng_key,
                        self.fixed_starts,
                        self.fixed_pickups,
                        self.neighbor_mask_static)

    @jax.jit
    def step(self, state: TaxiState, action: int) -> Tuple[TaxiState, float, bool]:
        # 1) Base move
        curr = state.current_node
        nxt = self.adj_list[curr, action].astype(jnp.int32)
        travel = self.travel_times[curr, action]

        # 2) Terminal logic
        invalid = (nxt == -1)
        reach = (nxt == state.pickup_node)
        step_n = state.step_count + 1
        timeout = (step_n >= self.max_steps) & (~reach)
        done = invalid | reach | timeout

        # 3) Time after moving
        t1 = state.time + travel

        # 4) Signal phase & wait
        cycle    = (t1 + self.offsets[nxt]) % self.periods[nxt]
        wait     = jnp.where(cycle < self.green_durations[nxt], 0.0, self.periods[nxt] - cycle)
        t2       = t1 + wait

        # no congestion term any more
        total_delay = travel + wait
        dist_c = self.distances[curr, state.pickup_node]
        dist_n = self.distances[nxt, state.pickup_node]
        shaping = dist_c - dist_n
        bonus = jnp.where(reach, 50.0, 0.0)
        reward      = - total_delay/60.0 + shaping + bonus

        new_state = TaxiState(
        current_node   = nxt,
        pickup_node    = state.pickup_node,
        done           = done,
        step_count     = jnp.where(done, 0, state.step_count+1),
        neighbor_mask  = self.neighbor_mask_static[nxt],
        time           = t2
        )

        # jax.debug.print("step: {}, done: {}, curr: {}, nxt: {}, travel: {}, wait: {}, shaping: {}, reward: {}",
        #                 step_n[0], done[0], curr[0], nxt[0], travel[0]/60.0, wait[0]/60.0, shaping[0], reward[0])

        # per = self.periods[nxt]
        # grd = self.green_durations[nxt]
        # off = self.offsets[nxt]
        # cycle_pos = (t1 + off) % per
        # wait = jnp.where(cycle_pos < grd, 0.0, per - cycle_pos)

        # # 5) Queue update
        # join = (wait > 0.0).astype(jnp.int32)
        # queues_inc = state.queues.at[nxt].add(join)
        # delay_q = self.alpha * queues_inc[nxt]
        # queues_fin = queues_inc.at[nxt].add(-join)

        # # 6) Final time & reward
        # t2 = t1 + wait
        # time_pen = - (travel + wait + delay_q) / 60.0
        # dist_c = self.distances[curr, state.pickup_node]
        # dist_n = self.distances[nxt, state.pickup_node]
        # shaping = dist_c - dist_n
        # bonus = jnp.where(reach, 50.0, 0.0)
        # to_pen = jnp.where(timeout, self.timeout_penalty, 0.0)
        # rew = time_pen + shaping + bonus + to_pen

        # 7) New neighbor mask
        nm = self.neighbor_mask_static[nxt]

        # 8) New state
        new_state = TaxiState(
            current_node=nxt,
            pickup_node=state.pickup_node,
            done=done,
            step_count=jnp.where(done, jnp.int32(0), step_n),
            neighbor_mask=nm,
            time=t2,
        )
        return new_state, reward, reach, {"wait": wait, "travel": travel}
