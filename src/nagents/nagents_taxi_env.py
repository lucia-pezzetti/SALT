import jax
import jax.numpy as jnp
import jax.random as jrandom  
import equinox as eqx
from typing import NamedTuple, Sequence, Tuple, Dict
import gymnasium as gym

# ----- State -----
class TaxiState(NamedTuple):
    current_node:  int       # use explicit int32
    pickup_node:   int
    done:          bool       # this is fine
    step_count:    int
    neighbor_mask: jnp.ndarray     # shape [max_deg]
    time:          float     # global clock in seconds         # global clock


class MultiTaxiEnv:
    def __init__(self, n_agents, env_kwargs):
        self.n = n_agents
        self.envs = [TaxiEnv(**env_kwargs) for _ in range(n_agents)]
        self.observation_space = gym.spaces.Tuple([e.observation_space for e in self.envs])
        self.action_space = gym.spaces.Tuple([e.action_space for e in self.envs])

    def reset(self):
        states = [e.reset() for e in self.envs]
        return jnp.stack(states)

    def step(self, actions):
        next_states, rewards, dones, infos = [], [], [], []
        for e, a in zip(self.envs, actions):
            s2, r, d, i = e.step(a)
            next_states.append(s2)
            rewards.append(r)
            dones.append(d)
            infos.append(i)
        return jnp.stack(next_states), jnp.array(rewards), jnp.array(dones), infos

# ----- Reset / Init -----
# @jax.jit
# def init_env(
#     rng_key,
#     fixed_starts: Sequence[int],
#     fixed_pickups: Sequence[int],
#     neighbor_mask_static: jnp.ndarray,
# ) -> Tuple[TaxiState, jnp.ndarray]:
#     # sample a start and pickup
#     starts = jnp.asarray(fixed_starts, dtype=jnp.int32)
#     pickups = jnp.asarray(fixed_pickups, dtype=jnp.int32)
#     k1, k2, new_key = jrandom.split(rng_key, 3)
#     taxi_idx = jrandom.randint(k1, (), 0, starts.shape[0])
#     pickup_idx = jrandom.randint(k2, (), 0, pickups.shape[0])
#     curr = starts[taxi_idx]
#     pickup = pickups[pickup_idx]
#     nm = neighbor_mask_static[curr]
#     done = jnp.where(curr == pickup, True, False)

#     # initial time & empty queues
#     time0 = jnp.array(0.0, dtype=jnp.float32)

#     state = TaxiState(
#         current_node=curr,
#         pickup_node=pickup,
#         done=done,
#         step_count=jnp.int32(0),
#         neighbor_mask=nm,
#         time=time0
#     )
#     return state, new_key

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

        # global traffic params shape [3, num_nodes] (green/period rate, is green, time to next switch)
        self.global_state_dim = 3 * self.num_nodes

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
        # Base move
        curr = state.current_node
        nxt = self.adj_list[curr, action].astype(jnp.int32)
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
        norm_time = t2 / 600.0

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
        #     jax.debug.print("1: step: {}, done: {}, curr: {}, nxt: {}, pickup: {}, action: {}, travel: {:.2f}, wait: {:.2f}, shaping: {:.2f}, reward: {:.2f}",
        #             step_n[1], done[1], curr[1], nxt[1], state.pickup_node[1], action[1],
        #             travel[1]/60.0, wait[1]/60.0, shaping[1], reward[1])
        #     jax.debug.print("2: step: {}, done: {}, curr: {}, nxt: {}, pickup: {}, action: {}, travel: {:.2f}, wait: {:.2f}, shaping: {:.2f}, reward: {:.2f}",
        #             step_n[2], done[2], curr[2], nxt[2], state.pickup_node[2], action[2],
        #             travel[2]/60.0, wait[2]/60.0, shaping[2], reward[2])
        #     jax.debug.print("3: step: {}, done: {}, curr: {}, nxt: {}, pickup: {}, action: {}, travel: {:.2f}, wait: {:.2f}, shaping: {:.2f}, reward: {:.2f}",
        #             step_n[3], done[3], curr[3], nxt[3], state.pickup_node[3], action[3],
        #             travel[3]/60.0, wait[3]/60.0, shaping[3], reward[3])
        #     jax.debug.print("4: step: {}, done: {}, curr: {}, nxt: {}, pickup: {}, action: {}, travel: {:.2f}, wait: {:.2f}, shaping: {:.2f}, reward: {:.2f}",
        #             step_n[4], done[4], curr[4], nxt[4], state.pickup_node[4], action[4],
        #             travel[4]/60.0, wait[4]/60.0, shaping[4], reward[4])
        #     jax.debug.print("5: step: {}, done: {}, curr: {}, nxt: {}, pickup: {}, action: {}, travel: {:.2f}, wait: {:.2f}, shaping: {:.2f}, reward: {:.2f}",
        #             step_n[5], done[5], curr[5], nxt[5], state.pickup_node[5], action[5],
        #             travel[5]/60.0, wait[5]/60.0, shaping[5], reward[5])
        #     jax.debug.print("6: step: {}, done: {}, curr: {}, nxt: {}, pickup: {}, action: {}, travel: {:.2f}, wait: {:.2f}, shaping: {:.2f}, reward: {:.2f}",
        #             step_n[6], done[6], curr[6], nxt[6], state.pickup_node[6], action[6],
        #             travel[6]/60.0, wait[6]/60.0, shaping[6], reward[6])
        #     jax.debug.print("7: step: {}, done: {}, curr: {}, nxt: {}, pickup: {}, action: {}, travel: {:.2f}, wait: {:.2f}, shaping: {:.2f}, reward: {:.2f}",
        #             step_n[7], done[7], curr[7], nxt[7], state.pickup_node[7], action[7],
        #             travel[7]/60.0, wait[7]/60.0, shaping[7], reward[7])
        #     jax.debug.print("8: step: {}, done: {}, curr: {}, nxt: {}, pickup: {}, action: {}, travel: {:.2f}, wait: {:.2f}, shaping: {:.2f}, reward: {:.2f}",
        #             step_n[8], done[8], curr[8], nxt[8], state.pickup_node[8], action[8],
        #             travel[8]/60.0, wait[8]/60.0, shaping[8], reward[8])
        #     jax.debug.print("9: step: {}, done: {}, curr: {}, nxt: {}, pickup: {}, action: {}, travel: {:.2f}, wait: {:.2f}, shaping: {:.2f}, reward: {:.2f}",
        #             step_n[9], done[9], curr[9], nxt[9], state.pickup_node[9], action[9],
        #             travel[9]/60.0, wait[9]/60.0, shaping[9], reward[9])
            
        # 7) New neighbor mask
        nm = self.neighbor_mask_static[nxt]

        # 8) New state
        new_state = TaxiState(
            current_node=nxt,
            pickup_node=state.pickup_node,
            done=done,
            step_count=jnp.where(done, jnp.int32(0), step_n),
            neighbor_mask=nm,
            time=norm_time,
        )
        return new_state, reward, reach, {"wait": wait, "travel": travel}