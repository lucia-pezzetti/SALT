import networkx as nx
from nagents_taxi_env import TaxiState, TaxiEnv, init_env
import jax.numpy as jnp
from jax import random as jrandom
from typing import Callable, Sequence

class BaselineEvaluator:
    def __init__(self, G: nx.DiGraph, node_to_idx: dict,
                 init_state_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], TaxiState],
                 idx_to_node: Sequence[int]):
        self.G = G
        self.node_to_idx = node_to_idx
        self.init_state_fn = init_state_fn  # function(key, starts, pickups) -> TaxiState
        self.idx_to_node = idx_to_node      # map from index back to node ID

    def compute_shortest_path(self, start_idx: int, goal_idx: int) -> list:
        # Map from internal index to actual graph node
        start_node = self.idx_to_node[start_idx]
        goal_node = self.idx_to_node[goal_idx]
        # compute shortest path on OSM node IDs
        path = nx.shortest_path(self.G, start_node, goal_node, weight='travel_time_congested')
        return path

    def evaluate(
        self,
        env: TaxiEnv,
        key: jnp.ndarray,
        fixed_starts: list,
        fixed_pickups: list,
        agent_policy: Callable[[TaxiState], int],
        num_episodes: int
        ):
        """
        Compare RL agent vs shortest-path baseline over multiple episodes.
        For the baseline, compute the shortest path ignoring traffic parameters,
        then *execute* that path in the real environment (with traffic delays) to
        measure true travel + wait time.
        """
        rl_times = []
        sp_times = []
        path1_times = []
        path2_times = []
        keys = jrandom.split(key, num_episodes)
        for i, k in enumerate(keys):
            # Initialize state for this episode
            num_taxis = 10
            key, rng = jrandom.split(k)
            rng1, rng2 = jrandom.split(rng)
            starts = jnp.array(jrandom.choice(rng1, jnp.array(fixed_starts), shape=(num_taxis,)))
            pickups = jnp.array(jrandom.choice(rng2, jnp.array(fixed_pickups), shape=(num_taxis,)))
            state = self.init_state_fn(k, starts, pickups)

            # Compute shortest path between current_node and pickup_node
            current_idx = int(state.current_node)
            pickup_idx = int(state.pickup_node)
            shortest_path = self.compute_shortest_path(current_idx, pickup_idx)

            # --- RL policy simulation ---
            total_rl = 0.0
            count = 0
            print(f"RL: {i}, start: {state.current_node}, pickup: {state.pickup_node}")
            while not state.done:
                count += 1
                action = agent_policy(state)
                state, _, _, info = env.step(state, action)
                total_rl += float(info['travel'] + info['wait'])
                print(f"Step {count}: next node; {state.current_node}, action {action}, travel {info['travel']}, wait {info['wait']}")
            rl_times.append(total_rl)
            print(f"Rl count: {count}, total_rl: {total_rl}")

            # --- Shortest-path baseline simulation ---
            total_sp = 0.0

            # reset the taxi state
            state_sp = TaxiState(
                current_node=current_idx,
                pickup_node=pickup_idx,
                done=False,
                step_count=jnp.int32(0),
                neighbor_mask=env.neighbor_mask_static,
                time=jnp.array(0.0, dtype=jnp.float32)
            )


            # for each node in the path
            count = 0
            print(f"SP: {i}, start: {state_sp.current_node}, pickup: {state_sp.pickup_node}")
            for u in shortest_path[1:]:
                count += 1
                # find the index of the node in the adjacency list
                u_idx = self.node_to_idx[u]
                # find the action to take to get to that node
                nbrs = env.adj_list[state_sp.current_node]
                poss = jnp.where(nbrs == u_idx, size=1)[0]
                if len(poss) == 0:
                    raise ValueError(f"Node {u_idx} not in neighbors of {state_sp.current_node}")
                action = int(poss[0])

                # step and accumulate true (travel + wait)
                state_sp, _, _, info_sp = env.step(state_sp, action)
                print(f"SP Step {count}: next node; {state_sp.current_node}, action {action}, travel {info_sp['travel']}, wait {info_sp['wait']}")
                total_sp += float(info_sp['travel'] + info_sp['wait'])
            print(f"SP count: {count}, total_sp: {total_sp}")
            sp_times.append(total_sp)

            # reset the taxi state
            state_sp = TaxiState(
                current_node=current_idx,
                pickup_node=pickup_idx,
                done=False,
                step_count=jnp.int32(0),
                neighbor_mask=env.neighbor_mask_static,
                time=jnp.array(0.0, dtype=jnp.float32)
            )

            path1= [0,1,4,7]
            path2= [0,3,6,7]

            total_path1 = 0.0
            total_path2 = 0.0
            for u in path1[1:]:
                count += 1
                # find the index of the node in the adjacency list
                u_idx = self.node_to_idx[u]
                # find the action to take to get to that node
                nbrs = env.adj_list[state_sp.current_node]
                poss = jnp.where(nbrs == u_idx, size=1)[0]
                if len(poss) == 0:
                    raise ValueError(f"Node {u_idx} not in neighbors of {state_sp.current_node}")
                action = int(poss[0])

                # step and accumulate true (travel + wait)
                state_sp, _, _, info_sp = env.step(state_sp, action)
                print(f"SP Step {count}: next node; {state_sp.current_node}, action {action}, travel {info_sp['travel']}, wait {info_sp['wait']}")
                total_path1 += float(info_sp['travel'] + info_sp['wait'])
            print(f"SP count: {count}, total_sp: {total_path1}")
            path1_times.append(total_path1)

            state_sp = TaxiState(
                current_node=current_idx,
                pickup_node=pickup_idx,
                done=False,
                step_count=jnp.int32(0),
                neighbor_mask=env.neighbor_mask_static,
                time=jnp.array(0.0, dtype=jnp.float32)
            )
            total_sp = 0.0
            for u in path2[1:]:
                count += 1
                # find the index of the node in the adjacency list
                u_idx = self.node_to_idx[u]
                # find the action to take to get to that node
                nbrs = env.adj_list[state_sp.current_node]
                poss = jnp.where(nbrs == u_idx, size=1)[0]
                if len(poss) == 0:
                    raise ValueError(f"Node {u_idx} not in neighbors of {state_sp.current_node}")
                action = int(poss[0])

                # step and accumulate true (travel + wait)
                state_sp, _, _, info_sp = env.step(state_sp, action)
                print(f"SP Step {count}: next node; {state_sp.current_node}, action {action}, travel {info_sp['travel']}, wait {info_sp['wait']}")
                total_path2 += float(info_sp['travel'] + info_sp['wait'])
            print(f"SP count: {count}, total_sp: {total_path2}")
            path2_times.append(total_path2)

        return rl_times, sp_times