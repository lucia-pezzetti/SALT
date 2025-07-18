import numpy as np
from typing import Dict, Tuple, List, Any
from taxi_env import TaxiEnv
from itertools import product
from jax import numpy as jnp

def brute_force_shortest_path(env: TaxiEnv, num_layers: int, width: int) -> List[Tuple[int, int]]:
    """
    Brute-force search for the shortest path for num_agents agents in the TaxiEnv.
    Returns a list of tuples (start_idx, pickup_idx) for each agent.
    """
    optimal_travel_data = {}
    paths = product(np.array(range(width)), repeat=num_layers)
    for path in paths:
        total = 0.0
        path = list(path)
        for i, u in enumerate(path[1:]):
            curr = path[i]
            a_idx = width*(i+1) + u
            # find the action to take to get to that node
            nbrs = env.adj_list[curr]
            poss = jnp.where(nbrs == a_idx, size=1)[0]
            action = int(poss[0])
            travel = float(env.travel_times[curr, action])
            total += travel
            nxt = int(env.adj_list[curr, action])
            cycle    = (total + env.offsets[nxt]) % env.periods[nxt]
            wait     = float(jnp.where(cycle < env.green_durations[nxt], 0.0, env.periods[nxt] - cycle))
            total += wait
            path[i+1] = nxt
        start, pickup = path[0], path[-1]
        key = (int(start), int(pickup))
        best = optimal_travel_data.get(key)
        if best is None or best["travel_time"] > total:
            optimal_travel_data[(start, pickup)] = {
                "travel_time": total,
                "path": path
            }
    return optimal_travel_data


def build_cost_matrix(
    optimal_travel_data: Dict[Tuple[int,int], Dict[str, Any]],
    starts: List[int],
    pickups: List[int]
) -> np.ndarray:
    """
    Construct a cost matrix C of shape (len(starts), len(pickups)) where
    C[i,j] = travel_time from starts[i] to pickups[j], as found in
    optimal_travel_data. If (start,pickup) is missing, uses missing_cost.
    """
    n, m = len(starts), len(pickups)
    C = np.zeros((n, m), dtype=float)
    for i, s in enumerate(starts):
        for j, p in enumerate(pickups):
            key = (int(s), int(p))
            entry = optimal_travel_data.get(key)
            C[i, j] = entry["travel_time"]
    return C