import networkx as nx
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random as jax_random
from flax import linen as nn
import pickle
import equinox as eqx
import geopandas as gpd

from taxi_env_utils import build_adj_and_time_matrix, make_obs_fn, build_traffic_params
from taxi_env import JAXRideEnv, init_env, TaxiState
from dqn_trainer import train, QNetwork
from utils import load_graph, apply_congestion_model, compute_zone_mappings
from evaluation_utils import BaselineEvaluator

# --- Load and preprocess graph ---
place_name = "Manhattan, New York City, New York, USA"
zone_shp = "../data/processed/taxi_zones.shp"
G = load_graph(place_name)
apply_congestion_model(G)

# Map zones to nodes, then filter to Financial District
locationID_to_nodes, zone_to_nodes, node_to_zone, nodes_gdf = compute_zone_mappings(G, zone_shp_path=zone_shp)
zone_names = ["Financial District South", "Financial District North", "Battery Park"]
gdf_zones = gpd.read_file(zone_shp).to_crs("EPSG:4326")
filtered_zones = gdf_zones[gdf_zones["zone"].isin(zone_names)]
loc_ids = filtered_zones["LocationID"].tolist()
selected_nodes = [n for loc_id in loc_ids for n in zone_to_nodes.get(loc_id, [])]
G = G.subgraph(selected_nodes).copy()

# Prune trivial nodes
to_remove = [
    v for v in G.nodes()
    if (G.in_degree(v) == 1 and G.out_degree(v) == 1 and list(G.predecessors(v))[0] == list(G.successors(v))[0])
       or (G.in_degree(v) == 1 and G.out_degree(v) == 0)
       or (G.in_degree(v) == 0 and G.out_degree(v) == 1)
]
G.remove_nodes_from(to_remove)
largest_cc = max(nx.strongly_connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

all_nodes = list(G.nodes())
print(f"Number of nodes: {len(all_nodes)}")
node_to_idx = {n: i for i, n in enumerate(all_nodes)}
idx_to_node = [n for n, _ in sorted(node_to_idx.items(), key=lambda x: x[1])]

# # Choose fixed start & pickup sets (here: all nodes)
# fixed_starts_idx = list(range(len(all_nodes)))
# fixed_pickups_idx = list(range(len(all_nodes)))

# choose fixed start & pickup nodes
fixed_starts = []
fixed_pickups = []

nodes_gdf['zone'] = nodes_gdf.index.map(node_to_zone)
colored_nodes = nodes_gdf.dropna(subset=['zone'])

# Fix a node for every zone
for loc_id, nodes in zone_to_nodes.items():
    if nodes:
        # Use the first node in the list for each zone
        fixed_starts.append(nodes[0])
        fixed_pickups.append(nodes[0])

fixed_starts_idx = [node_to_idx[int(n)] for n in fixed_starts if int(n) in node_to_idx]
fixed_pickups_idx = [node_to_idx[int(n)] for n in fixed_pickups if int(n) in node_to_idx]
print(f"Fixed starts: {fixed_starts_idx}")
print(f"Fixed pickups: {fixed_pickups_idx}")


# --- Build JAX-ready graph structures ---
print("Building adjacency & time matrices")
adj_list, travel_times, neighbor_mask_static = build_adj_and_time_matrix(
    G, node_to_idx=node_to_idx
)

# --- Precompute shortest-path distance matrix for shaping ---
print("Precomputing distance matrix")
N = len(all_nodes)
dist_mat = np.zeros((N, N), dtype=np.float32)
for u, lengths in nx.all_pairs_dijkstra_path_length(G, weight="travel_time_congested"):
    ui = node_to_idx[u]
    for v, d in lengths.items():
        vi = node_to_idx[v]
        dist_mat[ui, vi] = d
distances = jnp.array(dist_mat)

traffic_params = build_traffic_params(G, node_to_idx, seed=42)

# --- Create environment ---
print("Instantiating environment")
env = JAXRideEnv(
    adj_list=adj_list,
    travel_times=travel_times,
    neighbor_mask_static=neighbor_mask_static,
    fixed_starts=fixed_starts_idx,
    fixed_pickups=fixed_pickups_idx,
    distances=distances,
    max_steps=128,
    traffic_params= traffic_params,
    timeout_penalty=-5.0
)

# --- Observation function (env-aware) ---
obs_fn_single, obs_fn_batch = make_obs_fn(env, G, node_to_idx, env.max_steps)


# --- Initialize batched environment states ---
num_envs = 16
key = jax_random.PRNGKey(0)
key1, key2 = jax_random.split(key)
key3, eval_key = jax_random.split(key1)

@jax.jit
def init_env_batch(keys, fixed_starts, fixed_pickups, neighbor_mask_static):
    states, _ = jax.vmap(init_env, in_axes=(0, None, None, None))(
        keys, fixed_starts, fixed_pickups, neighbor_mask_static
    )
    return states

keys = jax_random.split(key2, num_envs)
batched_states = init_env_batch(
    keys,
    jnp.array(fixed_starts_idx, dtype=jnp.int32),
    jnp.array(fixed_pickups_idx, dtype=jnp.int32),
    neighbor_mask_static
)

# --- Load pretrained parameters ---
with open("pretrained_q_params_512.pkl", "rb") as f:
    pretrained_params = pickle.load(f)

# --- Start training ---
print("Starting DQN training...")
params = train(
    env               = env,
    init_state_fn     = lambda: batched_states,
    obs_fn_batch      = obs_fn_batch,
    key               = eval_key,
    fixed_starts      = fixed_starts_idx,
    fixed_pickups     = fixed_pickups_idx,
    pretrain_ckpt     = "pretrained_q_params_512.pkl",
    num_steps         = 128,
    epochs            = 1_000,
    batch_size        = 128,
    lr                = 1e-4,
    gamma             = 0.99,
    epsilon_start     = 0.1,
    epsilon_end       = 0.01,
)

# # --- Save final parameters ---
# with open("trained_params_3zones_512.pkl", "wb") as f:
#     pickle.dump(params, f)

print("Training complete. Parameters saved to trained_params_test_3zones.pkl")

with open("trained_params_3zones_512.pkl", "rb") as f:
        params = pickle.load(f)


# --- Evaluate the trained agent against shortest path ---
def make_agent_policy(model, params, obs_fn_batch):
    """
    Wraps a Flax model into a single-state greedy policy using the batched obs_fn_batch.
    Builds a batch of size 1 for inference.
    """
    def agent_policy(state: TaxiState) -> int:
        # Build a batched TaxiState of size 1
        batch_state = TaxiState(
            current_node=state.current_node[None, ...],
            pickup_node=state.pickup_node[None, ...],
            done=state.done[None, ...],
            step_count=state.step_count[None, ...],
            neighbor_mask=state.neighbor_mask[None, ...],
            time=state.time[None, ...],
        )
        obs = obs_fn_batch(batch_state)
        sf = obs['state_feats'][0:1, :]    # [1, D_state]
        af = obs['action_feats'][0:1, ...]  # [1, max_deg, D_action]
        mask = batch_state.neighbor_mask     # [1, max_deg]
        q_vals = model.apply(params, sf, af, mask)  # [1, max_deg]
        return int(jnp.argmax(q_vals, axis=-1)[0])
    return agent_policy


def init_state_fn(key, starts, pickups):
    state, _ = init_env(key, starts, pickups, env.neighbor_mask_static)
    return state


model = QNetwork(max_deg=env.max_deg)

agent_policy = make_agent_policy(model, params, obs_fn_batch)

print("Evaluating agent policy against SP baseline...")
evaluator = BaselineEvaluator(G, node_to_idx, init_state_fn=init_state_fn, idx_to_node=idx_to_node)
# define agent_policy taking state -> action index via your Q-network
rl_times, sp_times = evaluator.evaluate(
    env, key3, fixed_starts_idx, fixed_pickups_idx,
    agent_policy, num_episodes=10
)

# Compare averages
print(f"RL avg time: {sum(rl_times)/len(rl_times):.2f}")
print(f"SP avg time: {sum(sp_times)/len(sp_times):.2f}")
