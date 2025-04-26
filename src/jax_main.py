import networkx as nx
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random as jax_random
import pickle
import random
import equinox as eqx
import geopandas as gpd

from jax_ppo_utils import build_adj_and_time_matrix, make_obs_fn
from jax_taxi_env import JAXRideEnv, init_env, TaxiState
# from jax_ppo_agent import make_agent
from jax_trainer import train
from nn import MLP

from utils import load_graph, apply_congestion_model, compute_zone_mappings

# --- Load and preprocess graph ---
place_name = "Manhattan, New York City, New York, USA"
zone_shp = "../data/processed/taxi_zones.shp"
G = load_graph(place_name)
apply_congestion_model(G)

# ---------- TEMP -----------------
locationID_to_nodes, zone_to_nodes, node_to_zone, nodes_gdf = compute_zone_mappings(G, zone_shp_path=zone_shp)

# Get the LocationID for the Financial District
zone_name = ["Financial District South", "Financial District North"] #, "Battery Park", "Seaport", "World Trade Center", "Battery Park City", "TriBeCa/Civic Center", "Chinatown", "Lower East Side", "East Village", "Little Italy/NoLiTa", "Two Bridges/Seward Park"]
gdf_zones = gpd.read_file(zone_shp).to_crs("EPSG:4326")
filtered_zones = gdf_zones[gdf_zones["zone"].isin(zone_name)]
loc_ids = filtered_zones["LocationID"].tolist()
# Now extract the node IDs
selected_nodes = [node for loc_id in loc_ids for node in zone_to_nodes.get(loc_id, [])] 
G_fds = G.subgraph(selected_nodes).copy()
largest_cc = max(nx.strongly_connected_components(G_fds), key=len)
G = G_fds.subgraph(largest_cc).copy()

all_nodes = list(G.nodes())
node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
idx_to_node = [node for node, idx in sorted(node_to_idx.items(), key=lambda x: x[1])]


# choose fixed start & pickup nodes
fixed_starts = jnp.array(np.array(random.sample(all_nodes, 1), dtype=np.int64))
fixed_pickups = jnp.array(np.array(random.sample(all_nodes, 1), dtype=np.int64))

def make_has_path_fn(G, idx_to_node):
    def has_path_fn(s_idx, t_idx):
        s = idx_to_node[s_idx]
        t = idx_to_node[t_idx]
        try:
            return nx.has_path(G, s, t)
        except:
            return False
    return has_path_fn

# --- Build JAX-ready graph structures ---
print("Building adjacency and travel time matrices")
adj_list, travel_times, neighbor_mask_static = build_adj_and_time_matrix(G, node_to_idx=node_to_idx)
# map fixed nodes to indices
fixed_starts_idx = [node_to_idx[int(n)] for n in fixed_starts if int(n) in node_to_idx]
fixed_pickups_idx = [node_to_idx[int(n)] for n in fixed_pickups if int(n) in node_to_idx]

# --- Precompute shortest-path distance matrix for reward shaping ---
print("Precomputing shortest-path distance matrix")
N = adj_list.shape[0]
dist_mat = np.zeros((N, N), dtype=np.float32)
for u, lengths in nx.all_pairs_dijkstra_path_length(G, weight="weight"):
    ui = node_to_idx[u]
    for v, d in lengths.items():
        vi = node_to_idx[v]
        dist_mat[ui, vi] = d
distances = jnp.array(dist_mat)

# --- Create environment ---
print("Creating environment")
env = JAXRideEnv(
    adj_list=adj_list,
    travel_times=travel_times,
    neighbor_mask_static=neighbor_mask_static,
    distances=distances,
    max_steps=82,
    fixed_starts=fixed_starts_idx,
    fixed_pickups=fixed_pickups_idx,
    timeout_penalty=-500.0
)

# --- Create normalized observation function ---
obs_fn, obs_fn_batch = make_obs_fn(G, node_to_idx, env.max_steps)


# --- Initialize batched environment state ---
num_envs = 16
key = jax_random.PRNGKey(0)
# split for agent init vs env init
eval_key, agent_key = jax_random.split(key)
# init agent
dim_obs = 5  # [current_lat, current_lon, pickup_lat, pickup_lon, timestep]
# agent = make_agent(agent_key, dim_obs, dim_act)

# init a single env state and update key
has_path_fn = make_has_path_fn(G, idx_to_node)
# TODO: the key should be split for each env
single_state, eval_key = init_env(eval_key, fixed_starts_idx, fixed_pickups_idx, neighbor_mask_static)

# shortest distance between pickup and dropoff
# print(f"Number of steps to pickup: {nx.dijkstra_path_length(G, single_state.current_node, single_state.pickup_node, weight='weight')}")

# tile fields to form a batch
def tile(x):
    x = jnp.array(x)
    return jnp.repeat(x[None, ...], num_envs, axis=0)

batched_state = TaxiState(
    current_node=tile(single_state.current_node),
    pickup_node=tile(single_state.pickup_node),
    # ride_phase=tile(single_state.ride_phase),
    step_count=tile(single_state.step_count),
    done=tile(single_state.done),
    neighbor_mask=tile(single_state.neighbor_mask)
)

print("Start training")

# --- Training ---
params = train(
    dim_obs=dim_obs,
    env=env,
    init_state_fn=lambda: batched_state,
    obs_fn=obs_fn,
    obs_fn_batch=obs_fn_batch,
    key=eval_key,
    fixed_starts=fixed_starts_idx,
    fixed_pickups=fixed_pickups_idx,
    num_steps=256,
    epochs=1000,
    batch_size=128,
    lr=3e-4,
    gamma=0.99,
    epsilon_start=0.2,
    epsilon_end=0.05,
)

# TODO: pickle dump of the params
# Save trained parameters
leaves, treedef = jax.tree_util.tree_flatten(params)
# Move arrays to CPU for pickling
leaves = [jax.device_get(x) for x in leaves]
with open("trained_params.pkl", "wb") as f:
    pickle.dump((leaves, treedef), f)
print("Saved trained parameters to trained_params.pkl")

# # --- Save trained agent ---
# leaves, treedef = eqx.tree_serialise_leaves(
#     pytree=trained_agent,
#     is_leaf=eqx.is_array
# )
# with open("test.eqx", "wb") as f:
#     # dump both parts so you can reconstruct later
#     pickle.dump((leaves, treedef), f)

# with open("test.eqx", "wb") as f:
#     serialized = eqx.tree_serialise_leaves(trained_agent, is_leaf=eqx.is_array)
#     pickle.dump(serialized, f)
