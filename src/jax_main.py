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
# from jax_trainer import train
from jax_dqn_trainer import train
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
zone_name = ["Financial District South" , "Financial District North", "Battery Park", "Seaport", "World Trade Center", "Battery Park City", "TriBeCa/Civic Center", "Chinatown", "Lower East Side", "East Village", "Little Italy/NoLiTa", "Two Bridges/Seward Park"]
gdf_zones = gpd.read_file(zone_shp).to_crs("EPSG:4326")
filtered_zones = gdf_zones[gdf_zones["zone"].isin(zone_name)]
loc_ids = filtered_zones["LocationID"].tolist()
# Now extract the node IDs
selected_nodes = [node for loc_id in loc_ids for node in zone_to_nodes.get(loc_id, [])] 
G = G.subgraph(selected_nodes).copy()

# remove nodes with only one in- and out-degree
to_remove = [v for v in G.nodes()
             if G.in_degree(v) == 1 and G.out_degree(v) == 1]
G.remove_nodes_from(to_remove)

largest_cc = max(nx.strongly_connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()


all_nodes = list(G.nodes())
print(f"Number of nodes in the graph: {len(all_nodes)}")
node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
idx_to_node = [node for node, idx in sorted(node_to_idx.items(), key=lambda x: x[1])]


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

# Convert to JAX arrays
fixed_starts = jnp.array(np.array(all_nodes, dtype=np.int64))
fixed_pickups = jnp.array(np.array(all_nodes, dtype=np.int64))


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

print(fixed_starts_idx)

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

print("example distances", distances[0, 1:10])

# --- Create environment ---
print("Creating environment")
env = JAXRideEnv(
    adj_list=adj_list,
    travel_times=travel_times,
    neighbor_mask_static=neighbor_mask_static,
    distances=distances,
    max_steps=128,
    fixed_starts=fixed_starts_idx,
    fixed_pickups=fixed_pickups_idx,
    timeout_penalty=-5.0
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
@jax.jit
def init_env_batch(keys, fixed_starts, fixed_pickups, neighbor_mask_static):
    return jax.vmap(init_env, in_axes=(0, None, None, None))(
        keys, fixed_starts, fixed_pickups, neighbor_mask_static
    )

keys = jax_random.split(eval_key, num_envs)
batched_states, eval_keys = init_env_batch(
    keys,
    jnp.array(fixed_starts_idx),
    jnp.array(fixed_pickups_idx),
    neighbor_mask_static
)

# Use batched_states directly to construct batched_state
batched_state = TaxiState(
    current_node=batched_states.current_node,
    pickup_node=batched_states.pickup_node,
    # ride_phase=batched_states.ride_phase,  # Uncomment if ride_phase is needed
    step_count=batched_states.step_count,
    done=batched_states.done,
    neighbor_mask=batched_states.neighbor_mask
)

# Split the eval_key into two separate keys
# key1, key2 = jax_random.split(eval_key)

# Use key1 for init_state_fn
# def init_state_fn():
#     keys = jax_random.split(key1, num_envs)
#     batched_states, _ = jax.vmap(init_env, in_axes=(0, None, None, None))(
#         keys, jnp.array(fixed_starts_idx), jnp.array(fixed_pickups_idx), neighbor_mask_static
#     )
#     return batched_states

# Upload pretrained parameters 
# with open("pretrained_q_params_pretrain2_negtime.pkl", "rb") as f:
#     pretrained_params = pickle.load(f)

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
    # init_q_params=pretrained_params,
    num_steps=256,
    epochs=100_000,
    batch_size=128,
    lr=1e-4,
    gamma=0.99,
    epsilon_start=0.3,
    epsilon_end=0.01,
)


# TODO: pickle dump of the params
# Save trained parameters
leaves, treedef = jax.tree_util.tree_flatten(params)
# Move arrays to CPU for pickling
leaves = [jax.device_get(x) for x in leaves]
with open("trained_params_all.pkl", "wb") as f:
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
