import argparse
import pickle
import networkx as nx
import numpy as np
import geopandas as gpd
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random as jax_random
from flax import linen as nn
import optax

from utils import load_graph, apply_congestion_model, compute_zone_mappings
from jax_ppo_utils import build_adj_and_time_matrix, make_obs_fn
from jax_taxi_env import TaxiState, JAXRideEnv
from jax_dqn_trainer import QNetwork


def generate_expert_data(env: JAXRideEnv,
                         n_samples: int,
                         dist_mat: jnp.ndarray,
                         key: jax_random.PRNGKey):
    """
    Simplistic pretraining sampler:
      1) Sample random starting nodes
      2) Sample a random valid neighbor action per node
      3) Compute base reward = dist(cur -> next)
      4) Sample random pickup node per sample
      5) Extra reward = dist(next -> pickup)
    Returns:
      curr_nodes [n_samples], actions [n_samples], rewards [n_samples], new key
    """
    adj_j      = env.adj_list     # [N, max_deg]
    mask_j     = env.neighbor_mask_static # [N, max_deg]
    dist_mat_j = dist_mat                           # [N, N]
    N, max_deg = adj_j.shape

    # sample current nodes
    key, subkey = jax_random.split(key)
    curr_nodes  = jax_random.randint(subkey, (n_samples,), 0, N, dtype=jnp.int32)

    # sample valid neighbor actions
    valid_mask  = mask_j[curr_nodes]                           # [n_samples, max_deg]
    probs       = valid_mask.astype(jnp.float32)
    probs      /= probs.sum(axis=1, keepdims=True)
    key, subkey = jax_random.split(key)
    actions     = jax_random.categorical(subkey, jnp.log(probs), axis=1).astype(jnp.int32)

    # base reward - travel time to next node
    next_nodes  = adj_j[curr_nodes, actions]                   # [n_samples]
    base_r      = - env.travel_times[curr_nodes, next_nodes]           # [n_samples]

    # random pickup nodes
    key, subkey = jax_random.split(key)
    pickups     = jax_random.randint(subkey, (n_samples,), 0, N, dtype=jnp.int32)

    # extra reward
    extra_r     = dist_mat_j[curr_nodes, pickups] - dist_mat_j[next_nodes, pickups]
    rewards     = base_r + extra_r                             # [n_samples]

    # Add pickup bonus if next_nodes == pickups
    pickup_bonus = 50.0  # Example value for the bonus
    rewards = jnp.where(next_nodes == pickups, rewards + pickup_bonus, rewards)

    return curr_nodes, actions, rewards, next_nodes, pickups, key


def main():
    parser = argparse.ArgumentParser(description="Pretrain DQN using random expert")
    parser.add_argument("--place", type=str, default="Manhattan, New York City, New York, USA")
    parser.add_argument("--n_expert_samples", type=int, default=10000)
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[256, 256])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--timeout_penalty", type=float, default=-5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="pretrained_q_params.pkl")
    parser.add_argument("--gamma",  type=float, default=0.99, help="Discount factor for rewards")
    args = parser.parse_args()

    # Load and preprocess graph
    G = load_graph(args.place)
    apply_congestion_model(G)
    zone_shp = "../data/processed/taxi_zones.shp"
    locationID_to_nodes, zone_to_nodes, node_to_zone, nodes_gdf = compute_zone_mappings(G, zone_shp_path=zone_shp)

    zone_name = ["Financial District South" , "Financial District North", "Battery Park", "Seaport", "World Trade Center", "Battery Park City", "TriBeCa/Civic Center", "Chinatown", "Lower East Side", "East Village", "Little Italy/NoLiTa", "Two Bridges/Seward Park"]
    gdf_zones = gpd.read_file(zone_shp).to_crs("EPSG:4326")
    filtered_zones = gdf_zones[gdf_zones["zone"].isin(zone_name)]
    loc_ids = filtered_zones["LocationID"].tolist()
    # Now extract the node IDs
    selected_nodes = [node for loc_id in loc_ids for node in zone_to_nodes.get(loc_id, [])] 
    G = G.subgraph(selected_nodes).copy()

    # Build node mappings and full distance matrix
    all_nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    N = len(all_nodes)
    dist_mat = np.zeros((N, N), dtype=np.float32)
    for u, lengths in nx.all_pairs_dijkstra_path_length(G, weight='weight'):
        i = node_to_idx[u]
        for v, d in lengths.items():
            j = node_to_idx[v]
            dist_mat[i, j] = d

    # Build adjacency, travel_times, and neighbor_mask using node_to_idx
    adj_list, travel_times, neighbor_mask_static = build_adj_and_time_matrix(
        G, node_to_idx=node_to_idx
    )

    # Create JAX environment
    env = JAXRideEnv(
        adj_list=adj_list,
        travel_times=travel_times,
        neighbor_mask_static=neighbor_mask_static,
        fixed_starts=np.array(list(node_to_idx.values()), dtype=np.int32),
        fixed_pickups=np.array(list(node_to_idx.values()), dtype=np.int32),
        distances=jnp.array(dist_mat),
        max_steps=args.max_steps,
        timeout_penalty=args.timeout_penalty
    )

    # Build observation functions
    obs_fn, obs_fn_batch = make_obs_fn(G, node_to_idx, args.max_steps)

    # Initialize model & optimizer
    rng = jax_random.PRNGKey(args.seed)
    dummy_state = TaxiState(
        current_node=jnp.array([0]),
        pickup_node=jnp.array([0]),
        done=jnp.array([False]),
        step_count=jnp.array([0]),
        neighbor_mask=jnp.array(neighbor_mask_static, dtype=bool)[0:1]
    )
    dummy_obs = obs_fn(dummy_state)
    model = QNetwork(dim_hidden=args.hidden_dims, num_actions=neighbor_mask_static.shape[-1])
    rng, init_key = jax_random.split(rng)
    params = model.init(init_key, dummy_obs)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    # Generate pretraining data
    curr_nodes, batch_actions, batch_rewards, next_nodes, pickups, rng = \
        generate_expert_data(env, args.n_expert_samples, env.distances, rng)

    pickup_nodes  = jnp.zeros_like(curr_nodes)
    done_flags    = jnp.zeros_like(curr_nodes, dtype=bool)
    step_counts   = jnp.zeros_like(curr_nodes, dtype=jnp.int32)
    masks_flat    = jnp.array(env.neighbor_mask_static, dtype=bool)[curr_nodes]
    batch_states  = TaxiState(
        current_node=curr_nodes,
        pickup_node=pickup_nodes,
        done=done_flags,
        step_count=step_counts,
        neighbor_mask=masks_flat
    )

    # Compute observations
    obs_all = obs_fn_batch(batch_states)
    N_steps, dim_obs = obs_all.shape
    num_actions = neighbor_mask_static.shape[-1]

       # build next-state batch for bootstrapping
    next_masks = jnp.array(env.neighbor_mask_static, dtype=bool)[next_nodes]
    next_states = TaxiState(
        current_node=next_nodes,
        pickup_node=pickups,
        done=jnp.zeros_like(next_nodes, dtype=bool),
        step_count=jnp.zeros_like(next_nodes, dtype=jnp.int32),
        neighbor_mask=next_masks
    )
    next_obs_all = obs_fn_batch(next_states)

    @jax.jit
    def train_step(params, opt_state, obs_b, act_b, rew_b, next_obs_b):
        def loss_fn(params):
            # current Q-values
            q_all      = model.apply(params, obs_b)
            q_sa       = jnp.take_along_axis(q_all, act_b[:, None], axis=1).squeeze(1)

            # bootstrap from next-state
            q_next_all = model.apply(params, next_obs_b)
            max_q_next = jnp.max(q_next_all, axis=1)
            target     = rew_b + args.gamma * max_q_next

            # TD loss
            return jnp.mean((q_sa - target) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        loss = loss_fn(new_params)
        return new_params, opt_state, loss

    n_batches = N_steps // args.batch_size
    @jax.jit
    def epoch_train(params, opt_state,
                    obs_all, acts_all, rews_all, next_obs_all, key):
        perm = jax_random.permutation(key, N_steps)
        perm = perm[:n_batches * args.batch_size].reshape((n_batches, args.batch_size))
        def batch_step(carry, batch_idx):
            p, o, tot = carry
            ob = obs_all[batch_idx]
            ac = acts_all[batch_idx]
            rw = rews_all[batch_idx]
            no = next_obs_all[batch_idx]
            p2, o2, l = train_step(p, o, ob, ac, rw, no)
            return (p2, o2, tot + l * args.batch_size), None
        (new_p, new_opt, total_loss), _ = jax.lax.scan(
            batch_step,
            (params, opt_state, 0.0),
            perm
        )
        return new_p, new_opt, total_loss / N_steps

    # Supervised pretraining loop
    print("Starting supervised pretraining...")
    for epoch in range(1, args.epochs + 1):
        rng, key = jax_random.split(rng)
        params, opt_state, avg_loss = epoch_train(
            params, opt_state,
            obs_all, batch_actions, batch_rewards, next_obs_all,
            key
        )
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

        # Compute Q-value statistics
        q_values = model.apply(params, obs_all)
        avg_q_value = jnp.mean(q_values)
        max_q_value = jnp.max(q_values)
        min_q_value = jnp.min(q_values)

        print(f"Q-Value Stats - Avg: {avg_q_value:.4f}, Max: {max_q_value:.4f}, Min: {min_q_value:.4f}")

    with open(args.output, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved pretrained parameters to {args.output}")

if __name__ == '__main__':
    main()
