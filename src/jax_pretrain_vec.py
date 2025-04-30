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


def build_env(G: nx.Graph, max_steps: int = 100, timeout_penalty: float = -50.0):
    # Map nodes to indices
    all_nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    # Build adjacency, travel_times, and masks
    adj_list, travel_times, neighbor_mask_static = build_adj_and_time_matrix(
        G, node_to_idx=node_to_idx
    )

    # Fixed sets for starts and pickups: one per node
    fixed = np.array(list(node_to_idx.values()), dtype=np.int32)

    # Precompute shortest-path distances for shaping
    N = len(node_to_idx)
    dist_mat = np.zeros((N, N), dtype=np.float32)
    for u, lengths in nx.all_pairs_dijkstra_path_length(G, weight='length'):
        ui = node_to_idx[u]
        for v, d in lengths.items():
            vi = node_to_idx[v]
            dist_mat[ui, vi] = d
    distances = jnp.array(dist_mat)

    env = JAXRideEnv(
        adj_list=adj_list,
        travel_times=travel_times,
        neighbor_mask_static=neighbor_mask_static,
        fixed_starts=fixed,
        fixed_pickups=fixed,
        distances=distances,
        max_steps=max_steps,
        timeout_penalty=timeout_penalty
    )
    return env, node_to_idx, idx_to_node


def generate_expert_data(env: JAXRideEnv,
                         n_samples: int,
                         dist_mat_np: np.ndarray,
                         key: jax.random.PRNGKey):
    """
    Vectorized expert data generation:
    - Precompute an oracle policy table `next_hop_idx[u, v]` giving the best neighbor index
    - Sample start & destination nodes in batch
    - Roll out greedy trajectories via `jax.lax.scan`
    - Flatten into batched states/actions
    - Compute returns with a vectorized, JIT-compiled function
    """
    # Convert environment data to JAX arrays
    adj_j = jnp.array(env.adj_list, dtype=jnp.int32)            # [N, max_deg]
    mask_j = jnp.array(env.neighbor_mask_static, dtype=bool)     # [N, max_deg]
    dist_mat = jnp.array(dist_mat_np)                            # [N, N]
    fixed = jnp.array(env.fixed_starts, dtype=jnp.int32)         # [F]

    N, max_deg = adj_j.shape
    max_hop = env.max_steps

    # 1) Oracle policy: next_hop_idx[u, v] -> neighbor index
    #    Compute distances of all neighbors to every destination at once
    v_idx = jnp.arange(N)[None, None, :]                         # [1, 1, N]
    neigh_idx = adj_j[:, :, None]                                # [N, max_deg, 1]
    dists = dist_mat[neigh_idx, v_idx]                          # [N, max_deg, N]
    dists = jnp.where(mask_j[:, :, None], dists, jnp.inf)
    next_hop_idx = jnp.argmin(dists, axis=1)                    # [N, N]

    # 2) Sample start & destination nodes
    key, k1, k2 = jax_random.split(key, 3)
    idx1 = jax_random.randint(k1, (n_samples,), 0, fixed.shape[0])
    idx2 = jax_random.randint(k2, (n_samples,), 0, fixed.shape[0])
    start_nodes = fixed[idx1]                                    # [n_samples]
    dest_nodes  = fixed[idx2]                                    # [n_samples]

    # 3) Roll out greedy policy with scan
    def step_fn(carry, _):
        curr, dst = carry                                        # both [n_samples]
        act = next_hop_idx[curr, dst]                            # best neighbor index [n_samples]
        next_node = adj_j[curr, act]                             # next node id [n_samples]
        return (next_node, dst), (curr, dst, act)

    # scan length = max_hop steps
    (final_carry, _), (curr_seq, dst_seq, act_seq) = jax.lax.scan(
        step_fn,
        (start_nodes, dest_nodes),
        None,
        length=max_hop
    )

    # curr_seq: [max_hop, n_samples], dst_seq same, act_seq same
    H, S = max_hop, n_samples
    nodes_flat = curr_seq.reshape(H * S)
    dest_flat  = dst_seq.reshape(H * S)
    acts_flat  = act_seq.reshape(H * S)
    step_counts = jnp.tile(jnp.arange(H), S)
    masks_flat  = mask_j[nodes_flat]

    # Build batched states & actions
    done_flat    = jnp.zeros_like(nodes_flat, dtype=bool)
    batch_states  = TaxiState(nodes_flat, dest_flat, done_flat, step_counts, masks_flat)
    batch_actions = acts_flat

    # 4) Compute returns vectorized
    @jax.jit
    def compute_return_jit(u, v):
        def cond_fn(carry):
            state, _ = carry
            return ~state.done
        def body_fn(carry):
            state, acc = carry
            nbrs  = adj_j[state.current_node]                    # [max_deg]
            valid = mask_j[state.current_node]                   # [max_deg]
            d     = dist_mat[nbrs, v]                            # [max_deg]
            d     = jnp.where(valid, d, jnp.inf)
            best  = jnp.argmin(d)                                # scalar
            state_next_raw, r, _ = env.step(state, best)
            nm = mask_j[state_next_raw.current_node]
            state_next = TaxiState(
                state_next_raw.current_node,
                state_next_raw.pickup_node,
                state_next_raw.done,
                state_next_raw.step_count,
                nm
            )
            return (state_next, acc + r)
        init_state = TaxiState(u, v, False, 0, mask_j[u])
        _, total_ret = jax.lax.while_loop(cond_fn, body_fn, (init_state, 0.0))
        return total_ret

    batch_returns = jax.vmap(compute_return_jit)(nodes_flat, dest_flat)

    return batch_states, batch_actions, batch_returns, key


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain DQN using shortest-path expert"
    )
    parser.add_argument("--place", type=str, default="Manhattan, New York City, New York, USA",
                        help="Place identifier for load_graph")
    parser.add_argument("--n_expert_samples", type=int, default=1000,
                        help="Number of expert transitions to generate")
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[256, 256],
                        help="Q-network hidden sizes")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Number of supervised pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for pretraining")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--max_steps", type=int, default=82,
                        help="Max steps per episode in environment")
    parser.add_argument("--timeout_penalty", type=float, default=-5.0,
                        help="Timeout penalty in environment reward shaping")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Mixing weight: alpha*CE + (1-alpha)*MSE")
    parser.add_argument("--output", type=str, default="pretrained_q_params.pkl",
                        help="Output path for pretrained params")
    args = parser.parse_args()

    # Load and preprocess graph
    G = load_graph(args.place)
    apply_congestion_model(G)

    # zone_shp = "../data/processed/taxi_zones.shp"
    # locationID_to_nodes, zone_to_nodes, node_to_zone, nodes_gdf = compute_zone_mappings(G, zone_shp_path=zone_shp)
    # zone_name = ["Financial District South" , "Financial District North", "Battery Park", "Seaport", "World Trade Center", "Battery Park City", "TriBeCa/Civic Center", "Chinatown", "Lower East Side", "East Village", "Little Italy/NoLiTa", "Two Bridges/Seward Park"]
    # gdf_zones = gpd.read_file(zone_shp).to_crs("EPSG:4326")
    # filtered_zones = gdf_zones[gdf_zones["zone"].isin(zone_name)]
    # loc_ids = filtered_zones["LocationID"].tolist()
    # # Now extract the node IDs
    # selected_nodes = [node for loc_id in loc_ids for node in zone_to_nodes.get(loc_id, [])] 
    # G = G.subgraph(selected_nodes).copy()
    to_remove = [v for v in G.nodes() if G.in_degree(v)==1 and G.out_degree(v)==1]
    G.remove_nodes_from(to_remove)
    cc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(cc).copy()

    key = jax_random.PRNGKey(args.seed)

    # Build environment
    env, node_to_idx, idx_to_node = build_env(
        G, max_steps=args.max_steps, timeout_penalty=args.timeout_penalty
    )
    obs_fn, obs_fn_batch = make_obs_fn(G, node_to_idx, args.max_steps)

    # Generate expert data
    batch_states, batch_actions, batch_returns, rng = generate_expert_data(
        env, args.n_expert_samples, env.distances, key
    )

    # Compute observations
    obs_all = obs_fn_batch(batch_states)
    N_steps, dim_obs = obs_all.shape
    num_actions = env.neighbor_mask_static.shape[-1]

    # Initialize Q-network & optimizer
    model = QNetwork(dim_hidden=args.hidden_dims, num_actions=num_actions)
    rng, init_rng = jax_random.split(rng)
    params = model.init(init_rng, jnp.zeros((1, dim_obs)))
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    # Single-step train function
    @jax.jit
    def train_step(params, opt_state, obs_b, act_b, ret_b):
        logits = model.apply(params, obs_b)                           # [B, A]
        ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            logits, act_b))                                         # scalar
        qv = model.apply(params, obs_b)
        q_taken = jnp.take_along_axis(qv, act_b[:,None], axis=1).squeeze()
        mse = jnp.mean((q_taken - ret_b)**2)
        loss = args.alpha * ce + (1-args.alpha) * mse
        grads = jax.grad(lambda p: args.alpha * jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                model.apply(p, obs_b), act_b))
            + (1-args.alpha) * jnp.mean((
                jnp.take_along_axis(model.apply(p, obs_b), act_b[:,None], axis=1).squeeze()
                - ret_b
            )**2)
        )(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Compile full epoch via lax.scan
    n_batches = N_steps // args.batch_size

    @jax.jit
    def epoch_train(params, opt_state, obs_all, acts_all, ret_all, key):
        perm = jax.random.permutation(key, N_steps)
        perm = perm[:n_batches * args.batch_size].reshape((n_batches, args.batch_size))
        def batch_step(carry, batch_idx):
            p, o, tot = carry
            ob = obs_all[batch_idx]
            ac = acts_all[batch_idx]
            rv = ret_all[batch_idx]
            p, o, l = train_step(p, o, ob, ac, rv)
            return (p, o, tot + l * args.batch_size), None
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
            params, opt_state, obs_all, batch_actions, batch_returns, key
        )
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

    # Save pretrained parameters
    with open(args.output, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved pretrained parameters to {args.output}")

if __name__ == '__main__':
    main()
