import argparse
import pickle
import networkx as nx
import numpy as np
import geopandas as gpd
import jax
from jax import random as jax_random
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
import optax
from functools import partial

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

from utils import load_graph, apply_congestion_model, compute_zone_mappings
from taxi_env_utils import build_adj_and_time_matrix, make_obs_fn
from taxi_env import TaxiState, JAXRideEnv
from dqn_trainer import QNetwork


def compute_true_returns(start_nodes: jnp.ndarray,
                          pickups:     jnp.ndarray,
                          dist_mat:    jnp.ndarray,
                          pickup_bonus: float = 50.0) -> jnp.ndarray:
    """
    Vectorized closed-form Monte Carlo return for shortest-path expert:
    G = (1 - 1/60) * dist[start, pickup] + pickup_bonus
    """
    d = dist_mat[start_nodes, pickups]
    return (1.0 - 1.0/60.0) * d + pickup_bonus


@partial(jax.jit, static_argnames=("n_samples",))
def generate_expert_data(adj_list:      jnp.ndarray,
                         neighbor_mask: jnp.ndarray,
                         dist_mat:      jnp.ndarray,
                         travel_times:  jnp.ndarray,
                         key:           jax.random.PRNGKey,
                         n_samples:     int,
                         pickup_bonus:  float = 50.0):
    """
    Samples expert transitions and returns entirely on-device.
    """
    N, max_deg = adj_list.shape
    key, k1, k2 = jax_random.split(key, 3)

    # Sample starting nodes
    start = jax_random.randint(k1, (n_samples,), 0, N)
    # Random neighbor action
    valid   = neighbor_mask[start]
    probs   = valid.astype(jnp.float32)
    probs   = probs / probs.sum(axis=1, keepdims=True)
    acts    = jax_random.categorical(k2, jnp.log(probs), axis=1).astype(jnp.int32)
    nextn   = adj_list[start, acts]

    # Sample pickup destinations
    key, k3 = jax_random.split(key)
    pickups = jax_random.randint(k3, (n_samples,), 0, N)

    # Compute full Monte Carlo returns
    returns = compute_true_returns(start, pickups, dist_mat, pickup_bonus)
    return start, acts, returns, nextn, pickups, key


def main():
    parser = argparse.ArgumentParser(description="Pretrain DQN using random expert")
    parser.add_argument("--place",               type=str,   default="Manhattan, New York City, New York, USA")
    parser.add_argument("--n_expert_samples",    type=int,   default=50000)
    parser.add_argument("--hidden_dims", nargs='+',      type=int,   default=[512, 512])
    parser.add_argument("--lr",                  type=float, default=3e-4)
    parser.add_argument("--epochs",              type=int,   default=100000)
    parser.add_argument("--batch_size",          type=int,   default=64)
    parser.add_argument("--max_steps",           type=int,   default=30)
    parser.add_argument("--timeout_penalty",     type=float, default=-5.0)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--output",              type=str,   default="pretrained_q_params_512.pkl")
    args = parser.parse_args()

    # 1) Load and filter graph
    G = load_graph(args.place)
    apply_congestion_model(G)
    zone_shp = "../data/processed/taxi_zones.shp"
    locationID_to_nodes, zone_to_nodes, node_to_zone, _ = compute_zone_mappings(G, zone_shp_path=zone_shp)

    selected_zones = ["Financial District South", "Financial District North", "Battery Park"]
    gdf_zones = gpd.read_file(zone_shp).to_crs("EPSG:4326")
    loc_ids   = gdf_zones[gdf_zones["zone"].isin(selected_zones)]["LocationID"].tolist()
    nodes     = [n for loc in loc_ids for n in zone_to_nodes.get(loc, [])]
    G         = G.subgraph(nodes).copy()

    # Prune low-degree nodes and take largest SCC
    to_remove = [v for v in G.nodes()
                 if (G.in_degree(v)==1 and G.out_degree(v)==1 and
                     list(G.predecessors(v))[0]==list(G.successors(v))[0])
                 or (G.in_degree(v)==1 and G.out_degree(v)==0)
                 or (G.in_degree(v)==0 and G.out_degree(v)==1)]
    G.remove_nodes_from(to_remove)
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    # Build index mappings
    all_nodes   = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    N           = len(all_nodes)

    # Compute full-pair distance matrix via NetworkX (host-side once)
    dist_mat = np.zeros((N, N), dtype=np.float32)
    for u, lengths in nx.all_pairs_dijkstra_path_length(G, weight='travel_time_congested'):
        i = node_to_idx[u]
        for v, d in lengths.items():
            j = node_to_idx[v]
            dist_mat[i, j] = d

    # Build adjacency list, travel times, neighbor mask (use keyword arg)
    adj_list, travel_times, neighbor_mask = build_adj_and_time_matrix(
        G,
        node_to_idx=node_to_idx
    )

    # Move arrays to device
    adj_list      = jax.device_put(jnp.array(adj_list,      dtype=jnp.int32))
    travel_times  = jax.device_put(jnp.array(travel_times,  dtype=jnp.float32))
    neighbor_mask = jax.device_put(jnp.array(neighbor_mask, dtype=bool))
    dist_mat_j    = jax.device_put(jnp.array(dist_mat,      dtype=jnp.float32))

    # Create environment
    env = JAXRideEnv(
        adj_list=adj_list,
        travel_times=travel_times,
        neighbor_mask_static=neighbor_mask,
        fixed_starts=jnp.arange(N, dtype=jnp.int32),
        fixed_pickups=jnp.arange(N, dtype=jnp.int32),
        distances=dist_mat_j,
        max_steps=args.max_steps,
        timeout_penalty=args.timeout_penalty
    )

    # Build observation functions
    obs_fn, obs_fn_batch = make_obs_fn(G, node_to_idx, args.max_steps)

    # Initialize model & optimizer
    rng = jax_random.PRNGKey(args.seed)
    dummy_state = TaxiState(
        current_node=jnp.zeros((1,), dtype=jnp.int32),
        pickup_node=jnp.zeros((1,), dtype=jnp.int32),
        done=jnp.zeros((1,), dtype=bool),
        step_count=jnp.zeros((1,), dtype=jnp.int32),
        neighbor_mask=neighbor_mask[0:1]
    )
    dummy_obs = obs_fn(dummy_state)
    model = QNetwork(dim_hidden=args.hidden_dims,
                     num_actions=neighbor_mask.shape[-1])
    rng, init_key = jax_random.split(rng)
    params        = model.init(init_key, dummy_obs)
    optimizer     = optax.adam(args.lr)
    opt_state     = optimizer.init(params)

    # Sample expert data ONCE
    curr, acts, true_returns, nextn, pickups, rng = generate_expert_data(
        adj_list, neighbor_mask, dist_mat_j, travel_times,
        rng, args.n_expert_samples, pickup_bonus=50.0
    )

    # Build batch of initial states
    masks_flat = neighbor_mask[curr]
    batch_states = TaxiState(
        current_node=curr,
        pickup_node=pickups,
        done=jnp.zeros_like(curr, dtype=bool),
        step_count=jnp.zeros_like(curr, dtype=jnp.int32),
        neighbor_mask=masks_flat
    )
    obs_all = obs_fn_batch(batch_states)

    # Training and stats functions
    @jax.jit
    def train_step(params, opt_state, obs_b, act_b, ret_b):
        def loss_fn(p):
            q_all = model.apply(p, obs_b)
            q_sa  = jnp.take_along_axis(q_all, act_b[:, None], 1).squeeze(1)
            return jnp.mean((q_sa - ret_b) ** 2)
        loss, grads    = jax.value_and_grad(loss_fn)(params)
        updates, opt_s = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_s, loss

    @jax.jit
    def epoch_train(params, opt_state, obs_all, acts, rets, key):
        N = obs_all.shape[0]
        key, subkey = jax_random.split(key)
        perm = jax_random.permutation(subkey, N)
        n_batches  = N // args.batch_size
        perm = perm[:n_batches * args.batch_size].reshape((n_batches, args.batch_size))

        def batch_step(carry, idx):
            p, o, tot = carry
            batch_idx = perm[idx]
            ob, ac, rt = obs_all[batch_idx], acts[batch_idx], rets[batch_idx]
            p2, o2, l = train_step(p, o, ob, ac, rt)
            return (p2, o2, tot + l * args.batch_size), None

        (new_p, new_opt, total_loss), _ = lax.scan(batch_step,
                                                  (params, opt_state, 0.0),
                                                  jnp.arange(n_batches))
        return new_p, new_opt, total_loss / N, key

    @jax.jit
    def q_stats(params, obs, acts, rets):
        q_all = model.apply(params, obs)
        q_sa  = jnp.take_along_axis(q_all, acts[:, None], 1).squeeze(1)
        avg_q, max_q, min_q = jnp.mean(q_all), jnp.max(q_all), jnp.min(q_all)
        avg_sa, max_sa, min_sa = jnp.mean(q_sa), jnp.max(q_sa), jnp.min(q_sa)
        mse  = jnp.mean((q_sa - rets) ** 2)
        corr = jnp.corrcoef(jnp.stack([q_sa, rets]))[0, 1]
        return avg_q, max_q, min_q, avg_sa, max_sa, min_sa, mse, corr

    @jax.jit
    def pretrain_step(params, opt_state, obs_all, acts, rets, key):
        p, o, loss, key = epoch_train(params, opt_state, obs_all, acts, rets, key)
        stats = q_stats(p, obs_all, acts, rets)
        return p, o, loss, stats, key

    # Supervised pretraining loop
    print("Starting supervised pretraining...")
    for epoch in range(1, args.epochs + 1):
        params, opt_state, loss, stats, rng = pretrain_step(
            params, opt_state, obs_all, acts, true_returns, rng
        )
        avg_q, max_q, min_q, avg_sa, max_sa, min_sa, mse, corr = stats
        # print(f"Epoch {epoch}/{args.epochs} — loss {loss:.4f} — "
        #       f"Q=[avg {avg_q:.2f}, min {min_q:.2f}, max {max_q:.2f}] — "
        #       f"Q_sa=[avg {avg_sa:.2f}, min {min_sa:.2f}, max {max_sa:.2f}] — "
        #       f"mse {mse:.4f}, corr {corr:.4f}")

    # Save parameters
    with open(args.output, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved pretrained parameters to {args.output}")


if __name__ == '__main__':
    main()
