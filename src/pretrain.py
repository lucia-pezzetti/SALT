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
from jax.tree_util import tree_map
import matplotlib.pyplot as plt

# Enable 64-bit precision\anjax.config.update("jax_enable_x64", True)

from utils import build_env
from taxi_env_utils import build_adj_and_time_matrix, make_obs_fn
from taxi_env import TaxiState, JAXRideEnv
from dqn_trainer import QNetwork


def compute_true_returns(start_nodes: jnp.ndarray,
                         pickups:     jnp.ndarray,
                         dist_mat:    jnp.ndarray,
                         pickup_bonus: float = 50.0) -> jnp.ndarray:
    """
    Expert return: (1 - 1/60)*distance + bonus
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
    Samples expert transitions on-device:
    Returns (start, action, pickup, return, new_key)
    """
    N, _ = adj_list.shape
    key, k1, k2 = jax_random.split(key, 3)

    # sample start and random neighbor action
    start = jax_random.randint(k1, (n_samples,), 0, N)
    valid = neighbor_mask[start]
    probs = valid.astype(jnp.float32)
    probs = probs / probs.sum(axis=1, keepdims=True)
    acts = jax_random.categorical(k2, jnp.log(probs), axis=1).astype(jnp.int32)

    # sample pickup destinations
    key, k3 = jax_random.split(key)
    pickups = jax_random.randint(k3, (n_samples,), 0, N)

    # compute returns
    returns = compute_true_returns(start, pickups, dist_mat, pickup_bonus)
    return start, acts, pickups, returns, key


def main():
    parser = argparse.ArgumentParser(description="Pretrain DQN using random expert")
    parser.add_argument("--env_type", type=str, choices=["manhattan", "simple"], default="manhattan", help="Type of environment to use")
    parser.add_argument("--place_name", type=str, default="Manhattan, New York City, New York, USA")
    parser.add_argument("--zone_shp", type=str, default="../data/processed/taxi_zones.shp")
    parser.add_argument("--n_expert_samples", type=int, default=50000)
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[512, 512])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--timeout_penalty", type=float, default=-5.0)
    parser.add_argument("--pickup_bonus", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="pretrained_params.pkl")
    args = parser.parse_args()

    # build env and graph
    G, node_to_idx, _, fixed_starts, fixed_pickups, traffic_params = build_env(args)
    adj_list, travel_times, neighbor_mask = build_adj_and_time_matrix(
        G, node_to_idx=node_to_idx
    )
    N = len(G)

    # precompute distance matrix
    dist_mat = np.zeros((N, N), dtype=np.float32)
    for u, lengths in nx.all_pairs_dijkstra_path_length(G, weight='travel_time_congested'):
        ui = node_to_idx[u]
        for v, d in lengths.items():
            dist_mat[ui, node_to_idx[v]] = d
    distances = jax.device_put(jnp.array(dist_mat))

    # move arrays to device
    adj_dev  = jax.device_put(jnp.array(adj_list, dtype=jnp.int32))
    tt_dev   = jax.device_put(jnp.array(travel_times, dtype=jnp.float32))
    mask_dev = jax.device_put(jnp.array(neighbor_mask, dtype=bool))

    # init env
    env = JAXRideEnv(
        adj_list=adj_dev,
        travel_times=tt_dev,
        neighbor_mask_static=mask_dev,
        fixed_starts=jnp.array(fixed_starts, dtype=jnp.int32),
        fixed_pickups=jnp.array(fixed_pickups, dtype=jnp.int32),
        distances=distances,
        max_steps=args.max_steps,
        timeout_penalty=args.timeout_penalty,
        pickup_bonus=args.pickup_bonus,
        gamma=args.gamma,
        traffic_params=traffic_params
    )

    # build obs fns
    obs_fn, obs_fn_batch = make_obs_fn(env, G, node_to_idx)

    # init model & optimizer
    key = jax_random.PRNGKey(args.seed)
    rng, init_key = jax_random.split(key)
    dummy_state = TaxiState(
        current_node=jnp.zeros((1,), dtype=jnp.int32),
        pickup_node=jnp.zeros((1,), dtype=jnp.int32),
        done=jnp.zeros((1,), dtype=bool),
        step_count=jnp.zeros((1,), dtype=jnp.int32),
        neighbor_mask=mask_dev[None],
        time=jnp.zeros((1,), dtype=jnp.float32)
    )
    s_dummy, a_dummy, g_dummy = obs_fn(dummy_state)
    # mask dummy based on a_dummy shape
    m_dummy = jnp.ones_like(a_dummy[...,0], dtype=bool)
    model = QNetwork(ctx_dim=128, node_hidden=64, pool="mean", num_actions=mask_dev.shape[-1])
    params = model.init(init_key, s_dummy, a_dummy, m_dummy, g_dummy)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    # sample expert data once
    curr, acts, pickups, returns, rng = generate_expert_data(
        adj_dev, mask_dev, distances, tt_dev,
        rng, args.n_expert_samples, args.pickup_bonus
    )

    # build batch states
    batch_states = TaxiState(
        current_node=curr,
        pickup_node=pickups,
        done=jnp.zeros_like(curr),
        step_count=jnp.zeros_like(curr),
        neighbor_mask=mask_dev[curr],
        time=jnp.zeros_like(curr, dtype=jnp.float32)
    )
    obs_all = obs_fn_batch(batch_states)          # tuple of (s_feats, a_feats, g_feats)
    mask_all = batch_states.neighbor_mask         # shape [B, max_deg]

    # loss history
    loss_history = []

    @jax.jit
    def train_step(params, opt_state, s_feats, a_feats, g_feats, mask, act_b, ret_b):
        def loss_fn(p):
            q_all = model.apply(p, s_feats, a_feats, mask, g_feats)
            q_sa  = jnp.take_along_axis(q_all, act_b[:, None], 1).squeeze(1)
            return jnp.mean((q_sa - ret_b) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), new_opt, loss

    @jax.jit
    def epoch_train(params, opt_state, obs_all, mask_all, acts, rets, key):
        s_all = obs_all["state_feats"]
        a_all = obs_all["action_feats"]
        g_all = obs_all["global_feats"]
        B = s_all.shape[0]
        key, subkey = jax_random.split(key)
        perm = jax_random.permutation(subkey, B)
        nbatches = B // args.batch_size
        perm = perm[:nbatches * args.batch_size].reshape((nbatches, args.batch_size))

        def batch_step(carry, idx):
            p, o, tot = carry
            idxs = perm[idx]
            s_b = s_all[idxs]
            a_b = a_all[idxs]
            g_b = g_all[idxs]
            m_b = mask_all[idxs]
            act_b = acts[idxs]
            ret_b = rets[idxs]
            p2, o2, l = train_step(p, o, s_b, a_b, g_b, m_b, act_b, ret_b)
            return (p2, o2, tot + l * args.batch_size), None

        (new_p, new_opt, total_loss), _ = lax.scan(
            batch_step,
            (params, opt_state, 0.0),
            jnp.arange(nbatches)
        )
        return new_p, new_opt, total_loss / B, key

    @jax.jit
    def q_stats(params, obs_all, mask_all, acts, rets):
        s_all = obs_all["state_feats"]
        a_all = obs_all["action_feats"]
        g_all = obs_all["global_feats"]
        q_all = model.apply(params, s_all, a_all, mask_all, g_all)
        q_sa  = jnp.take_along_axis(q_all, acts[:, None], 1).squeeze(1)
        mse = jnp.mean((q_sa - rets) ** 2)
        corr = jnp.corrcoef(jnp.stack([q_sa, rets]))[0, 1]
        return mse, corr

    @jax.jit
    def pretrain_step(params, opt_state, obs_all, mask_all, acts, rets, key):
        p, o, loss, key = epoch_train(params, opt_state, obs_all, mask_all, acts, rets, key)
        stats = q_stats(p, obs_all, mask_all, acts, rets)
        return p, o, loss, stats, key

    # training loop
    print("Starting supervised pretraining...")
    for epoch in range(1, args.epochs + 1):
        params, opt_state, loss, stats, rng = pretrain_step(
            params, opt_state, obs_all, mask_all, acts, returns, rng
        )
        loss_history.append(float(loss))
        if epoch % 1000 == 0:
            mse, corr = stats
            print(f"Epoch {epoch}/{args.epochs} loss={loss:.4f} mse={mse:.4f} corr={corr:.4f}")

    # plot loss
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.show()

    # save params
    with open(args.output, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved pretrained parameters to {args.output}")

if __name__ == '__main__':
    main()
