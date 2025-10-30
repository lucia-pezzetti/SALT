#!/usr/bin/env python3
import argparse
import pickle
import networkx as nx
import numpy as np
import jax
from jax import lax, random as jax_random
import jax.numpy as jnp
from flax import linen as nn
import optax
from functools import partial
import wandb

# project imports
from utils import build_env
from taxi_env_utils import build_adj_and_time_matrix, make_obs_fn
from taxi_env import TaxiState, TaxiEnv
from dqn_trainer import QNetwork


def compute_true_returns(start_nodes: jnp.ndarray,
                         pickups:     jnp.ndarray,
                         dist_mat:    jnp.ndarray,
                        ) -> jnp.ndarray:
    """
    Expert return: (1 - 1/60)*distance + bonus
    """
    return dist_mat[start_nodes, pickups]


@partial(jax.jit, static_argnames=('n_samples',))
def generate_expert_data(adj_list:      jnp.ndarray,
                         neighbor_mask: jnp.ndarray,
                         dist_mat:      jnp.ndarray,
                         travel_times:  jnp.ndarray,
                         key:           jax.random.PRNGKey,
                         n_samples:     int):
    """
    Samples expert transitions on-device.
    Returns (start, action, pickup, return, new_key)
    """
    N = adj_list.shape[0]
    key, k1, k2, k3 = jax_random.split(key, 4)

    # sample start nodes
    starts = jax_random.randint(k1, (n_samples,), 0, N)
    # sample random neighbor actions in [0, max_deg)
    max_deg = neighbor_mask.shape[-1]
    acts = jax_random.randint(k2, (n_samples,), 0, max_deg)
    # sample pickup destinations
    pickups = jax_random.randint(k3, (n_samples,), 0, N)
    # compute expert returns
    rets = compute_true_returns(starts, pickups, dist_mat)
    # sample valid neighbor actions for each start
    valid_mask = neighbor_mask[starts]  # [n_samples, max_deg]
    valid_counts = jnp.sum(valid_mask, axis=1, keepdims=True)
    probs = jnp.where(valid_mask,
                        1.0 / valid_counts,
                        0.0)  # uniform over valid actions
    acts = jax_random.categorical(k2, jnp.log(probs), axis=1)

    return starts, acts, pickups, rets, key


def main():
    parser = argparse.ArgumentParser(description='Pretrain DQN with expert data')
    # env settings
    parser.add_argument('--env_type', type=str, default='manhattan')
    parser.add_argument("--place_name", type=str, default="Manhattan, New York City, New York, USA", help="Place name for the graph (used for Manhattan)")
    parser.add_argument("--zone_shp", type=str, default="../data/processed/taxi_zones.shp", help="Path to the shapefile for zones (used for Manhattan)")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents in the environment")
    parser.add_argument('--offset', type=float, default=0.0)
    parser.add_argument('--cycle_length', type=int, default=0)
    # data & training
    parser.add_argument('--n_expert_samples', type=int, default=50000)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512,512])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--pickup_bonus', type=float, default=5.0)
    parser.add_argument('--timeout_penalty', type=float, default=-5.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='pretrained_params.pkl')
    args = parser.parse_args()

    # init W&B
    wandb.init(project='pretrain-taxi-dqn', config=vars(args))

    # build env
    G, node_to_idx, idx_to_node, fixed_starts, fixed_pickups, traffic_params = build_env(args)
    adj_list, travel_times, neighbor_mask = build_adj_and_time_matrix(G, node_to_idx=node_to_idx)
    N = len(G)

    # precompute distances via networkx
    dist_np = np.full((N, N), np.inf, dtype=np.float32)
    for u, lengths in nx.all_pairs_dijkstra_path_length(G, weight='travel_time_congested'):
        ui = node_to_idx[u]
        for v, d in lengths.items():
            vi = node_to_idx[v]
            dist_np[ui, vi] = d * 60.0  # Convert minutes to seconds
    distances = jax.device_put(jnp.array(dist_np))

    # device arrays
    adj_dev  = jax.device_put(jnp.array(adj_list, dtype=jnp.int32))
    tt_dev   = jax.device_put(jnp.array(travel_times, dtype=jnp.float32))
    mask_dev = jax.device_put(jnp.array(neighbor_mask, dtype=bool))

    # instantiate environment
    env = TaxiEnv(adj_list=adj_dev,
                  travel_times=tt_dev,
                  neighbor_mask_static=mask_dev,
                  fixed_starts=fixed_starts,
                  fixed_pickups=fixed_pickups,
                  distances=distances,
                  max_steps=args.max_steps,
                  traffic_params=traffic_params,
                  pickup_bonus=args.pickup_bonus,
                  timeout_penalty=args.timeout_penalty,
                  gamma=args.gamma)

    # obs fns
    obs_fn, obs_fn_batch = make_obs_fn(env, G, node_to_idx)

    # model & optimizer
    key = jax_random.PRNGKey(args.seed)
    key, subkey = jax_random.split(key)
    dummy_state = TaxiState(
        current_node=jnp.zeros((1,),dtype=jnp.int32),
        pickup_node=jnp.zeros((1,),dtype=jnp.int32),
        done=jnp.zeros((1,),dtype=bool),
        step_count=jnp.zeros((1,),dtype=jnp.int32),
        neighbor_mask=mask_dev[None],
        time=jnp.zeros((1,),dtype=jnp.float32)
    )
    s0, a0, g0 = obs_fn(dummy_state)
    m0 = dummy_state.neighbor_mask
    model = QNetwork(ctx_dim=128, node_hidden=64, pool='mean', num_actions=mask_dev.shape[-1])
    params = model.init(subkey, s0, a0, m0, g0)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    # generate expert dataset
    starts, acts, pickups, returns, key = generate_expert_data(
        adj_dev, mask_dev, distances, tt_dev, key,
        args.n_expert_samples)

    # prepare batch observations
    batch_states = TaxiState(
        current_node=starts,
        pickup_node=pickups,
        done=jnp.zeros_like(starts, dtype=bool),
        step_count=jnp.zeros_like(starts, dtype=jnp.int32),
        neighbor_mask=mask_dev[starts],
        time=jnp.zeros_like(starts, dtype=jnp.float32)
    )
    obs_all = obs_fn_batch(batch_states)
    mask_all = batch_states.neighbor_mask

    # single-step gradient on sampled batch
    @jax.jit
    def train_step(params, opt_state, obs_all, mask_all, acts, returns):
        def loss_fn(p):
            # compute all Q-values
            sf = obs_all['state_feats']       # [T, B, D_state]
            af = obs_all['action_feats']      # [T, B, max_deg, D_action]
            mask = mask_all                   # [T, B, max_deg]
            gf = obs_all['global_feats']      # [T, B, D_global]
            # reshape to (N, ...)
            T, B = sf.shape[0], sf.shape[1]
            N = T * B
            D_state = sf.shape[-1]
            max_deg = af.shape[-2]
            D_action = af.shape[-1]
            D_global = gf.shape[-1]
            sf = sf.reshape(-1, D_state)
            af = af.reshape(-1, max_deg, D_action)
            mask = mask.reshape(-1, max_deg)
            gf = gf.reshape(-1, D_global)
            # forward
            q_all = model.apply(p, sf, af, mask, gf)  # [N, max_deg]
            q_sa = jnp.take_along_axis(q_all, acts[:,None], 1).squeeze(1)
            return jnp.mean((q_sa - returns)**2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # training loop
    print('Starting supervised pretraining...')
    for epoch in range(1, args.epochs+1):
        params, opt_state, loss = train_step(params, opt_state, obs_all, mask_all, acts, returns)
        
        wandb.log({'pretrain/loss': float(loss)}, step=epoch)

    # save
    with open(args.output,'wb') as f:
        pickle.dump(params, f)
    wandb.save(args.output)
    print(f'Saved pretrained parameters to {args.output}')

if __name__ == '__main__':
    main()
