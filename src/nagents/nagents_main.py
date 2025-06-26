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
import matplotlib.pyplot as plt

from nagents_taxi_env_utils import build_adj_and_time_matrix, make_obs_fn
from nagents_taxi_env import TaxiEnv, init_env, TaxiState
from nagents_dqn_trainer import train, QNetwork, estimate_returns
from nagents_utils import build_env
from nagents_evaluation_utils import BaselineEvaluator
from nagents_random import run_random_policy
import argparse
from functools import partial
import numpy as onp
from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser(description="Ride-sharing Simulator")
parser.add_argument("--env_type", type=str, choices=["manhattan", "simple"], default="manhattan", help="Type of environment to use")
parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the customised grid environment")
parser.add_argument("--offset", type=float, default=0.0, help="Offset for the customised grid environment")
parser.add_argument("--cycle_length", type=int, default=200, help="Cycle length for the customised grid environment")
parser.add_argument("--place_name", type=str, default="Manhattan, New York City, New York, USA", help="Place name for the graph (used for Manhattan)")
parser.add_argument("--zone_shp", type=str, default="../../data/processed/taxi_zones.shp", help="Path to the shapefile for zones (used for Manhattan)")
parser.add_argument("--num_agents", type=int, default=10, help="Number of agents in the environment")
parser.add_argument("--base_time", type=float, default=1.0, help="Base travel time for grid environment")
parser.add_argument("--max_steps", type=int, default=128, help="Maximum number of steps per episode")
parser.add_argument("--pickup_bonus", type=float, default=10.0, help="Bonus for picking up a passenger")
parser.add_argument("--timeout_penalty", type=float, default=-5.0, help="Penalty for timeout")
parser.add_argument("--n_expert_samples", type=int, default=50000, help="Number of expert samples for pretraining")
parser.add_argument("--hidden_dims", nargs='+', type=int, default=[512, 512], help="Hidden dimensions for the neural network")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for training")
parser.add_argument("--epochs", type=int, default=10_000, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--num_steps", type=int, default=128, help="Number of steps for training")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for training")
parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon for epsilon-greedy policy")
parser.add_argument("--epsilon_end", type=float, default=0.1, help="Final epsilon for epsilon-greedy policy")
parser.add_argument("--output", type=str, default="pretrained_q_params_512.pkl", help="Output file for pretrained parameters")
parser.add_argument("--pretrain_ckpt", type=str, default="pretrained_params.pkl", help="Checkpoint file for pretrained parameters")
parser.add_argument("--mode", type=str, default="train", choices=["train", "random"])

args = parser.parse_args()

# --- Load or build the environment ---
G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params = build_env(args)

# --- Build graph structures ---
# print("Building adjacency & time matrices")
adj_list, travel_times, neighbor_mask_static = build_adj_and_time_matrix(
    G, node_to_idx=node_to_idx
)

# print(f"travel_times: {travel_times}, adj_list: {adj_list}, neighbor_mask_static: {neighbor_mask_static}")

# --- Precompute shortest-path distance matrix for shaping ---
# print("Precomputing distance matrix")
all_nodes = list(G.nodes())
N = len(all_nodes)
dist_mat = np.zeros((N, N), dtype=np.float32)
for u, lengths in nx.all_pairs_dijkstra_path_length(G, weight="travel_time_congested"):
    ui = node_to_idx[u]
    for v, d in lengths.items():
        vi = node_to_idx[v]
        dist_mat[ui, vi] = d
distances = jnp.array(dist_mat)

# --- Create environment ---
# print("Instantiating environment")
env = TaxiEnv(
    adj_list=adj_list,
    travel_times=travel_times,
    neighbor_mask_static=neighbor_mask_static,
    fixed_starts=fixed_starts_idx,
    fixed_pickups=fixed_pickups_idx,
    distances=distances,
    max_steps=args.max_steps,
    traffic_params= traffic_params,
    pickup_bonus=args.pickup_bonus,
    timeout_penalty=args.timeout_penalty,
    gamma=args.gamma,   
)

# --- Observation function ---
obs_fn_single, obs_fn_batch = make_obs_fn(env, G, node_to_idx)


# --- Initialize batched environment states for metrics logging and evaluation ---
num_envs = args.num_agents
key = jax_random.PRNGKey(0)
key1, key2 = jax_random.split(key)
all_keys = jax_random.split(key2, 2 * num_envs)
start_keys, pickup_keys = all_keys[:num_envs], all_keys[num_envs:]
key3, eval_key = jax_random.split(key1)

start_idxs = jnp.stack([jax_random.choice(k, fixed_starts_idx) for k in start_keys])
pickup_idxs = jnp.stack([jax_random.choice(k, fixed_pickups_idx) for k in pickup_keys])

@jax.jit
def init_env_batch(keys, starts, pickups, neighbor_mask_static):
    # now vmap over *three* varying axes
    states, info = jax.vmap(
        init_env,
        in_axes=(0, 0, 0, None),
    )(keys, starts, pickups, neighbor_mask_static)
    return states

keys = jax_random.split(key3, num_envs)
batched_states = init_env_batch(
    keys,
    start_idxs.astype(jnp.int32),
    pickup_idxs.astype(jnp.int32),
    neighbor_mask_static,
)

# --- Start training ---
if args.mode == "random":
    run_random_policy(env)
elif args.mode == "train":
    # print("Starting DQN training...")
    params = train(
        env               = env,
        init_state_fn     = lambda: batched_states,
        obs_fn_batch      = obs_fn_batch,
        key               = eval_key,
        fixed_starts      = fixed_starts_idx,
        fixed_pickups     = fixed_pickups_idx,
        # pretrain_ckpt     = args.pretrain_ckpt,
        num_steps         = args.num_steps,
        epochs            = args.epochs,
        batch_size        = args.batch_size,
        lr                = args.lr,
        gamma             = args.gamma,
        epsilon_start     = args.epsilon_start,
        epsilon_end       = args.epsilon_end,
    )

    # --- Save final parameters ---
    with open("trained_params_last_manh.pkl", "wb") as f:
        pickle.dump(params, f)

    # print("Training complete. Parameters saved to trained_params_last_manh.pkl")

    with open("trained_params_last_manh.pkl", "rb") as f:
            params = pickle.load(f)

    def make_agent_policy(model, params, obs_fn_batch):
        """
        Wraps a Flax model into a single-state greedy policy using the batched obs_fn_batch.
        Correctly builds a batch of size 1 for inference.
        """
        def agent_policy(state: TaxiState) -> int:
            # Create a batched TaxiState of size 1 using tree_map
            batch_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)
            
            obs = obs_fn_batch(batch_state)
            sf = obs['state_feats']      # Already [1, D_state]
            af = obs['action_feats']     # Already [1, max_deg, D_action]
            gf = obs['global_feats']     # Already [1, D_global]
            mask = batch_state.neighbor_mask       # Already [1, max_deg]
            q_vals = model.apply(params, sf, af, mask, gf)
            return int(jnp.argmax(q_vals, axis=-1)[0])
        return agent_policy


    def init_state_fn(key, starts, pickups):
        state, _ = init_env(key, starts, pickups, env.neighbor_mask_static)
        return state


    D_action = 2
    D_global = 64
    # model = QNetwork(dim_ctx=128, dim_action=D_action, dim_global=D_global)
    # model = QNetwork(ctx_dim=128, node_hidden=64, pool="mean")
    model = QNetwork(ctx_dim=128, node_hidden=64, pool="mean", num_actions=env.max_deg)

    agent_policy = make_agent_policy(model, params, obs_fn_batch)

    # print("Evaluating agent policy against SP baseline...")
    # evaluator = BaselineEvaluator(G, node_to_idx, init_state_fn=init_state_fn, idx_to_node=idx_to_node)
    # # define agent_policy taking state -> action index via your Q-network
    # rl_times, sp_times = evaluator.evaluate(
    #     env, key3, fixed_starts_idx, fixed_pickups_idx,
    #     agent_policy, num_episodes=10
    # )

    # Compare averages
    # print(f"RL avg time: {sum(rl_times)/len(rl_times):.2f}")
    # print(f"SP avg time: {sum(sp_times)/len(sp_times):.2f}")

    B = args.num_agents  # number of simultaneous taxi–passenger pairs

    # 1) Sample one fixed batch of start & pickup indices
    key, subkey = jax.random.split(eval_key)
    all_keys    = jax.random.split(subkey, 2 * B)
    start_keys, pickup_keys = all_keys[:B], all_keys[B:]
    starts  = jnp.array([jax.random.choice(k, fixed_starts_idx)  for k in start_keys])
    # pickups = jnp.array([jax.random.choice(k, fixed_pickups_idx) for k in pickup_keys])


    batched_init = jax.jit(
            jax.vmap(
                partial(init_env, neighbor_mask_static=neighbor_mask_static),
                # in_axes=(0, None, None),
                in_axes=(0, 0, 0),     # split keys, start_idxs, pickup_idxs
                out_axes=(0, 0),
            )
        )

    # 2) Compute the RL‐matching via estimated returns (using your helper)
    def init_env_fn(starts: jnp.ndarray, pickups: jnp.ndarray):
            # make one new RNG‐key per trajectory
            rng_keys = jax.random.split(jax.random.PRNGKey(0), starts.shape[0])
            return batched_init(rng_keys, starts, pickups)

    R = estimate_returns(
        env, params, model, obs_fn_batch,
        init_env_fn,
        fixed_starts_idx, fixed_pickups_idx,
        rollout_steps=10,  # you can choose this horizon
        gamma=args.gamma
    )  # → shape [B, B]
    # print(R)

    # solve max‐weight assignment for RL
    _, rl_cols = linear_sum_assignment(-onp.array(R))
    rl_pickups = [fixed_pickups_idx[rl_cols[s]] for s in starts]
    # print(f"RL matching: {rl_pickups}")

    # 3) Compute the SP‐matching via true SP distances
    #    (distances is your [N,N] JAX array from main)
    dist_np = onp.array(distances)  # to numpy for indexing
    # build cost matrix D[i,j] = dist(start_i, pickup_j)
    D = dist_np[onp.array(fixed_starts_idx), :][:, onp.array(fixed_pickups_idx)]
    # print(f"Distance matrix D:\n{D}")
    _, sp_cols = linear_sum_assignment(D)
    sp_pickups = [fixed_pickups_idx[sp_cols[s]] for s in starts]
    # print(f"SP matching: {sp_pickups}")

    # 4) Define a shortest-path greedy policy
    def sp_policy(G, state: TaxiState) -> int:
        curr = int(state.current_node)
        goal = int(state.pickup_node)
        path = nx.shortest_path(G, curr, goal, weight='travel_time_congested')
        return path

    # 5) Rollout helper
    def eval_matching(starts, pickups, policy, G, node_to_idx) -> onp.ndarray:
        times = []
        for s, p in zip(onp.array(starts), onp.array(pickups)):
            # initialize a fresh single‐env state
            state = init_env(jax.random.PRNGKey(0), int(s), int(p), env.neighbor_mask_static)[0]
            total = 0.0
            if hasattr(policy, "__call__") and policy is agent_policy:
                # print(f"Using agent policy for RL matching: {s} -> {p}")
                while not bool(state.done):
                    a = policy(state) if hasattr(policy, "__call__") and policy is agent_policy else policy(state)
                    state, _, _, info = env.step(state, a)
                    total += float(info['travel'] + info['wait'])
                    # print(f"Current node: {state.current_node}, action: {a}, travel: {info['travel']}, wait: {info['wait']}\n\n")
            else:
                # use the SP policy
                # print(f"Using SP policy for SP matching: {s} -> {p}")
                path = policy(G, state)
                for u in path[1:]:
                    a_idx = node_to_idx[u]
                    # find the action to take to get to that node
                    nbrs = env.adj_list[state.current_node]
                    poss = jnp.where(nbrs == a_idx, size=1)[0]
                    action = int(poss[0])
                    # step and accumulate true (travel + wait)
                    state, _, _, info = env.step(state, action)
                    total += float(info['travel'] + info['wait'])
                    # print(f"Current node: {state.current_node}, action: {action}, travel: {info['travel']}, wait: {info['wait']}\n\n")
            times.append(total)
        return onp.array(times)

    # 6) Evaluate all four combinations
    rl_on_rl_times = eval_matching(starts,   rl_pickups, agent_policy, G, node_to_idx)
    sp_on_rl_times = eval_matching(starts,   rl_pickups, sp_policy, G, node_to_idx)
    rl_on_sp_times = eval_matching(starts,   sp_pickups, agent_policy, G, node_to_idx)
    sp_on_sp_times = eval_matching(starts,   sp_pickups, sp_policy, G, node_to_idx)

    # print("=== Matching & Policy Evaluation ===")
    print(f"RL matching + RL policy avg time: {rl_on_rl_times.mean():.2f}")
    print(f"RL matching + SP policy avg time: {sp_on_rl_times.mean():.2f}")
    print(f"SP matching + RL policy avg time: {rl_on_sp_times.mean():.2f}")
    print(f"SP matching + SP policy avg time: {sp_on_sp_times.mean():.2f}")
