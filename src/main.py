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

from taxi_env_utils import build_adj_and_time_matrix, make_obs_fn, build_traffic_params
from taxi_env import JAXRideEnv, init_env, TaxiState
from dqn_trainer import train, QNetwork
from utils import load_graph, fixed_starts_pickups, load_simple_graph
from evaluation_utils import BaselineEvaluator
import argparse

# --- Load and preprocess graph ---
def build_env(args):
    """
    Build graph G, node_to_idx mapping, fixed start/pickup indices.
    Returns: G, node_to_idx, fixed_starts_idx, fixed_pickups_idx
    """
    if args.env_type == 'manhattan':
        # --- Load and preprocess Manhattan graph ---
        G, nodes_gdf, node_to_zone, zone_to_nodes = load_graph(place_name = args.place_name, zone_shp = args.zone_shp)

        fixed_starts_idx, fixed_pickups_idx, node_to_idx, idx_to_node = fixed_starts_pickups(
            G, nodes_gdf, node_to_zone, zone_to_nodes, all = True
        )

    elif args.env_type == 'simple':
        G = load_simple_graph()
        # index mappings
        nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        idx_to_node = [n for n, _ in sorted(node_to_idx.items(), key=lambda x: x[1])]

        # only one fixed start (node 0) and one fixed pickup (last one)
        fixed_starts_idx = [node_to_idx[0]]
        fixed_pickups_idx = [node_to_idx[nodes[-1]]]

    else:
        raise ValueError(f"Unknown env_type: {args.env_type}")
    
    traffic_params = build_traffic_params(G, node_to_idx, seed=42)

    return G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params


parser = argparse.ArgumentParser(description="Ride-sharing Simulator")
parser.add_argument("--env_type", type=str, choices=["manhattan", "simple"], default="manhattan", help="Type of environment to use")
parser.add_argument("--place_name", type=str, default="Manhattan, New York City, New York, USA", help="Place name for the graph (used for Manhattan)")
parser.add_argument("--zone_shp", type=str, default="../data/processed/taxi_zones.shp", help="Path to the shapefile for zones (used for Manhattan)")
parser.add_argument("--rows", type=int, default=2, help="Number of rows for grid environment")
parser.add_argument("--cols", type=int, default=5, help="Number of columns for grid environment")
parser.add_argument("--diag", action="store_true", help="Allow diagonal connections in grid environment")
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
parser.add_argument("--pretrain_ckpt", type=str, default="pretrained_q_params_512.pkl", help="Checkpoint file for pretrained parameters")
parser.add_argument("--mode", type=str, default="train", choices=["train", "random"])

args = parser.parse_args()

G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params = build_env(args)

# --- Build JAX-ready graph structures ---
print("Building adjacency & time matrices")
adj_list, travel_times, neighbor_mask_static = build_adj_and_time_matrix(
    G, node_to_idx=node_to_idx
)

print(f"travel_times: {travel_times}, adj_list: {adj_list}, neighbor_mask_static: {neighbor_mask_static}")

# --- Precompute shortest-path distance matrix for shaping ---
print("Precomputing distance matrix")
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
print("Instantiating environment")
env = JAXRideEnv(
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

def run_random_policy(env, num_episodes=100, seed=0):
    key = jax.random.PRNGKey(seed)

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, key = env.reset(key)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            legal_actions = jnp.where(jnp.array(obs[4]))[0]
            key, subkey = jax.random.split(key)
            action_idx = jax.random.randint(subkey, (), 0, len(legal_actions))
            action = int(legal_actions[action_idx])

            obs, reward, done, _ = env.step(obs, action)

            total_reward += reward
            steps += 1

        jax.debug.print("Episode {}: Return = {:.2f}, Steps = {}",
                        ep+1, total_reward, steps)
        episode_returns.append(total_reward)
        episode_lengths.append(steps)

    return episode_returns, episode_lengths

# --- Start training ---
if args.mode == "random":
    run_random_policy(env)
elif args.mode == "train":
    print("Starting DQN training...")
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

    print("Training complete. Parameters saved to trained_params_last_manh.pkl")

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
