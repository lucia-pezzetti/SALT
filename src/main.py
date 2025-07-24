import networkx as nx
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jax_random
import wandb
import json
from tqdm import tqdm

from taxi_env_utils import build_adj_and_time_matrix, make_obs_fn
from taxi_env import TaxiEnv, init_env, TaxiState
from dqn_trainer import train
from models.q_network import QNetwork, QNetworkSimple, QNetworkUntied
from utils import build_env, estimate_returns_jit, EstimateReturnsState
from brute_force import brute_force_shortest_path, build_cost_matrix
from policy_improvement import get_init_fn, get_recurrent_fn, get_agent_loop, estimate_returns_batch

import argparse
from functools import partial
import numpy as onp
from scipy.optimize import linear_sum_assignment
import optax
import pickle

parser = argparse.ArgumentParser(description="Ride-sharing Simulator")
parser.add_argument("--env_type", type=str, choices=["manhattan", "simple"], default="manhattan", help="Type of environment to use")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the customised grid environment")
parser.add_argument("--layer_width", type=int, default=3, help="Width of each layer in the customised grid environment")
parser.add_argument("--offset", type=float, default=0.0, help="Offset for the customised grid environment")
parser.add_argument("--cycle_length", type=int, default=200, help="Cycle length for the customised grid environment")
parser.add_argument("--no_congestion", type=bool, default=False, help="different types of roads have different congestion levels")
parser.add_argument("--place_name", type=str, default="Manhattan, New York City, New York, USA", help="Place name for the graph (used for Manhattan)")
parser.add_argument("--zone_shp", type=str, default="../data/processed/taxi_zones.shp", help="Path to the shapefile for zones (used for Manhattan)")
parser.add_argument("--num_agents", type=int, default=1, help="Number of agents in the environment")
parser.add_argument("--base_time", type=float, default=1.0, help="Base travel time for grid environment")
parser.add_argument("--max_steps", type=int, default=128, help="Maximum number of steps per episode")
parser.add_argument("--pickup_bonus", type=float, default=10.0, help="Bonus for picking up a passenger")
parser.add_argument("--timeout_penalty", type=float, default=-5.0, help="Penalty for timeout")
parser.add_argument("--n_expert_samples", type=int, default=50000, help="Number of expert samples for pretraining")
parser.add_argument("--hidden_dims", nargs='+', type=int, default=[512, 512], help="Hidden dimensions for the neural network")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for training")
parser.add_argument("--epochs", type=int, default=1_000, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--num_steps", type=int, default=128, help="Number of steps for training")
parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor for training")
parser.add_argument("--epsilon_start", type=float, default=0.5, help="Initial epsilon for epsilon-greedy policy")
parser.add_argument("--epsilon_end", type=float, default=0.1, help="Final epsilon for epsilon-greedy policy")
parser.add_argument("--params_dir", type=str, default=None, help="Output file for pretrained parameters")
parser.add_argument("--pretrain_ckpt", type=str, default="pretrained_params.pkl", help="Checkpoint file for pretrained parameters")
parser.add_argument("--model", type=str, default="pi", choices=["dqn", "pi"])
parser.add_argument("--pi_config", "-c", type=str, default="config.json", help="Path to pi configuration file")
parser.add_argument("--wandb_project", type=str, default="taxi-pi", help="WandB project name")
parser.add_argument("--use_untied", type=bool, default=True, help="untied heads for Q-network")

args = parser.parse_args()

if args.params_dir is None:
    args.params_dir = (
        f"{args.env_type}"
        f"_layers{args.num_layers}"
        f"width{args.layer_width}"
        f"_offset{args.offset:g}"
        f"_cycle{args.cycle_length}"
        f"_epochs{args.epochs}"
        f"_agents{args.num_agents}"
        f"_model{args.model}"
        ".pkl"
    )

# --- Load or build the environment ---
G, node_to_idx, idx_to_node, fixed_starts_idx, fixed_pickups_idx, traffic_params = build_env(args)

# --- Build graph structures ---
# print("Building adjacency & time matrices")
adj_list, travel_times, neighbor_mask_static = build_adj_and_time_matrix(
    G, node_to_idx=node_to_idx
)

# print(f"Adjiacency list: {adj_list}, travel times: {travel_times}, neighbor mask: {neighbor_mask_static}")

# --- Precompute shortest-path distance matrix for shaping ---
# print("Precomputing distance matrix")
all_nodes = list(G.nodes())
N = len(all_nodes)
# print(f"Number of nodes in the graph: {N}")
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
key = jax_random.PRNGKey(1)
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

# Pre-compute this once outside the training loop
estimate_state = EstimateReturnsState.create(rollout_steps=10, gamma=args.gamma)

logger = wandb.init(
        project="taxi_pi",
        name=f"taxi-pi_run-{args.model}_100layers_200sim_{args.offset}",
        config={
            "num_steps": args.num_steps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "cycle_length": args.cycle_length,
            "offset": args.offset,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "lr": args.lr,
            "num_nodes": env.num_nodes,
            "max_deg": env.max_deg,
        }
    )

# --- DQN ---
if args.model == "dqn":
    # print("Starting DQN training...")
    params = train(
        env               = env,
        init_state_fn     = lambda: batched_states,
        obs_fn_batch      = obs_fn_batch,
        key               = eval_key,
        fixed_starts      = fixed_starts_idx,
        fixed_pickups     = fixed_pickups_idx,
        estimate_state    = estimate_state,
        # pretrain_ckpt     = args.pretrain_ckpt,
        num_steps         = args.num_steps,
        epochs            = args.epochs,
        batch_size        = args.batch_size,
        lr                = args.lr,
        gamma             = args.gamma,
        epsilon_start     = args.epsilon_start,
        epsilon_end       = args.epsilon_end,
        logger            = logger,
        use_untied        = args.use_untied,
    )

    # save the trained parameters
    with open(args.params_dir,'wb') as f:
        pickle.dump(params, f)

    def make_agent_policy(model, params, obs_fn_batch):
        """
        Wraps a Flax model into a single-state greedy policy using the batched obs_fn_batch.
        Correctly builds a batch of size 1 for inference.
        """
        def agent_policy(state: TaxiState) -> int:
            # Create a batched TaxiState of size 1 using tree_map
            batch_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)
            
            sf = obs_fn_batch(batch_state)
            # sf = obs['state_feats']      # Already [1, D_state]
            # af = obs['action_feats']     # Already [1, max_deg, D_action]
            # gf = obs['global_feats']     # Already [1, D_global]
            mask = batch_state.neighbor_mask       # Already [1, max_deg]
            q_vals = model.apply(params, sf, mask)
            return int(jnp.argmax(q_vals, axis=-1)[0])
        return agent_policy
    
    # model = QNetwork(ctx_dim=128, node_hidden=64, pool="mean", num_actions=env.max_deg)
    if args.use_untied:
        model = QNetworkUntied(hidden_dim=128, num_actions=env.max_deg)
    else:
        model = QNetworkSimple(hidden_dim=128, num_actions=env.max_deg)
    agent_policy = make_agent_policy(model, params, obs_fn_batch)


    def init_state_fn(key, starts, pickups):
        state, _ = init_env(key, starts, pickups, env.neighbor_mask_static)
        return state


    # print("Evaluating agent policy against SP baseline...")

    B = args.num_agents  # number of simultaneous taxi–passenger pairs

    # 1) Sample one fixed batch of start & pickup indices
    key, subkey = jax.random.split(eval_key)
    all_keys    = jax.random.split(subkey, 2 * B)
    start_keys, pickup_keys = all_keys[:B], all_keys[B:]
    starts  = jnp.array([jax.random.choice(k, fixed_starts_idx)  for k in start_keys])
    pickups = jnp.array([jax.random.choice(k, fixed_pickups_idx) for k in pickup_keys])
    # print(f"Sampled starts: {starts}, pickups: {pickups}")

    batched_init = jax.vmap(
        partial(init_env, neighbor_mask_static=neighbor_mask_static),
        in_axes=(0, 0, 0),
        out_axes=(0, 0)
    )

    # 2) Compute the RL‐matching via estimated returns (using your helper)
    def init_env_fn(starts: jnp.ndarray, pickups: jnp.ndarray):
            # make one new RNG‐key per trajectory
            rng_keys = jax.random.split(jax.random.PRNGKey(0), starts.shape[0])
            return batched_init(rng_keys, starts, pickups)


    R = estimate_returns_jit(
        env, params, model, obs_fn_batch,
        init_env_fn,
        starts, pickups,
        estimate_state=estimate_state,
        rollout_steps=10  # you can choose this horizon
    )  # → shape [B, B]
    R = jnp.array(R, dtype=jnp.float32)

    # solve the optimal transport problem
    _, rl_col = optax.assignment.hungarian_algorithm(-R)
    rl_pickups = pickups[rl_col]
    # print(f"RL matching: starts - {starts}, pickups - {rl_pickups}")

    # 3) Compute the SP‐matching via true SP distances
    #    (distances is your [N,N] JAX array from main)
    dist_np = onp.array(distances)  # to numpy for indexing
    D = dist_np[onp.array(starts), :][:, onp.array(pickups)]
    D = jnp.array(D, dtype=jnp.float32)
    _, sp_col = optax.assignment.hungarian_algorithm(D)
    sp_pickups = pickups[sp_col]
    # print(f"SP matching: starts - {starts}, pickups - {sp_pickups}")

    # 4) Define a shortest-path greedy policy
    def sp_policy(G, state: TaxiState) -> int:
        curr = idx_to_node[int(state.current_node)]
        goal = idx_to_node[int(state.pickup_node)]
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
                    a = policy(state)
                    state, _, _, info = env.step(state, a)
                    total += float(info['travel'] + info['wait'])
                    # print(f"RL: Current node: {state.current_node}, action: {a}, travel: {info['travel']}, wait: {info['wait']}\n")
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
                    # print(f"SP: Current node: {state.current_node}, action: {action}, travel: {info['travel']}, wait: {info['wait']}\n")
            times.append(total)
        return onp.array(times)

    # 6) Evaluate all four combinations
    rl_on_rl_times = eval_matching(starts,   rl_pickups, agent_policy, G, node_to_idx)
    sp_on_sp_times = eval_matching(starts,   sp_pickups, sp_policy, G, node_to_idx)

    # print("=== Matching & Policy Evaluation ===")
    print(f"RL matching + RL policy avg time: {rl_on_rl_times.mean():.2f}")
    print(f"SP matching + SP policy avg time: {sp_on_sp_times.mean():.2f}")

    # brute force for small environments
    # if args.env_type == "simple":
    #     best_routes = brute_force_shortest_path(env, args.num_layers, args.layer_width)
    #     C = build_cost_matrix(
    #         best_routes,
    #         starts=starts.tolist(),
    #         pickups=pickups.tolist()
    #     )
    #     _, bf_col = linear_sum_assignment(C)
    #     bf_pickups = pickups[bf_col]
    #     # print(f"Brute-force matching: starts - {starts}, pickups - {bf_pickups}")
    #     avg_bf_time = 0
    #     for i, (s, p) in enumerate(zip(onp.array(starts), onp.array(bf_pickups))):
    #         # print(f"Brute-force matching: {i+1}/{len(starts)}: {s} -> {p}")
    #         avg_bf_time += best_routes[(int(s), int(p))]["travel_time"]
    #         # print(f"Brute-force matching: {i+1}/{len(starts)}: {s} -> {p}, travel time: {best_routes[(int(s), int(p))]['travel_time']}")
    #     avg_bf_time /= len(starts)
    #     print(f"Brute-force matching + SP policy avg time: {avg_bf_time:.2f}")



# --- Policy Improvement ---
elif args.model == "pi":
    obs_fn_single, obs_fn_batch = make_obs_fn(env, G, node_to_idx)
    # init
    if args.pi_config is None:
        raise ValueError("Must pass --pi_config path to your JSON for policy improvement")
    with open(args.pi_config, "r") as f:
        config = json.load(f)

    # batch_size = num_agents
    config['batch_size'] = args.num_agents
    config['num_steps'] = args.epochs * config['eval_frequency']

    init_fn = get_init_fn(env, config, obs_fn_single)
    key, env_states, V_apply, V_opt_state, V_opt_update, get_V_params, V_target_params = init_fn(key)

    def linear_epsilon_decay(initial_eps=0.9, final_eps=0.1, decay_steps=10000):
        """Linear decay from initial_eps to final_eps over decay_steps"""
        def schedule(step):
            progress = jnp.clip(step / decay_steps, 0.0, 1.0)
            return initial_eps * (1.0 - progress) + final_eps * progress
        return schedule
    
    # build rec fn & agent loop
    key, subkey = jax_random.split(key)
    epsilon_schedule = linear_epsilon_decay(initial_eps=0.9, final_eps=0.05, decay_steps= config['num_steps'])
    recurrent_fn = get_recurrent_fn(env, V_apply, obs_fn_batch, epsilon_schedule)
    agent_loop = get_agent_loop(env, config, obs_fn_batch, V_apply, recurrent_fn, V_opt_update, get_V_params, epsilon_schedule)

    # initialize stats
    state_dict = {
        'key': key,
        'env_states': env_states,
        'last_start': env_states.current_node,
        'V_opt_state': V_opt_state,
        'V_params': V_target_params,
        'V_target_params': V_target_params,
        'opt_t': 0,
        'avg_return': jnp.zeros(config['batch_size']),
        'episode_return': jnp.zeros(config['batch_size']),
        'num_episodes': jnp.zeros(config['batch_size']),
        'visit_counts': jnp.zeros(env.num_nodes, dtype=jnp.int32), 
        'cumulative_visits': jnp.zeros(env.num_nodes, dtype=jnp.int32),
        'loss': jnp.array(0.0),  # Added loss tracking
        'episode_travel': jnp.zeros(config['batch_size']),
        'episode_wait': jnp.zeros(config['batch_size']),
        'avg_wait': jnp.zeros(config['batch_size']),
        'avg_travel': jnp.zeros(config['batch_size']),
    }

    avg_returns, times = [], []
    for i in tqdm(range(config['num_steps'] // config['eval_frequency'])):
        state_dict, metrics = agent_loop(state_dict)
            
        logger.log({
            "avg_return": state_dict['avg_return'].mean(),
            "visit_counts": wandb.Histogram(state_dict['visit_counts'].tolist()),
            "cumulative_visits": wandb.Histogram(state_dict['cumulative_visits'].tolist()),
            "loss": state_dict['loss']/config['eval_frequency'],
            "avg_wait": state_dict['avg_wait'].mean(),
            "avg_travel": state_dict['avg_travel'].mean(),
        })
        state_dict.update({
            'episode_return': jnp.zeros(config['batch_size']),
            'avg_travel': jnp.zeros(config['batch_size']),
            'avg_wait': jnp.zeros(config['batch_size']),
            'avg_return': jnp.zeros(config['batch_size']),
            'num_episodes': jnp.zeros(config['batch_size']),
            'episode_travel': jnp.zeros(config['batch_size']),
            'episode_wait': jnp.zeros(config['batch_size']),
            'visit_counts': jnp.zeros(env.num_nodes, dtype=jnp.int32),
            'loss': jnp.array(0.0),
        })
            

    # Get final parameters
    final_V_params = state_dict['V_target_params']
    # final_pi_params = state_dict['pi_params']

    # save
    with open(args.params_dir + '.out', 'wb') as f:
        pickle.dump({'config': config, 'avg_returns': avg_returns, 'times': times}, f)
    with open(args.params_dir + '.params', 'wb') as f:
        pickle.dump({'V': final_V_params}, f)

    def greedy_V_policy(state: TaxiState) -> int:
        """Use the learned value function to select actions greedily"""
        curr = state.current_node
        neighbors = env.adj_list[curr]
        mask = state.neighbor_mask
        
        best_a = None
        best_v = -jnp.inf
        
        for i, (n, valid) in enumerate(zip(neighbors, mask)):
            if not valid:
                continue
            
            # Create next state without stepping (simulate transition)
            next_state, reward, done, info = env.step(state, i)
        
            if done:
                # If this action leads to completion, use the immediate reward
                value = reward
            else:
                next_obs = obs_fn_single(next_state)
                next_v = V_apply(final_V_params, next_obs.astype(float))
                value = reward + env.gamma * next_v
            
            if value > best_v:
                best_v, best_a = value, i
        
        return int(best_a) if best_a is not None else 0

    def sp_policy(state: TaxiState) -> int:
        """Shortest path policy"""
        curr = state.current_node
        pickup = state.pickup_node
        neighbors = env.adj_list[curr]
        dists = env.distances[neighbors, pickup]
        mask = state.neighbor_mask
        dists = jnp.where(mask, dists, jnp.inf)
        return int(jnp.argmin(dists))

    def evaluate_policy(policy, starts, pickups, max_steps=100):
        """Evaluate a policy on a given initial state"""
        times = []
        for start, pickup in zip(starts, pickups):
            total_time = 0.0
            step_count = 0
            # Initialize a fresh single‐env state
            state = init_env(jax.random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
            while not state.done and step_count < max_steps:
                action = policy(state)
                state, _, _, info = env.step(state, action)
                total_time += float(info['travel'] + info['wait'])
                print(f"Step {step_count}: Current node: {state.current_node}, action: {action}, travel: {info['travel']}, wait: {info['wait']}")
                step_count += 1
            times.append(total_time)
        
        return onp.array(times)

    for _ in range(1):
        B = args.num_agents  # number of agents
        key, eval_key = jax.random.split(key)
        
        # Sample evaluation start and pickup points
        eval_keys = jax.random.split(eval_key, 2*B+1)
        start_keys, pickup_keys, base_key = eval_keys[:B], eval_keys[B:2*B], eval_keys[-1]
        init_keys  = jax.random.split(base_key, B**2)
        eval_starts = jnp.array([jax.random.choice(k, env.fixed_starts) for k in start_keys])
        eval_pickups = jnp.array([jax.random.choice(k, env.fixed_pickups) for k in pickup_keys])
        
        batch_init = jax.vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0))
        
        returns_matrix = estimate_returns_batch(
            init_keys, V_apply, obs_fn_batch, batch_init,
            final_V_params,
            eval_starts, eval_pickups
        )
        
        # Solve optimal transport problem
        _, rl_assignment = optax.assignment.hungarian_algorithm(-returns_matrix)
        rl_pickups = eval_pickups[rl_assignment]
        # print(f"Learned Policy: starts - {eval_starts}, pickups - {rl_pickups}")

        print(f"Value Policy: starts - {eval_starts}, pickups - {rl_pickups}")

        v_time = evaluate_policy(greedy_V_policy, eval_starts, rl_pickups)

        dist_np = onp.array(distances)  # to numpy for indexing
        D = dist_np[onp.array(eval_starts), :][:, onp.array(eval_pickups)]
        D = jnp.array(D, dtype=jnp.float32)
        _, sp_col = optax.assignment.hungarian_algorithm(D)
        sp_pickups = eval_pickups[sp_col]
        print(f"SP Policy: starts - {eval_starts}, pickups - {sp_pickups}")

        sp_time = evaluate_policy(sp_policy, eval_starts, sp_pickups)

        print(f"Value Policy   avg time: {v_time.mean():.2f}")
        print(f"SP Policy      avg time: {sp_time.mean():.2f}")

        # Brute force for small environments
        # if args.env_type == "simple":
        #     best_routes = brute_force_shortest_path(env, args.num_layers, args.layer_width)
        #     C = build_cost_matrix(
        #         best_routes,
        #         starts=eval_starts.tolist(),
        #         pickups=eval_pickups.tolist()
        #     )
        #     _, bf_col = linear_sum_assignment(C)
        #     bf_pickups = eval_pickups[bf_col]
        #     # print(f"Brute-force matching: starts - {starts}, pickups - {bf_pickups}")
        #     avg_bf_time = 0
        #     for i, (s, p) in enumerate(zip(onp.array(eval_starts), onp.array(bf_pickups))):
        #         # print(f"Brute-force matching: {i+1}/{len(starts)}: {s} -> {p}")
        #         avg_bf_time += best_routes[(int(s), int(p))]["travel_time"]
        #         # print("Brute force path:", best_routes[(int(s), int(p))]["path"])
        #         # print(f"Brute-force matching: {i+1}/{len(starts)}: {s} -> {p}, travel time: {best_routes[(int(s), int(p))]['travel_time']}")
        #     avg_bf_time /= len(eval_starts)
        #     print(f"Brute-force    avg time: {avg_bf_time:.2f}")