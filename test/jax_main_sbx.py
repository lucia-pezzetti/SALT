import os
import networkx as nx
import numpy as np
from datetime import datetime
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random as jax_random
import equinox as eqx
import geopandas as gpd
import random

from jax_utils import build_adj_and_time_matrix, make_obs_fn
from jax_taxi_env import JAXRideEnv, init_env, TaxiState
from utils import load_graph, apply_congestion_model, compute_zone_mappings

# Use SBX PPO from Stable Baselines Jax (sbx-rl)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from sbx import PPO, DQN
import gymnasium as gym
from gymnasium import spaces

# Wrapper to expose JAXRideEnv as a Gymnasium Env
class GymTaxiEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, jax_env: JAXRideEnv, obs_fn, seed: int = 0):
        super().__init__()
        self.env = jax_env
        self.obs_fn = obs_fn
        self.key = jax_random.PRNGKey(seed)
        # initialize state and batch for shape inference
        state, self.key = init_env(self.key,
            self.env.fixed_starts,
            self.env.fixed_pickups,
            self.env.distances)
        batched_state = jax.tree_util.tree_map(lambda x: jnp.array(x)[None], state)
        sample_obs = np.array(self.obs_fn(batched_state)[0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float32)
        self.action_space = spaces.Discrete(self.env.max_deg)
        self.state = state

    def reset(self, seed=None, options=None):
        self.key, subkey = jax_random.split(self.key)
        state, _ = init_env(subkey,
            self.env.fixed_starts,
            self.env.fixed_pickups,
            self.env.distances)
        self.state = state
        batched_state = jax.tree_util.tree_map(lambda x: jnp.array(x)[None], state)
        obs = np.array(self.obs_fn(batched_state)[0], dtype=np.float32)
        return obs, {}
    
    def step(self, action):
        self.key, subkey = jax_random.split(self.key)
        prev_phase = int(self.state.ride_phase)
        new_state, reward = self.env.step(self.state, int(action))
        self.state = new_state
        batched_state = jax.tree_util.tree_map(lambda x: jnp.array(x)[None], new_state)
        obs = np.array(self.obs_fn(batched_state)[0], dtype=np.float32)
        done = bool(new_state.done)
        pickup = (prev_phase == 0 and int(new_state.ride_phase) == 1)
        if done:
            info = {"pickup": pickup}
        else:
            info = {}
        return obs, float(reward), done, False, info
    
# --- Callback to log rewards and pickups ---
class RewardLoggerCallback(BaseCallback):
    """Logs episode rewards and pickup counts using SB3 callback hooks."""
    def __init__(self):
        super().__init__()
        self.rewards = []
        self.pickups = []
        self._pickup_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        # print("infos:", infos)
        dones = self.locals.get('dones', [])
        for info in infos:
            # log pickups on phase change
            if info.get('pickup', False):
                self._pickup_count += 1
            # log reward when episode ends
            ep_info = info.get('episode')
            if ep_info is not None and 'r' in ep_info:
                self.rewards.append(ep_info['r'])
        for done in dones:
            if done:
                # record pickups for this episode
                self.pickups.append(self._pickup_count)
                self._pickup_count = 0
        return True



def main():
    # --- Load and preprocess graph ---
    place_name = "Manhattan, New York City, New York, USA"
    zone_shp = "../data/processed/taxi_zones.shp"
    G = load_graph(place_name)
    apply_congestion_model(G)

    # ---------- TEMP -----------------
    locationID_to_nodes, zone_to_nodes, node_to_zone, nodes_gdf = compute_zone_mappings(G, zone_shp_path=zone_shp)

    # Get the LocationID for the Financial District
    zone_name = ["Financial District South", "Financial District North" , "Battery Park", "Seaport", "World Trade Center", "Battery Park City", "TriBeCa/Civic Center", "Chinatown", "Lower East Side", "East Village", "Little Italy/NoLiTa", "Two Bridges/Seward Park"]
    gdf_zones = gpd.read_file(zone_shp).to_crs("EPSG:4326")
    filtered_zones = gdf_zones[gdf_zones["zone"].isin(zone_name)]
    loc_ids = filtered_zones["LocationID"].tolist()
    # Now extract the node IDs
    selected_nodes = [node for loc_id in loc_ids for node in zone_to_nodes.get(loc_id, [])] 
    G_fds = G.subgraph(selected_nodes).copy()
    largest_cc = max(nx.strongly_connected_components(G_fds), key=len)
    G_fds = G_fds.subgraph(largest_cc).copy()

    #------------------------------------

    all_nodes = list(G_fds.nodes())
    print(f"Number of nodes: {len(all_nodes)}")
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    idx_to_node = [node for node, idx in sorted(node_to_idx.items(), key=lambda x: x[1])]


    # choose fixed start & pickup nodes
    fixed_starts = jnp.array(np.array(random.sample(all_nodes, 1), dtype=np.int64))
    fixed_pickups = jnp.array(np.array(random.sample(all_nodes, 1), dtype=np.int64))

    def make_has_path_fn(G_fds, idx_to_node):
        def has_path_fn(s_idx, t_idx):
            s = idx_to_node[s_idx]
            t = idx_to_node[t_idx]
            try:
                return nx.has_path(G_fds, s, t)
            except:
                return False
        return has_path_fn

    # --- Build JAX-ready graph structures ---
    print("Building adjacency and travel time matrices")
    adj_list, travel_times = build_adj_and_time_matrix(G_fds, node_to_idx=node_to_idx)
    # map fixed nodes to indices
    fixed_starts_idx = [node_to_idx[int(n)] for n in fixed_starts if int(n) in node_to_idx]
    fixed_pickups_idx = [node_to_idx[int(n)] for n in fixed_pickups if int(n) in node_to_idx]

    # --- Precompute shortest-path distance matrix for reward shaping ---
    print("Precomputing shortest-path distance matrix")
    N = adj_list.shape[0]
    dist_mat = np.zeros((N, N), dtype=np.float32)
    for u, lengths in nx.all_pairs_dijkstra_path_length(G_fds, weight="weight"):
        ui = node_to_idx[u]
        for v, d in lengths.items():
            vi = node_to_idx[v]
            dist_mat[ui, vi] = d
    distances = jnp.array(dist_mat)

    print("Distance between pickup and dropoff: ", distances[fixed_starts_idx, fixed_pickups_idx])

    # Create JAX environment
    jax_env = JAXRideEnv(
        adj_list=jnp.array(adj_list, dtype=jnp.int32),
        travel_times=jnp.array(travel_times, dtype=jnp.float32),
        distances=jnp.array(distances, dtype=jnp.float32),
        max_deg=adj_list.shape[1],
        num_nodes=N,
        max_steps=128,
        fixed_starts=fixed_starts_idx,
        fixed_pickups=fixed_pickups_idx,
        timeout_penalty=-50.0,
    )
    # Build observation function
    # If you have node_to_idx mapping, pass it as in your original code
    obs_fn = make_obs_fn(G_fds, node_to_idx, jax_env.max_steps)  # adjust args accordingly

    # Wrap in Gym environment
    gym_env = GymTaxiEnv(jax_env, obs_fn, seed=42)

    # --- Vectorize and normalize ---
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback

    # Custom callback to log episodic rewards
    env_callback = DummyVecEnv([lambda: gym_env])  # single-env vectorization
    vec_env = VecNormalize(env_callback, norm_obs=True, norm_reward=True)

    network_width = 256


    # Initialize and train SBX PPO
    model_ppo = PPO(
        policy="MlpPolicy",
        env=gym_env,
        verbose=0,
        seed=42,
        normalize_advantage=True,
        vf_coef=0.5,
        ent_coef=0.01,
        learning_rate=3e-4,
        batch_size=128,
        policy_kwargs={
            "net_arch": [network_width, network_width]
        }
    )

    model_dqn = DQN(
        "MlpPolicy",
        env=env_callback,
        verbose=0,
        seed=42,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        policy_kwargs={"net_arch": [256, 256]}
    )

    # Instantiate reward logger
    reward_logger = RewardLoggerCallback()

    total_timesteps = 1_000_000  # adjust as necessary
    # model_ppo.learn(total_timesteps=total_timesteps, callback=reward_logger, progress_bar=True)
    model_dqn.learn(total_timesteps=total_timesteps, callback=reward_logger, progress_bar=True)

    # Save the trained model
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_path = os.path.join(output_dir, f"sbx_ppo_taxi_{timestamp}")
    # model_ppo.save(model_path)
    # print(f"Trained SBX PPO model saved to {model_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"sb3_dqn_taxi_{timestamp}")
    model_dqn.save(model_path)
    print(f"Trained SB3 DQN model saved to {model_path}")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(reward_logger.rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards over Training')
    plot_path = os.path.join(output_dir, f'rewards_{timestamp}.png')
    plt.savefig(plot_path)
    print(f"Rewards plot saved to: {plot_path}")

    # plot pickups per episode
    plt.figure()
    plt.plot(reward_logger.pickups)
    plt.xlabel('Episode')
    plt.ylabel('Number of Pickups')
    plt.title('Pickups per Episode over Training')
    pickups_plot_path = os.path.join(output_dir, f'pickups_{timestamp}.png')
    plt.savefig(pickups_plot_path)
    print(f"Pickups plot saved to: {pickups_plot_path}")

    # also print the raw pickups list
    # print("Pickups per episode:", reward_logger.pickups)

        # --- Episode termination analysis ---
    total_episodes = len(reward_logger.pickups)
    pickup_episodes = sum(1 for p in reward_logger.pickups if p > 0)
    pickup_ratio = pickup_episodes / total_episodes if total_episodes > 0 else 0.0
    print(f"Episodes ending with pickup: {pickup_episodes}/{total_episodes} ({pickup_ratio:.2%})")




if __name__ == "__main__":
    main()
