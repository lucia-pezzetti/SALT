import jax
import jax.numpy as jnp

# --- Random policy for comparisons ---
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