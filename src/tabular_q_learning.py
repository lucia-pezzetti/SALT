import jax
from jax import lax, jit
from functools import partial
import jax.random as random
from taxi_env import init_env, TaxiEnv, TaxiState
import jax.numpy as jnp

class QLearningTrainer:
    """
    Tabular Q-learning trainer for the Taxi environment.
    """
    def __init__(
        self,
        env: TaxiEnv,
        all_pickups: jnp.ndarray,
        num_episodes: int = 100_000,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.1,
        seed: int = 0,
    ):
        self.env = env
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.seed = seed

        # Environment dimensions
        self.num_states = int(env.num_nodes)
        self.num_actions = int(env.max_deg)

        # static data for JIT
        self.neighbor_mask_static = jnp.array(env.neighbor_mask_static)
        self.fixed_starts = jnp.array(env.fixed_starts, dtype=jnp.int32)
        self.fixed_pickups = jnp.array(env.fixed_pickups, dtype=jnp.int32)

        # Initialize Q-table
        self.num_pickups = all_pickups.shape[0]
        self.Q = jnp.zeros((self.num_states, self.num_pickups, self.num_actions), dtype=jnp.float32)

    def q_step(self, Q, state, key, alpha, gamma, epsilon):
        curr = state.current_node              # scalar int
        pickup = state.pickup_node             # scalar int
        mask = state.neighbor_mask             # shape (num_actions,)

        # Mask invalid actions
        invalid = mask < 0                      # boolean mask
        q_slice = Q[curr, pickup%3]                        # shape (num_actions,)
        q_masked = jnp.where(invalid, -jnp.inf, q_slice)

        # Greedy
        a_greedy = jnp.argmax(q_masked)         # scalar int

        # Rejection-sample a truly uniform valid action
        def cond_fn(carry):
            _key, act = carry
            return invalid[act]                 # keep looping if invalid

        def body_fn(carry):
            key, _ = carry
            key, subkey = random.split(key)
            # sample in [0, num_actions)
            act = random.randint(subkey, (), 0, self.num_actions)
            return (key, act)

        # initial draw
        key, subkey = random.split(key)
        act0 = random.randint(subkey, (), 0, self.num_actions)
        key, a_random = lax.while_loop(cond_fn, body_fn, (key, act0))

        # ε-greedy mix
        key, subkey = random.split(key)
        rnd = random.uniform(subkey)
        action = jnp.where(rnd < epsilon, a_random, a_greedy)

        # Take the step & do the TD update
        next_state, reward, reached, _ = self.env.step(state, action)
        next_cur = next_state.current_node

        best_next = jnp.max(Q[next_cur, pickup%3])  # max Q-value for next state
        td_target = reward + gamma * best_next
        td_error = td_target - Q[curr, pickup%3, action]
        Q = Q.at[curr, pickup%3, action].add(alpha * td_error)
        # jax.debug.print("Q-learning step with state: {state}", state=state)
        # jax.debug.print("TD target: {td_target}, TD error: {td_error}", td_target=td_target, td_error=td_error)
        # jax.debug.print("Action taken: {action}, Reward: {reward}", action=action, reward=reward)
        # jax.debug.print("Next state: {next_state}", next_state=next_state)
        # jax.debug.print("Current Q-value: {q_value}", q_value=Q[curr, pickup%3, action])
        # jax.debug.print("Best next Q-value: {best_next}", best_next=best_next)
        # jax.debug.print("\n")

        # 6) Decay ε
        epsilon = jnp.maximum(self.min_epsilon, epsilon * self.epsilon_decay)

        return Q, next_state, key, epsilon
    
    @partial(jax.jit, static_argnums=(0,))
    def run_episode(self, Q, key, epsilon, start_idx, pickup_idx):
        # initialize state
        state, key = init_env(key, start_idx, pickup_idx, self.neighbor_mask_static)
        def body_fn(carry, _):
            Q, state, key, eps = carry
            Q, state, key, eps = self.q_step(Q, state, key, self.alpha, self.gamma, eps)
            return (Q, state, key, eps), None

        # maximum of max_steps steps; loop will break early if state.done
        (Q, state, key, epsilon), _ = lax.scan(body_fn,
                                        (Q, state, key, epsilon),
                                        None,
                                        length=self.env.max_steps)
        return Q, key, epsilon
    

    @partial(jax.jit, static_argnums=(0,))
    def train_jitted(self, Q, key, epsilon, starts, pickups):
        def episode_body(carry, idx):
            Q, key, eps = carry
            s = starts[idx]
            p = pickups[idx]
            Q, key, eps = self.run_episode(Q, key, eps, s, p)
            return (Q, key, eps), None

        # if you want N episodes, pass an array of indices 0…N−1
        carries, _ = lax.scan(episode_body,
                            (Q, key, epsilon),
                            jnp.arange(self.num_episodes))
        return carries[0]  # that's Q at the end
    
    def train(self) -> jnp.ndarray:
        key = random.PRNGKey(self.seed)
        # run the fully-JIT’d loop
        final_Q = self.train_jitted(
            self.Q, key, self.epsilon,
            self.fixed_starts, self.fixed_pickups
        )
        jax.debug.print("Q-table after training: {final_Q}", final_Q=final_Q)
        return final_Q