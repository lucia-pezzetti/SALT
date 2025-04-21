import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
from jax_taxi_env import init_env, TaxiState
from jax_ppo_agent import Transition, sample_action, ppo_loss

# helper functions

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = jnp.zeros_like(rewards)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv = adv.at[t].set(gae)
        next_value = values[t]
    returns = adv + values
    return adv, returns

def maybe_reset(env, state: TaxiState, key, fixed_starts, fixed_pickups, distances):
    def reset_one(subkey):
        return init_env(subkey, fixed_starts, fixed_pickups, distances)

    keys = random.split(key, state.done.shape[0] + 1)
    new_key = keys[0]
    subkeys = keys[1:]

    new_states, _ = jax.vmap(reset_one)(subkeys)

    def select(old, new):
        return jnp.where(state.done, new, old)

    return TaxiState(
        current_node = select(state.current_node, new_states.current_node),
        pickup_node  = select(state.pickup_node,  new_states.pickup_node),
        ride_phase   = select(state.ride_phase,   new_states.ride_phase),
        done         = jnp.zeros_like(state.done),
        step_count   = select(state.step_count,   new_states.step_count),
    ), new_key



@eqx.filter_jit
def batched_rollout(agent, env, state: TaxiState, key, num_steps, obs_fn):
    """
    Runs `num_steps` in parallel across `num_envs` envs,
    returns raw final state (no in-scan resets).
    """
    def step_fn(carry, _):
        state, key = carry

        # sample actions
        key, subkey = random.split(key)
        obs = obs_fn(state)
        keys = random.split(subkey, state.current_node.shape[0])
        actions, logps, values = jax.vmap(sample_action, in_axes=(None, 0, 0))(
            agent, obs, keys
        )

        # step env
        next_state, rewards = env.step(state, actions)
        dones = next_state.done

        # record transition
        trans = Transition(
            obs=obs,
            action=actions,
            reward=rewards,
            next_obs=obs_fn(next_state),
            done=dones,
            log_prob=logps,
            value=values
        )

        # Reset done envs immediately
        next_state, key = maybe_reset(env, next_state, key, env.fixed_starts, env.fixed_pickups, env.distances)

        return (next_state, key), trans

    (final_state, final_key), transitions = jax.lax.scan(
        step_fn,
        (state, key),
        xs=None,
        length=num_steps
    )
    return transitions, final_state, final_key


def train(
    agent,
    env,
    init_state_fn,
    obs_fn,
    key,
    fixed_starts,
    fixed_pickups,
    distances,
    num_steps=128,
    epochs=4,
    batch_size=64,
    lr=3e-4
):
    """
    PPO training loop. After each rollout, any sub-env whose `done`==True
    is re-initialized by calling init_env (in plain Python).
    """
    # optimizer setup
    params = eqx.filter(agent, eqx.is_array)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # initialize batch of states
    state = init_state_fn()

    for update in range(1000):
        # rollout
        transitions, state, key = batched_rollout(
            agent, env, state, key, num_steps, obs_fn
        )

        # flatten & compute GAE
        T, N = transitions.obs.shape[:2]
        obs      = transitions.obs.reshape(T * N, -1)
        act      = transitions.action.reshape(T * N)
        rew      = transitions.reward.reshape(T * N)
        nxt      = transitions.next_obs.reshape(T * N, -1)
        don      = transitions.done.reshape(T * N)
        logp     = transitions.log_prob.reshape(T * N)
        val      = transitions.value.reshape(T * N)

        adv, ret = compute_gae(rew, val, don)
        flat_trans = Transition(obs, act, rew, nxt, don, logp, val)

        # PPO updates
        for _ in range(epochs):
            key, subkey = random.split(key)
            idx = random.permutation(subkey, T * N)[:batch_size]
            batch     = jax.tree_util.tree_map(lambda x: x[idx], flat_trans)
            batch_adv = adv[idx]
            batch_ret = ret[idx]

            loss, grads = ppo_loss(agent, batch, batch_adv, batch_ret, 0.2)
            grads = eqx.filter(grads, eqx.is_array)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            agent  = eqx.apply_updates(agent, updates)

        # reset any finished sub-envs in Python
        # bring state into NumPy for indexing
        state_np = jax.tree_util.tree_map(lambda x: jnp.array(x), state)
        new_states = []
        for i in range(N):
            if state_np.done[i]:
                s_i, key = init_env(key, fixed_starts, fixed_pickups, distances)
                new_states.append(s_i)
            else:
                # clear done flag but keep everything else
                new_states.append(TaxiState(
                    current_node=int(state_np.current_node[i]),
                    pickup_node=int(state_np.pickup_node[i]),
                    ride_phase=int(state_np.ride_phase[i]),
                    done=False,
                    step_count=int(state_np.step_count[i])
                ))

        # stack back into JAX arrays
        state = TaxiState(
            current_node = jnp.array([s.current_node for s in new_states], dtype=jnp.int64),
            pickup_node  = jnp.array([s.pickup_node  for s in new_states], dtype=jnp.int64),
            ride_phase   = jnp.array([s.ride_phase   for s in new_states], dtype=jnp.int64),
            done         = jnp.array([s.done         for s in new_states], dtype=bool),
            step_count   = jnp.array([s.step_count   for s in new_states], dtype=jnp.int64),
        )

        # logging
        jax.debug.print("Update {}: Mean Reward {}, Mean Advantage: {}", update, jnp.mean(rew), jnp.mean(adv))

    return agent