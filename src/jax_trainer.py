import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
from jax_taxi_env import init_env, TaxiState
from jax_ppo_agent import Transition, sample_action, ppo_loss

# helper functions

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE) and returns.
    Args:
        rewards: Array of rewards.
        values: Array of value estimates.
        dones: Array of done flags.
        last_value: Value estimate for the last state.
        gamma: Discount factor.
        lam: GAE parameter.
    Returns:
        adv: Array of advantages.
        returns: Array of returns.
    """

    adv = jnp.zeros_like(rewards)
    gae = 0.0
    next_value = last_value

    # TODO: use jax.lax.scan + fixed length loop.
    # len(rewards) is not jittable, but you know the length at compile time
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv = adv.at[t].set(gae)
        next_value = values[t]
    returns = adv + values

    # advantage normalization
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return adv, returns

@jax.jit
def maybe_reset(state: TaxiState,
                key,
                fixed_starts,
                fixed_pickups,
                distances):
    """
    Resets the environment if `state.done` is True.
    * If state.done[i]==True → replace the ith env by a freshly
      initialised one.
    * Else → keep the existing state.
    Returns (new_state, new_key).
    """
    keys = random.split(key, state.done.shape[0] + 1)
    new_key = keys[0]
    subkeys = keys[1:]                       # shape (N,)

    # Candidate reset states for every env
    # vmap over sub‑keys (axis 0) – other args are broadcast (None)
    reset_states, _ = jax.vmap(init_env,
                           in_axes=(0, None, None, None))(
        subkeys, fixed_starts, fixed_pickups, distances
    )

    # Choose old or reset values field‑wise
    def choose(old, new):
        return jnp.where(state.done, new, old)

    return TaxiState(
        current_node = choose(state.current_node, reset_states.current_node),
        pickup_node  = choose(state.pickup_node,  reset_states.pickup_node),
        ride_phase   = choose(state.ride_phase,   reset_states.ride_phase),
        done         = jnp.zeros_like(state.done),   # clear flags
        step_count   = choose(state.step_count,   reset_states.step_count),
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
        # TODO: step is not jitted correctly I believe
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
        # TODO: I think this is triggering recompilations, state may be considered static
        next_state, key = maybe_reset(next_state, key, env.fixed_starts, env.fixed_pickups, env.distances)

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
    num_updates = 1000,
    num_steps=128,
    epochs=4,
    batch_size=64,
    lr=3e-4,
    clip_eps=0.2,
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

    # main training loop
    # TODO: Make sure this is jitted... jax.lax.scan
    for update in range(num_updates):
        # rollout
        # TODO: I think filter jit will give you a weird behavior.
        # In fact, you want state to be traced and not static, 
        # but it is considered static at the moment as it is not a pytree.
        # Better to use plain jit and rather have an error...
        transitions, state, key = batched_rollout(
            agent, env, state, key, num_steps, obs_fn
        )

        T, N = transitions.obs.shape[:2]

        # critic value of next state after the rollout (one per env)
        last_next_obs   = transitions.next_obs[-1]
        last_values_N   = jax.vmap(agent.critic)(last_next_obs)

        #  keep [T,N] until GAE; swap axes to [N,T] for vmapping
        # TODO: I do not think map is jitted, if really needed jax.lax.map
        rew_NT, val_NT, don_NT = map(
            lambda x: jnp.swapaxes(x, 0, 1),
            (transitions.reward, transitions.value, transitions.done)
        )                                                   # (N, T)

        # TODO: jax.vmap is not jitted automatically
        adv_NT, ret_NT = jax.vmap(compute_gae, in_axes=(0, 0, 0, 0))(
            rew_NT, val_NT, don_NT, last_values_N
        )                                                   # (N, T)

        # back to [T,N] then flatten to [T*N]
        adv = jnp.swapaxes(adv_NT, 0, 1).reshape(-1)
        ret = jnp.swapaxes(ret_NT, 0, 1).reshape(-1)

        # flatten other tensors
        flat_obs  = transitions.obs.reshape(T * N, -1)
        flat_act  = transitions.action.reshape(T * N)
        flat_logp = transitions.log_prob.reshape(T * N)
        flat_val  = transitions.value.reshape(T * N)
        flat_rew  = transitions.reward.reshape(T * N)

        # TODO: Transition is not pytree. Not traced. should be traced, else becomes static and triggers recompilation.
        flat_trans = Transition(
            obs      = flat_obs,
            action   = flat_act,
            reward   = flat_rew,   # optional
            next_obs = transitions.next_obs.reshape(T * N, -1),  # optional
            done     = transitions.done.reshape(T * N),     # optional
            log_prob = flat_logp,
            value    = flat_val
        )

        # PPO updates
        num_samples = T * N
        num_minibatches = num_samples // batch_size

        #  "old" param view for optax.update
        params = eqx.filter(agent, eqx.is_array)

        # TODO: jax lax scan
        for _ in range(epochs):
            key, subkey = random.split(key)
            perm = random.permutation(subkey, num_samples)

            # TODO: jax lax scan
            for mb in range(num_minibatches):
                mb_idx = perm[mb*batch_size : (mb+1)*batch_size]

                batch = jax.tree_util.tree_map(lambda x: x[mb_idx], flat_trans)
                batch_adv = adv[mb_idx]
                batch_ret = ret[mb_idx]

                # ----- loss and grads -----
                loss, grads = eqx.filter_value_and_grad(ppo_loss)(
                    agent, batch, batch_adv, batch_ret, clip_eps
                )

                grads = eqx.filter(grads, eqx.is_array)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                agent = eqx.apply_updates(agent, updates)


        # reset any finished sub-envs in Python
        # bring state into NumPy for indexing
        state_np = jax.tree_util.tree_map(lambda x: jnp.array(x), state)
        new_states = []
        # TODO: jax lax scan
        for i in range(N):
            # TODO: jax lax cond
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
        # TODO: make a pytree, this is not jitted and may trigger recompilations
        state = TaxiState(
            current_node = jnp.array([s.current_node for s in new_states], dtype=jnp.int64),
            pickup_node  = jnp.array([s.pickup_node  for s in new_states], dtype=jnp.int64),
            ride_phase   = jnp.array([s.ride_phase   for s in new_states], dtype=jnp.int64),
            done         = jnp.array([s.done         for s in new_states], dtype=bool),
            step_count   = jnp.array([s.step_count   for s in new_states], dtype=jnp.int64),
        )

        # logging
        # TODO: if everything done correctly, you should not see this print
        jax.debug.print("Update {}/{}: Mean Reward {}, Mean Return {}, Mean Advantage: {}, Std Advantage: {}", update, num_updates, jnp.mean(flat_rew), jnp.mean(ret), jnp.mean(adv), jnp.std(adv))

    return agent