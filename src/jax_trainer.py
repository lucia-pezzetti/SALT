import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
from functools import partial
from jax_taxi_env import init_env, TaxiState
# from jax_ppo_agent import Transition, sample_action, ppo_loss
from nn import MLP
from typing import NamedTuple
import pickle

# helper functions

# @partial(jax.jit, static_argnames=["gamma", "lam"])
# def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
#     """
#     Compute Generalized Advantage Estimation (GAE) and returns.
#     Args:
#         rewards: Array of rewards.
#         values: Array of value estimates.
#         dones: Array of done flags.
#         last_value: Value estimate for the last state.
#         gamma: Discount factor.
#         lam: GAE parameter.
#     Returns:
#         adv: Array of advantages.
#         returns: Array of returns.
#     """

#     def gae_step(carry, t):
#         gae, next_value = carry
#         mask = 1.0 - dones[t]
#         delta = rewards[t] + gamma * next_value * mask - values[t]
#         gae = delta + gamma * lam * mask * gae
#         adv_t = gae
#         return (gae, values[t]), adv_t

#     _, adv = jax.lax.scan(
#         gae_step,
#         (last_value, 0.0),
#         jnp.arange(len(rewards)),
#         reverse=True
#     )

#     returns = adv + values
#     # Normalize advantages
#     adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
#     return adv, returns

class Batch(NamedTuple):
    state: TaxiState       # [T, B, ...]
    reward: jnp.ndarray    # [T, B]
    next_state: TaxiState  # [T, B, ...]
    done: jnp.ndarray      # [T, B]
    pickups: jnp.ndarray = None  # [T, B] (optional)

@jax.jit
def maybe_reset(state: TaxiState, key: jnp.ndarray, fixed_starts, fixed_pickups, neighbor_mask_static):
    batch_size = state.done.shape[0]
    keys = random.split(key, batch_size + 1)
    new_key = keys[0]
    subkeys = keys[1:]

    def reset_fn(subkey):
        reset_state, _ = init_env(subkey, fixed_starts, fixed_pickups, neighbor_mask_static)
        return reset_state

    # vmap reset
    reset_states = jax.vmap(reset_fn)(subkeys)

    def choose_one(done, old, new):
        return jnp.where(done, new, old)

    new_state = TaxiState(
        current_node = jax.vmap(choose_one)(state.done, state.current_node, reset_states.current_node),
        pickup_node  = jax.vmap(choose_one)(state.done, state.pickup_node, reset_states.pickup_node),
        # ride_phase   = jax.vmap(choose_one)(state.done, state.ride_phase, reset_states.ride_phase),
        step_count   = jax.vmap(choose_one)(state.done, state.step_count, reset_states.step_count),
        done         = jnp.zeros_like(state.done),  # reset done flags
        neighbor_mask= jax.vmap(choose_one)(state.done, state.neighbor_mask, reset_states.neighbor_mask)
    )

    return new_state, new_key


def get_batched_rollout_v(
    model,
    obs_fn_batch,
    *,
    num_steps: int = 128,
):
    @jax.jit
    def batched_rollout_v(
        env,
        init_states: TaxiState,
        key: jnp.ndarray,
        params,
        epsilon: float = 0.1
    ) -> Batch:
        
        B = init_states.current_node.shape[0]
        keys = random.split(key, num_steps + 1)

        def step_fn(state, k):
            # split once per step
            k1, k2 = jax.random.split(k, 2)
            obs = obs_fn_batch(state)
            neighbor_mask = state.neighbor_mask
            q_values = model.apply(params, obs, s=False)       # [B, A]

            # make mask boolean
            bool_mask = neighbor_mask.astype(bool)

            # greedy action over masked Qs
            masked_q = jnp.where(bool_mask, q_values, -1e10)
            # jax.debug.print("masked_q: {}", masked_q)
            greedy_action = jnp.argmax(masked_q, axis=-1)      # [B]

            # random action only among real neighbors
            probs = jnp.where(bool_mask, neighbor_mask, 0.0)
            probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
            rand_action = jax.random.categorical(k1, jnp.log(probs))  # [B]

            # epsilon‐greedy
            explore = jax.random.uniform(k2, (B,)) < epsilon
            action = jnp.where(explore, rand_action, greedy_action)

            next_state, reward, reach_pickup = env.step(state, action)
            done = next_state.done

            return next_state, (state, reward, next_state, done, reach_pickup)


        _, traj = jax.lax.scan(step_fn, init_states, keys)
        states, rewards, next_states, dones, pickups = traj

        return Batch(states, rewards, next_states, dones, pickups)


    return batched_rollout_v



# TD(0) loss function
# @jax.jit
def td_loss(
    params,
    model_apply,
    states: jnp.ndarray,
    targets: jnp.ndarray
) -> jnp.ndarray:
    values = model_apply(params, states)
    return jnp.mean((values - targets) ** 2)



def train(
    dim_obs: int,
    env,
    init_state_fn,
    obs_fn,
    obs_fn_batch,
    key: jnp.ndarray,
    fixed_starts,
    fixed_pickups,
    num_steps: int = 128,
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    epsilon_start: float = 0.2,
    epsilon_end: float = 0.01,
):
    """
    Train a value network using TD(0) updates under a random policy.
    Returns final value-function parameters.
    """
    # Initialize value network and optimizer
    model = MLP(dim_hidden=[64, 64])
    params = model.init(key, jnp.zeros((1, dim_obs)))
    # jax.debug.print("Model parameters: ", params)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # For percentage of pickups
    pickup_count = 0
    done_count = 0

    # Model apply: returns scalar value
    @jax.jit
    def model_apply(p, state: TaxiState) -> jnp.ndarray:
        obs = obs_fn(state)
        v = model.apply(p, obs, s=False)
        return jnp.squeeze(v, axis=-1)
    
    batched_rollout = get_batched_rollout_v(model, obs_fn_batch, num_steps=num_steps)


    # Main training loop
    for epoch in range(1, epochs + 1):
        # Linearly decay epsilon
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (epoch / epochs)

        # Initialize states and rollout
        states = init_state_fn()
        key, subkey = random.split(key)
        
        batch = batched_rollout(env, states, subkey, params, epsilon)
        final_state = jax.tree_util.tree_map(lambda x: x[-1], batch.next_state)
        states, key = maybe_reset(final_state, key, fixed_starts, fixed_pickups, neighbor_mask_static=env.neighbor_mask_static)

        # Count pickups
        per_env_picked  = jnp.any(batch.pickups,  axis=0)
        per_env_done = jnp.any(batch.done, axis=0)
        n_picked  = int(jnp.sum(per_env_picked))
        n_done   = int(jnp.sum(per_env_done))
        pickup_count  += n_picked
        done_count   += n_done

        pickup_rate = 100.0 * pickup_count / done_count
        epoch_rate = n_picked / n_done


        # Compute TD targets: r + gamma * V(next) * (1 - done)
        # Evaluate V(next) for each time and env: [T, B]
        def v_next_fn(ns):
            return model_apply(params, ns)
        
        v_next = jax.vmap(v_next_fn, in_axes=0)(batch.next_state)
        targets = batch.reward + gamma * v_next * (1 - batch.done)

        # Flatten time and batch dims
        T, B = targets.shape
        states_flat = jax.tree_util.tree_map(
            lambda x: x.reshape((T * B, -1)), batch.state
        )
        targets_flat = targets.reshape(-1)

        # Shuffle indices
        idx = random.permutation(key, T * B)
        for i in range(0, T * B, batch_size):
            mb_idx = idx[i : i + batch_size]
            mb_states = jax.tree_util.tree_map(lambda x: x[mb_idx], states_flat)
            mb_targets = targets_flat[mb_idx]
            loss, grads = jax.value_and_grad(td_loss)(
                params, model_apply, mb_states, mb_targets
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        print(f"Epoch {epoch}/{epochs} - TD Loss: {loss:.4f} - Pickup Rate: {pickup_rate:.2f}% - Epoch Rate: {epoch_rate:.2f} - Epsilon: {epsilon:.4f}")

    # Save trained value parameters
    with open("trained_value_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Saved trained value parameters to trained_value_params.pkl")

    return params


# def get_batched_rollout(model, adj, env, obs_fn, num_steps):
#     def batched_rollout(params, state: TaxiState, key):
#         """
#         Runs `num_steps` in parallel across `num_envs` envs,
#         returns raw final state (no in-scan resets).
#         """
#         def step_fn(carry, _):
#             state, key = carry

#             # sample actions
#             key, subkey = random.split(key)
#             obs = obs_fn(state)
#             keys = random.split(subkey, state.current_node.shape[0])

#             # TODO:
#             # 1. evaluate value function at all out edges
#             # 2. Bellman
#             # 3. sample eps greedy
#             # TODO: get next lat,lon
#             neighbors = adj[state.current_node]
#             # TODO: actually, the stack should be of current lat, lon
#             obs = jnp.stack([
#                 jnp.tile(state.current_node, (neighbors.shape[1], 1)),
#                 neighbors
#                 # TODO: we can also stack the current time
#             ], axis=1)

#             # sample values
#             values = jax.vmap(lambda x : model.apply(params, x, key=subkey))(obs)

#             # TODO: sample actions epsilon greedy
#             # jax.random.uniform(subkey, )

#             arg_max = jnp.argmax(values, axis=0)

#             # step env
#             # TODO: step is not jitted correctly I believe
#             next_state, rewards = env.step(state, arg_max)
#             dones = next_state.done

#             # record transition
#             trans = Transition(
#                 obs=obs,
#                 action=actions,
#                 reward=rewards,
#                 next_obs=obs_fn(next_state),
#                 done=dones,
#                 log_prob=logps,
#                 value=values
#             )

#             # Reset done envs immediately
#             # TODO: I think this is triggering recompilations, state may be considered static
#             next_state, key = maybe_reset(next_state, key, env.fixed_starts, env.fixed_pickups, env.distances)

#             return (next_state, key), trans

#         (final_state, final_key), transitions = jax.lax.scan(
#             step_fn,
#             (state, key),
#             xs=None,
#             length=num_steps
#         )
#         return transitions, final_state, final_key
#     return batched_rollout


# def train(
#     dim_obs,
#     env,
#     init_state_fn,
#     obs_fn,
#     key,
#     fixed_starts,
#     fixed_pickups,
#     distances,
#     num_updates = 1000,
#     num_steps=128,
#     epochs=4,
#     batch_size=64,
#     lr=3e-4,
#     clip_eps=0.2,
# ):
#     """
#     PPO training loop. After each rollout, any sub-env whose `done`==True
#     is re-initialized by calling init_env (in plain Python).
#     """
#     # Define the model
#     dim_hidden = [64, 64]  # Example hidden layer dimensions
#     model = MLP(dim_hidden=dim_hidden)

#     # Initialize the model
#     # TODO: shape of keys?
#     keys = jax.random.split(key, N)
#     input_shape = (1, dim_obs)
#     params = model.init(key, jnp.ones(input_shape))

#     # optimizer setup
#     tx = optax.adam(learning_rate=lr)
#     opt_state = tx.init(params)

#     # initialize batch of states
#     states = init_state_fn()
#     # TODO: jit but the agent needs to be passed
#     batched_rollout = get_batched_rollout(model, env, obs_fn, num_steps)

#     # main training loop
#     # TODO: Make sure this is jitted... jax.lax.scan
#     for update in range(num_updates):

#         # Option 1
#         def loss_fn(p):
#             transitions, state, key = jax.vmap(
#                 lambda s, k: batched_rollout(p, s, k))(states, keys)
#             # TODO: Compute loss as per value iteration (bellman error) or TD(0) maybe easier to start
#             return loss
        
#         # Option 2: batched_rollout outside loss
#         # Define loss for each minibatch
#         def get_loss_fn(minibatch):
#             def loss_fn(p):
#                 # TODO: Compute loss as per minibatch
#                 return loss

#         loss, grads = jax.value_and_grad(loss_fn)(params)
#         updates, new_opt_state = tx.update(grads, opt_state, params)
#         params = optax.apply_updates(params, updates)
#         print("Loss: ", loss)


#         T, N = transitions.obs.shape[:2]

#         # critic value of next state after the rollout (one per env)
#         last_next_obs   = transitions.next_obs[-1]
#         last_values_N   = jax.vmap(agent.critic)(last_next_obs)

#         #  keep [T,N] until GAE; swap axes to [N,T] for vmapping
#         # TODO: I do not think map is jitted, if really needed jax.lax.map
#         rew_NT, val_NT, don_NT = map(
#             lambda x: jnp.swapaxes(x, 0, 1),
#             (transitions.reward, transitions.value, transitions.done)
#         )                                                   # (N, T)

#         adv, ret = jax.vmap(compute_gae, in_axes=(0, 0, 0, 0))(
#             rew_NT, val_NT, don_NT, last_values_N
#         )                                                   # (N, T)
        
#         adv = jnp.swapaxes(adv, 0, 1).reshape(T * N)  # (T*N,)
#         ret = jnp.swapaxes(ret, 0, 1).reshape(T * N)    # (T*N,)

#         # flatten other tensors
#         flat_obs  = transitions.obs.reshape(T * N, -1)
#         flat_act  = transitions.action.reshape(T * N)
#         flat_logp = transitions.log_prob.reshape(T * N)
#         flat_val  = transitions.value.reshape(T * N)
#         flat_rew  = transitions.reward.reshape(T * N)

#         # TODO: Transition is not pytree. Not traced. should be traced, else becomes static and triggers recompilation.
#         flat_trans = Transition(
#             obs      = flat_obs,
#             action   = flat_act,
#             reward   = flat_rew,   # optional
#             next_obs = transitions.next_obs.reshape(T * N, -1),  # optional
#             done     = transitions.done.reshape(T * N),     # optional
#             log_prob = flat_logp,
#             value    = flat_val
#         )

#         # PPO updates
#         num_samples = T * N
#         num_minibatches = num_samples // batch_size

#         #  "old" param view for optax.update
#         params = eqx.filter(agent, eqx.is_array)

#         # TODO: jax lax foriloop
#         for _ in range(epochs):
#             key, subkey = random.split(key)
#             perm = random.permutation(subkey, num_samples)

#             # TODO: jax lax scan
#             for mb in range(num_minibatches):
#                 mb_idx = perm[mb*batch_size : (mb+1)*batch_size]

#                 batch = jax.tree_util.tree_map(lambda x: x[mb_idx], flat_trans)
#                 batch_adv = adv[mb_idx]
#                 batch_ret = ret[mb_idx]

#                 # ----- loss and grads -----
#                 loss, grads = eqx.filter_value_and_grad(ppo_loss)(
#                     agent, batch, batch_adv, batch_ret, clip_eps
#                 )

#                 grads = eqx.filter(grads, eqx.is_array)
#                 updates, opt_state = optimizer.update(grads, opt_state, params)
#                 params = optax.apply_updates(params, updates)
#                 agent = eqx.apply_updates(agent, updates)


#         # reset any finished sub-envs in Python
#         # bring state into NumPy for indexing
#         state_np = jax.tree_util.tree_map(lambda x: jnp.array(x), state)
#         new_states = []
#         # TODO: jax lax scan
#         for i in range(N):
#             # TODO: jax lax cond
#             if state_np.done[i]:
#                 s_i, key = init_env(key, fixed_starts, fixed_pickups, distances)
#                 new_states.append(s_i)
#             else:
#                 # clear done flag but keep everything else
#                 new_states.append(TaxiState(
#                     current_node=int(state_np.current_node[i]),
#                     pickup_node=int(state_np.pickup_node[i]),
#                     ride_phase=int(state_np.ride_phase[i]),
#                     done=False,
#                     step_count=int(state_np.step_count[i])
#                 ))

#         # stack back into JAX arrays
#         # TODO: make a pytree, this is not jitted and may trigger recompilations
#         state = TaxiState(
#             current_node = jnp.array([s.current_node for s in new_states], dtype=jnp.int64),
#             pickup_node  = jnp.array([s.pickup_node  for s in new_states], dtype=jnp.int64),
#             ride_phase   = jnp.array([s.ride_phase   for s in new_states], dtype=jnp.int64),
#             done         = jnp.array([s.done         for s in new_states], dtype=bool),
#             step_count   = jnp.array([s.step_count   for s in new_states], dtype=jnp.int64),
#         )

#         # logging
#         # TODO: if everything done correctly, you should not see this print
#         jax.debug.print("Update {}/{}: Mean Reward {}, Mean Return {}, Mean Advantage: {}, Std Advantage: {}", update, num_updates, jnp.mean(flat_rew), jnp.mean(ret), jnp.mean(adv), jnp.std(adv))

#     return agent