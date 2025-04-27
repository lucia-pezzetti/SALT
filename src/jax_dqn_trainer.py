import pickle
from typing import NamedTuple, Callable, Sequence

import jax
import jax.numpy as jnp
from jax import random
import optax
from functools import partial
from nn import MLP
from jax_taxi_env import TaxiState, init_env
from flax import linen as nn


# --- Q-network definition ---
class QNetwork(nn.Module):
    dim_hidden: Sequence[int]
    num_actions: int
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for h in self.dim_hidden:
            x = self.act_fn(nn.Dense(features=h)(x))
        return nn.Dense(features=self.num_actions)(x)


# --- Batch now includes actions for Q-learning ---
class Batch(NamedTuple):
    state: TaxiState        # [T, B, ...]
    action: jnp.ndarray     # [T, B]
    reward: jnp.ndarray     # [T, B]
    next_state: TaxiState   # [T, B, ...]
    done: jnp.ndarray       # [T, B]
    pickups: jnp.ndarray = None  # [T, B] (optional)



# @jax.jit
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

def get_batched_rollout_q(
    model,
    obs_fn_batch,
    fixed_starts,
    fixed_pickups,
    neighbor_mask_static,
    *,
    num_steps: int = 128,
):
    def batched_rollout_q(
        env,
        init_states: TaxiState,
        key: jnp.ndarray,
        params,
        epsilon: float = 0.1
    ) -> Batch:
        B = init_states.current_node.shape[0]
        keys = random.split(key, num_steps + 1)

        def step_fn(state, k):
            # split for action, exploration, reset
            k1, k2, k3 = random.split(k, 3)
            obs = obs_fn_batch(state)
            q_values = model.apply(params, obs)    # [B, A]

            # mask invalid neighbors
            mask = state.neighbor_mask.astype(bool)
            masked_q = jnp.where(mask, q_values, -1e10)

            # greedy and random actions among neighbors
            greedy = jnp.argmax(masked_q, axis=-1)
            probs = jnp.where(mask, state.neighbor_mask, 0.0)
            probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
            rand = random.categorical(k1, jnp.log(probs))
            explore = random.uniform(k2, (B,)) < epsilon
            action = jnp.where(explore, rand, greedy)

            next_state, reward, reach_pickup = env.step(state, action)
            done = next_state.done

            # reset any finished envs
            next_state, _ = maybe_reset(
                next_state, k3,
                fixed_starts, fixed_pickups,
                neighbor_mask_static
            )

            return next_state, (state, action, reward, next_state, done, reach_pickup)

        final_state, traj = jax.lax.scan(step_fn, init_states, keys)
        states, actions, rewards, next_states, dones, pickups = traj
        return Batch(states, actions, rewards, next_states, dones, pickups), final_state

    return jax.jit(batched_rollout_q)


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
    # --- Q-network outputs one Q per action ---
    action_dim = env.neighbor_mask_static.shape[-1]
    model = QNetwork(dim_hidden=[128,128], num_actions=action_dim)
    params = model.init(key, jnp.zeros((1, dim_obs)))
    # target network
    target_params = params

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # For percentage of pickups
    pickup_count = 0
    done_count = 0

    rollout = get_batched_rollout_q(
        model, obs_fn_batch, 
        fixed_starts, fixed_pickups, 
        env.neighbor_mask_static, num_steps=num_steps
    )

    @partial(jax.jit, static_argnums=(8,))
    def train_step(params, target_params, opt_state, obs_flat, acts_flat, rews_flat, next_obs_flat, dones_flat, gamma: float):
        # compute Q and target on full batch
        def loss_fn(p):
            q = model.apply(p, obs_flat)
            q_taken = jnp.take_along_axis(q, acts_flat[:, None], -1).squeeze(-1)
            # compute next Q using frozen target network
            qn = model.apply(target_params, next_obs_flat)
            max_q_next = jnp.max(qn, axis=-1)
            td = rews_flat + gamma * max_q_next * (1.0 - dones_flat)
            td_target = jax.lax.stop_gradient(td)
            return jnp.mean((q_taken - td_target)**2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    states = init_state_fn()
    for epoch in range(1, epochs + 1):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (epoch / epochs)
        key, subkey = random.split(key)
        batch, states = rollout(env, states, subkey, params, epsilon)
        states_seq, acts_seq, rews_seq, next_states_seq, dones_seq, _ = batch

        # compute observations outside jit
        obs_seq = jax.vmap(obs_fn_batch)(states_seq)       # (T, B, dim_obs)
        next_obs_seq = jax.vmap(obs_fn_batch)(next_states_seq)

        # Count pickups
        per_env_picked  = jnp.any(batch.pickups,  axis=0)
        per_env_done = jnp.any(batch.done, axis=0)
        n_picked  = int(jnp.sum(per_env_picked))
        n_done   = int(jnp.sum(per_env_done))
        pickup_count  += n_picked
        done_count   += n_done

        pickup_rate = 100.0 * pickup_count / done_count
        epoch_rate = n_picked / n_done


        # flatten trajectories and get obs
        T, B = acts_seq.shape
        obs_flat = obs_seq.reshape((T * B, dim_obs))
        acts_flat = acts_seq.reshape((T * B,))
        rews_flat = rews_seq.reshape((T * B,))
        next_obs_flat = next_obs_seq.reshape((T * B, dim_obs))
        dones_flat = dones_seq.reshape((T * B,))

        # Shuffle and minibatch update
        idx = random.permutation(key, T * B)
        for i in range(0, T * B, batch_size):
            mb = idx[i : i + batch_size]
            mb_obs = obs_flat[mb]
            mb_act = acts_flat[mb]
            mb_rew = rews_flat[mb]
            mb_next_obs = next_obs_flat[mb]
            mb_done = dones_flat[mb]

            params, opt_state, loss = train_step(
                params, target_params, opt_state,
                mb_obs, mb_act, mb_rew, mb_next_obs, mb_done,
                gamma
            )

        print(f"Epoch {epoch}/{epochs} - Q Loss: {loss:.4f} - Pickup Rate: {pickup_rate:.2f}% - Epoch Rate: {epoch_rate:.2f}")
        # print(f"Model params: {jax.tree_util.tree_flatten(params)[0]}")

    with open("trained_q_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Saved trained Q-network parameters to trained_q_params.pkl")

    return params
