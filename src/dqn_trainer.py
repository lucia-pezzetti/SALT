import pickle
import time
from typing import NamedTuple, Callable, Sequence, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
import optax
from flax import linen as nn
from flax.training import checkpoints
from flax.core.frozen_dict import freeze, unfreeze

from taxi_env import init_env, TaxiState, JAXRideEnv
from visualization_utils import TrainingLogger

# --- Q-network definition (Flax) ---
class QNetwork(nn.Module):
    max_deg: int
    dim_ctx: int = 512
    dim_action: int = 512

    @nn.compact
    def __call__(self, state_feats, action_feats, mask):
        # 1) Embed state
        ctx = nn.relu(nn.Dense(self.dim_ctx)(state_feats))            # [B, dim_ctx]

        # 2) Broadcast and concatenate
        B, M, _ = action_feats.shape
        ctx_b = jnp.expand_dims(ctx, 1)                               # [B, 1, dim_ctx]
        ctx_b = jnp.broadcast_to(ctx_b,   (B, M, self.dim_ctx))       # [B, M, dim_ctx]
        h = jnp.concatenate([ctx_b, action_feats], axis=-1)           # [B, M, dim_ctx + D_action]

        # 3) Score all actions “in one shot”
        h = nn.relu(nn.Dense(self.dim_action)(h))                     # [B, M, dim_action]
        q_raw = nn.Dense(1)(h).squeeze(-1)                            # [B, M]

        large_neg = -1e9
        return jnp.where(mask, q_raw, large_neg)
    

def adapt_pretrained_zeroinit(params, pretrained, init_params):
    """
    Copy over old weights, zero‑initialize extra rows when input dims have grown.
    Handles both Dense_0 (state embed) and Dense_1 (action scoring).
    """
    params = unfreeze(params)
    pretrained = unfreeze(pretrained)
    init_params = unfreeze(init_params)

    # --- Dense_0: state embedding ---
    if 'Dense_0' in pretrained['params']:
        ker_pre0 = pretrained['params']['Dense_0']['kernel']   # [old_in0, dim_ctx]
        bias_pre0 = pretrained['params']['Dense_0'].get('bias')
        ker_init0 = init_params['params']['Dense_0']['kernel']  # [new_in0, dim_ctx]
        old_in0, dim_ctx = ker_pre0.shape
        new_in0, _ = ker_init0.shape
        # pad zeros for new input dims
        if new_in0 > old_in0:
            pad_rows = jnp.zeros((new_in0 - old_in0, dim_ctx), dtype=ker_pre0.dtype)
            new_kernel0 = jnp.vstack([ker_pre0, pad_rows])
        else:
            new_kernel0 = ker_pre0[:new_in0]
        params['params']['Dense_0']['kernel'] = new_kernel0
        if bias_pre0 is not None:
            params['params']['Dense_0']['bias'] = bias_pre0

    # --- Dense_1: action-scoring layer ---
    ker_pre1  = pretrained['params']['Dense_1']['kernel']
    bias_pre1 = pretrained['params']['Dense_1'].get('bias')
    ker_init1 = init_params['params']['Dense_1']['kernel']
    old_in1, out = ker_pre1.shape
    new_in1, _ = ker_init1.shape
    # pad zeros for new wait-feature row
    if new_in1 > old_in1:
        pad_rows = jnp.zeros((new_in1 - old_in1, out), dtype=ker_pre1.dtype)
        new_kernel1 = jnp.vstack([ker_pre1, pad_rows])
    else:
        new_kernel1 = ker_pre1[:new_in1]
    params['params']['Dense_1']['kernel'] = new_kernel1
    if bias_pre1 is not None:
        params['params']['Dense_1']['bias'] = bias_pre1

    # --- Dense_2: final output layer ---
    if 'Dense_2' in pretrained['params'] and \
       pretrained['params']['Dense_2']['kernel'].shape == init_params['params']['Dense_2']['kernel'].shape:
        params['params']['Dense_2'] = pretrained['params']['Dense_2']

    return freeze(params)


def mask_grads(grads, freeze_layers):
    """
    Zero out gradients for specified Dense layers, 
    but for Dense_1 only freeze the pretrained rows.
    """
    grads = unfreeze(grads)
    for layer in freeze_layers:
        if layer not in grads['params']:
            continue

        # for Dense_1: only freeze the old rows
        if layer == 'Dense_1':
            ker = grads['params'][layer]['kernel']   # shape [new_in1, out]
            n_rows, n_out = ker.shape

            # assume exactly 1 “new” row was added at the bottom
            n_pretrained = n_rows - 1
            # build a mask: zeros for pretrained rows, ones for the new row
            mask_kernel = jnp.vstack([
                jnp.zeros((n_pretrained, n_out), dtype=ker.dtype),
                jnp.ones ((1,          n_out), dtype=ker.dtype),
            ])

            # apply mask to kernel grads
            grads['params'][layer]['kernel'] = ker * mask_kernel

            # continue to freeze the bias entirely (no new bias entries)
            if 'bias' in grads['params'][layer]:
                grads['params'][layer]['bias'] = jnp.zeros_like(
                    grads['params'][layer]['bias']
                )

        # --- DEFAULT: freeze entire layer ---
        else:
            grads['params'][layer] = {
                name: jnp.zeros_like(val)
                for name, val in grads['params'][layer].items()
            }

    return freeze(grads)


# --- Batch definition ---
class Batch(NamedTuple):
    state: TaxiState        # [T, B]
    action: jnp.ndarray     # [T, B]
    reward: jnp.ndarray     # [T, B]
    next_state: TaxiState   # [T, B]
    done: jnp.ndarray       # [T, B]
    pickup: jnp.ndarray     # [T, B]
    wait: jnp.ndarray       # [T, B]
    travel: jnp.ndarray     # [T, B]


# --- State reset helper (unchanged) ---
# Precompile a batched init to reset B environments in one go
batched_init = jax.jit(
    jax.vmap(
        init_env,
        in_axes=(0, None, None, None),
        out_axes=(0, 0)
    )
)
@jax.jit
def maybe_reset(
    state: TaxiState,
    key: jnp.ndarray,
    fixed_starts: jnp.ndarray,
    fixed_pickups: jnp.ndarray
) -> Tuple[TaxiState, jnp.ndarray]:
    """
    Vectorized reset: for each instance in the batch, if state.done is True,
    replace it with a freshly initialized state, otherwise keep the existing one.
    Returns the new batch-state and an updated PRNGKey.
    """
    B = state.current_node.shape[0]
    keys = random.split(key, B + 1)
    new_key, subkeys = keys[0], keys[1:]

    # Batch-init B new states (states and new keys)
    resets, _ = batched_init(
        subkeys,
        fixed_starts,
        fixed_pickups,
        state.neighbor_mask
    )

    # Helper that broadcasts `state.done` to match old/new shapes
    def choose(old, new):
        cond = state.done
        # Expand cond dims to match `old`'s rank
        # e.g., if old.shape = (B, M), reshape cond to (B, 1)
        shape = cond.shape + (1,) * (old.ndim - cond.ndim)
        cond_b = cond.reshape(shape)
        return jnp.where(cond_b, new, old)

    new_state = TaxiState(
        current_node  = choose(state.current_node,  resets.current_node),
        pickup_node   = choose(state.pickup_node,   resets.pickup_node),
        done          = jnp.zeros_like(state.done),
        step_count    = choose(state.step_count,    resets.step_count),
        neighbor_mask = choose(state.neighbor_mask, resets.neighbor_mask),
        time          = choose(state.time,          resets.time),
    )
    return new_state, new_key

# --- Rollout function ---
def get_batched_rollout_q(
    model: nn.Module,
    obs_fn_batch: Callable[[TaxiState], Dict[str, jnp.ndarray]],
    fixed_starts: jnp.ndarray,
    fixed_pickups: jnp.ndarray,
    *, num_steps: int = 128
):
    @jax.jit
    def rollout(env: JAXRideEnv,
                init_states: TaxiState,
                key: jnp.ndarray,
                params: dict,
                epsilon: float
               ):
        B = init_states.current_node.shape[0]
        keys = random.split(key, num_steps + 1)

        def step_fn(state: TaxiState, k):
            k1, k2, k3 = random.split(k, 3)
            # Compute obs
            obs = obs_fn_batch(state)
            sf = obs['state_feats']         # [B, D_state]
            af = obs['action_feats']        # [B, max_deg, D_action]
            mask = state.neighbor_mask       # [B, max_deg]

            # Q-values and action selection
            q_vals = model.apply(params, sf, af, mask)  # [B, max_deg]
            greedy = jnp.argmax(q_vals, axis=-1)
            # uniform random over valid
            probs = mask / jnp.sum(mask, axis=-1, keepdims=True)
            rand = random.categorical(k1, jnp.log(probs))
            explore = random.uniform(k2, (B,)) < epsilon
            action = jnp.where(explore, rand, greedy)

            next_s, rew, pickup, info = env.step(state, action)
            wait = info['wait']               # [B]
            travel = info['travel']           # [B]
            done = next_s.done
            # reset finished
            next_s, _ = maybe_reset(next_s, k3, fixed_starts, fixed_pickups)
            return next_s, (state, action, rew, next_s, done, pickup, wait, travel)

        final_state, traj = jax.lax.scan(step_fn, init_states, keys)
        s, a, r, s2, d, p, waits, travels = traj
        batch = Batch(s, a, r, s2, d, p, waits, travels)
        return batch, final_state

    return rollout


# --- Training loop ---
def train(
    env: JAXRideEnv,
    init_state_fn: Callable[[], TaxiState],
    obs_fn_batch: Callable[[TaxiState], Dict[str, jnp.ndarray]],
    key: jnp.ndarray,
    fixed_starts: Sequence[int],
    fixed_pickups: Sequence[int],
    pretrain_ckpt: str = None,
    num_steps: int = 128,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    epsilon_start: float = 0.2,
    epsilon_end: float = 0.01,
) -> dict:
    # Initialize network and parameters
    max_deg = env.max_deg
    freeze_epochs = 0.1*epochs
    model = QNetwork(max_deg=max_deg)

        # Dummy init to get param shapes
    # D_state = 2(curr_xy)+2(pick_xy)+2(delta_xy)+1(norm_step) = 7
    D_state  = 7
    # D_action = 2 (travel_time, wait_time)
    D_action = 2
    dummy_sf   = jnp.zeros((1, D_state), dtype=jnp.float32)
    dummy_af   = jnp.zeros((1, max_deg, D_action), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, max_deg), dtype=jnp.bool_)
    init_params = model.init(key, dummy_sf, dummy_af, dummy_mask)

    # Merge pretrained if provided
    if pretrain_ckpt:
        with open(pretrain_ckpt, 'rb') as f:
            pretrained = pickle.load(f)
        # start from init_params
        params = init_params
        # merge old weights, zero‑init new wait feature
        params = adapt_pretrained_zeroinit(params, pretrained, init_params)
    else:
        params = init_params
    target_params = params

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    # For percentage of pickups
    pickup_count = 0
    done_count = 0

    rollout = get_batched_rollout_q(model, obs_fn_batch,
                                    jnp.array(fixed_starts), jnp.array(fixed_pickups),
                                    num_steps=num_steps)

    # Loss & update function
    @jax.jit
    def train_step(params, target_params, opt_state,
                   sf, af, mask,
                   act, rew, sf2, af2, mask2, done,
                   freeze_mask: bool):
        def loss_fn(p):
            q = model.apply(p, sf, af, mask)               # [N, max_deg]
            q_taken = jnp.take_along_axis(q, act[:,None], -1).squeeze(-1)
            qn = model.apply(target_params, sf2, af2, mask2)
            max_qn = jnp.max(qn, axis=-1)
            td = rew + gamma * max_qn * (1.0 - done)
            td_tgt = jax.lax.stop_gradient(td)
            return jnp.mean((q_taken - td_tgt)**2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        # during early epochs, freeze Dense_0 and Dense_1
        # freeze pretrained layers if freeze_mask True
        grads = lax.cond(
            freeze_mask,
            lambda g: mask_grads(g, ['Dense_0', 'Dense_1']),
            lambda g: g,
            grads
        )
        updates, new_opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Training epochs
    logger = TrainingLogger()
    states = init_state_fn()
    for ep in range(1, epochs+1):
        freeze_mask = (ep <= freeze_epochs)
        t0 = time.time()
        eps = epsilon_start + (epsilon_end - epsilon_start) * (ep/epochs)
        key, subkey = random.split(key)
        batch, states = rollout(env, states, subkey, params, eps)

        # Compute observations
        obs = obs_fn_batch(batch.state)
        obs2 = obs_fn_batch(batch.next_state)
        sf = obs['state_feats']         # [T,B,D_state]
        af = obs['action_feats']        # [T,B,max_deg,3]
        mask = batch.state.neighbor_mask
        sf2 = obs2['state_feats']
        af2 = obs2['action_feats']
        mask2 = batch.next_state.neighbor_mask

        # Flatten
        T,B = batch.action.shape
        N   = T*B
        sf = sf.reshape((N,-1));    
        af  = af.reshape((N, max_deg, D_action))
        mask = mask.reshape((N,max_deg))
        act = batch.action.reshape((N,))
        rew = batch.reward.reshape((N,))
        sf2 = sf2.reshape((N,-1))  
        af2 = af2.reshape((N,max_deg,D_action))
        mask2 = mask2.reshape((N,max_deg))
        done = batch.done.reshape((N,))

        waits   = batch.wait.reshape((T * B,)).tolist()
        travels = batch.travel.reshape((T * B,)).tolist()
        logger.log(float(jnp.sum(rew)), waits)

        # Count pickups
        per_env_picked  = jnp.any(batch.pickup,  axis=0)
        per_env_done = jnp.any(batch.done, axis=0)
        n_picked  = int(jnp.sum(per_env_picked))
        n_done   = int(jnp.sum(per_env_done))
        pickup_count  += n_picked
        done_count   += n_done

        pickup_rate = 100.0 * pickup_count / done_count
        epoch_rate = n_picked / n_done

        # Shuffle indices
        idx = random.permutation(key, N)
        for i in range(0, N, batch_size):
            mb = idx[i:i+batch_size]
            params, opt_state, loss = train_step(
                params, target_params, opt_state,
                sf[mb], af[mb], mask[mb],
                act[mb], rew[mb], sf2[mb], af2[mb], mask2[mb], done[mb],
                freeze_mask
            )

        # Sync target network
        target_params = params
        dt = time.time() - t0
        # print(f"Ep {ep}/{epochs} loss={loss:.4f} t={dt:.2f}s - Pickup Rate: {pickup_rate:.2f}% - Epoch Rate: {epoch_rate:.2f}%")

    logger.save_plots(out_dir="plots")   # writes reward_per_episode.png and avg_wait_per_episode.png


    # Save final params
    with open('trained_q_params.pkl','wb') as f:
        pickle.dump(params, f)
    print("Saved trained Q-network parameters.")
    return params
