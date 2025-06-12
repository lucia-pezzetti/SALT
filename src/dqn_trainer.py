import pickle
import time
from typing import NamedTuple, Callable, Sequence, Dict, Tuple
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax, tree_util
import optax
from flax import linen as nn
from flax.training import checkpoints
from flax.core.frozen_dict import freeze, unfreeze
from functools import partial

from taxi_env import init_env, TaxiState, JAXRideEnv
from visualization_utils import TrainingLogger

class QNetwork(nn.Module):
    ctx_dim: int
    node_hidden: int = 64
    pool: str = "mean"
    num_actions: int = None # max degree of the graph

    def setup(self):
        # Encoders for state and action features
        self.state_enc  = nn.Dense(self.ctx_dim)
        self.action_enc = nn.Dense(self.ctx_dim)
        # Shared MLP to embed individual nodes (actions)
        self.node_enc = nn.Sequential([
            nn.Dense(self.node_hidden),
            nn.relu,
            nn.Dense(self.node_hidden),
            nn.relu,
        ])
        # Projection from pooled node embeddings to global context dimension
        self.global_proj = nn.Dense(self.ctx_dim)
        # Untied heads: one weight vector and bias per action
        self.q_head_w = self.param(
            'q_head_w',
            nn.initializers.lecun_normal(),
            (self.num_actions, self.ctx_dim)
        )
        self.q_head_b = self.param(
            'q_head_b',
            nn.initializers.zeros,
            (self.num_actions,)
        )

    def __call__(self, s_feats, a_feats, mask, g_feats=None):
        """
        Args:
            s_feats: [B, state_feat_dim]  State feature vectors
            a_feats: [B, max_deg, action_feat_dim]  Action feature vectors per slot
            mask:    [B, max_deg] boolean mask of valid actions
            g_feats: (unused) Optional global features
        Returns:
            q: [B, max_deg]  Q-values per action slot
        """
        # Encode state and actions
        s_ctx = self.state_enc(s_feats)                   # [B, ctx_dim]
        a_ctx = self.action_enc(a_feats)                  # [B, max_deg, ctx_dim]

        # Embed nodes via shared MLP
        B, max_deg, _ = a_feats.shape
        flat_a = a_feats.reshape(B * max_deg, -1)
        node_emb = self.node_enc(flat_a)                  # [B*max_deg, node_hidden]
        node_emb = node_emb.reshape(B, max_deg, self.node_hidden)

        # Pool across nodes
        if self.pool == "mean":
            pooled = node_emb.mean(axis=1)                # [B, node_hidden]
        elif self.pool == "max":
            pooled = node_emb.max(axis=1)                 # [B, node_hidden]
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")

        # Global context projection
        g_ctx = self.global_proj(pooled)                  # [B, ctx_dim]
        g_ctx = jnp.expand_dims(g_ctx, axis=1)            # [B, 1, ctx_dim]

        # Combine contexts and nonlinearity
        x = s_ctx[:, None, :] + a_ctx + g_ctx             # [B, max_deg, ctx_dim]
        x = nn.relu(x)                                    # [B, max_deg, ctx_dim]

        # Compute untied Q-heads (one per action index)
        q_raw = jnp.einsum('bkc,kc->bk', x, self.q_head_w) + self.q_head_b  # [B, max_deg]

        # Mask invalid actions
        q = jnp.where(mask, q_raw, -1e9)
        return q

    

class ReplayBuffer:
    def __init__(self, max_size, state_dim, max_deg, action_dim, global_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        # preallocate host (numpy) arrays
        self.sf   = np.zeros((max_size, state_dim),       dtype=np.float32)
        self.af   = np.zeros((max_size, max_deg, action_dim), dtype=np.float32)
        self.mask = np.zeros((max_size, max_deg),           dtype=bool)
        self.gf   = np.zeros((max_size, global_dim),      dtype=np.float32)
        self.act  = np.zeros((max_size,), dtype=np.int32)
        self.rew  = np.zeros((max_size,), dtype=np.float32)
        self.sf2   = np.zeros((max_size, state_dim),       dtype=np.float32)
        self.af2   = np.zeros((max_size, max_deg, action_dim), dtype=np.float32)
        self.mask2 = np.zeros((max_size, max_deg),           dtype=bool)
        self.gf2   = np.zeros((max_size, global_dim),      dtype=np.float32)
        self.done = np.zeros((max_size,), dtype=np.float32)

    def add(self, sf, af, mask, gf, act, rew, sf2, af2, mask2, gf2, done):
        i = self.ptr
        self.sf[i]   = np.asarray(sf)
        self.af[i]   = np.asarray(af)
        self.mask[i] = np.asarray(mask)
        self.gf[i]   = np.asarray(gf)
        self.act[i]  = int(act)
        self.rew[i]  = float(rew)
        self.sf2[i]   = np.asarray(sf2)
        self.af2[i]   = np.asarray(af2)
        self.mask2[i] = np.asarray(mask2)
        self.gf2[i]   = np.asarray(gf2)
        self.done[i]  = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return dict(
            sf   = jnp.array(self.sf  [idx]),
            af   = jnp.array(self.af  [idx]),
            mask = jnp.array(self.mask[idx]),
            gf   = jnp.array(self.gf  [idx]),
            act  = jnp.array(self.act [idx]),
            rew  = jnp.array(self.rew [idx]),
            sf2   = jnp.array(self.sf2  [idx]),
            af2   = jnp.array(self.af2  [idx]),
            mask2 = jnp.array(self.mask2[idx]),
            gf2   = jnp.array(self.gf2  [idx]),
            done  = jnp.array(self.done[idx]),
        )


def adapt_pretrained_zeroinit(params, pretrained, init_params):
    """
    Copy over old weights, zero‑initialize extra rows when input dims have grown,
    remapping pretrained Dense layers to the new network layout.
    """
    params = unfreeze(params)
    pretrained = unfreeze(pretrained)
    init_params = unfreeze(init_params)

    # --- Dense_0: state embedding ---
    if 'Dense_0' in pretrained['params']:
        ker_pre0 = pretrained['params']['Dense_0']['kernel']
        bias_pre0 = pretrained['params']['Dense_0'].get('bias')
        ker_init0 = init_params['params']['Dense_0']['kernel']
        old_in0, dim_ctx = ker_pre0.shape
        new_in0, _ = ker_init0.shape
        if new_in0 > old_in0:
            pad = jnp.zeros((new_in0 - old_in0, dim_ctx), dtype=ker_pre0.dtype)
            new_ker0 = jnp.vstack([ker_pre0, pad])
        else:
            new_ker0 = ker_pre0[:new_in0]
        params['params']['Dense_0']['kernel'] = new_ker0
        if bias_pre0 is not None:
            params['params']['Dense_0']['bias'] = bias_pre0

    # --- Dense_2: action embedding (from pretrained Dense_1) ---
    if 'Dense_1' in pretrained['params'] and 'Dense_2' in init_params['params']:
        ker_pre1 = pretrained['params']['Dense_1']['kernel']
        bias_pre1 = pretrained['params']['Dense_1'].get('bias')
        ker_init2 = init_params['params']['Dense_2']['kernel']
        old_in1, out = ker_pre1.shape
        new_in1, _ = ker_init2.shape
        if new_in1 > old_in1:
            pad = jnp.zeros((new_in1 - old_in1, out), dtype=ker_pre1.dtype)
            new_ker1 = jnp.vstack([ker_pre1, pad])
        else:
            new_ker1 = ker_pre1[:new_in1]
        params['params']['Dense_2']['kernel'] = new_ker1
        if bias_pre1 is not None:
            params['params']['Dense_2']['bias'] = bias_pre1

    # --- Dense_3: final output layer (from pretrained Dense_2) ---
    if 'Dense_2' in pretrained['params'] and 'Dense_3' in init_params['params']:
        ker_pre2 = pretrained['params']['Dense_2']['kernel']
        bias_pre2 = pretrained['params']['Dense_2'].get('bias')
        ker_init3 = init_params['params']['Dense_3']['kernel']
        # ensure shape match
        if ker_pre2.shape == ker_init3.shape:
            params['params']['Dense_3']['kernel'] = ker_pre2
            if bias_pre2 is not None:
                params['params']['Dense_3']['bias'] = bias_pre2

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
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    ) -> dict:


    # --- State reset helper ---
    # Precompile a batched init to reset B environments in one go
    neighbor_mask_static = env.neighbor_mask_static 
    batched_init = jax.jit(
            jax.vmap(
                partial(init_env, neighbor_mask_static=neighbor_mask_static),
                in_axes=(0, None, None),
                out_axes=(0, 0),
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
            fixed_pickups
        )

        # Helper that broadcasts `state.done` to match old/new shapes
        def choose(old, new):
            cond = state.done
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
            keys = random.split(key, num_steps + 1)

            def step_fn(state: TaxiState, k):
                k1, k2, k3 = random.split(k, 3)
                # Compute obs
                obs = obs_fn_batch(state)
                sf = obs['state_feats']         # [B, D_state]
                af = obs['action_feats']        # [B, max_deg, D_action]
                mask = state.neighbor_mask       # [B, max_deg]
                gf = obs['global_feats']        # [B, D_global]

                # Q-values and action selection
                q_vals = model.apply(params, sf, af, mask, gf)  # [B, max_deg]
                greedy = jnp.argmax(q_vals, axis=-1)

                # uniform random over valid
                probs = mask / jnp.sum(mask, axis=-1, keepdims=True)
                rand = random.categorical(k1, jnp.log(probs))
                explore = random.uniform(k2, (state.current_node.shape[0],)) < epsilon
                action = jnp.where(explore, rand, greedy)
                # jax.debug.print("State: {}, mask: {} action: {}, explore: {}, probs: {}, greedy: {}, rand: {}", 
                #                 state.current_node, mask, action, explore, probs, greedy, rand)
                # jax.debug.print("States: {}, Q-values: {}", state, q_vals)

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

    
    # Initialize network and parameters
    max_deg = env.max_deg
    global_dim = env.global_state_dim
    freeze_epochs = 0.1 * epochs if pretrain_ckpt is not None else 0
    # D_state = 2(curr_xy)+2(pick_xy)+2(delta_xy) = 6
    D_state  = 6
    # D_action = 2 (travel_time, wait_time)
    D_action = 2
    # D_global = 64
    # model = QNetwork(dim_ctx=128, dim_action=D_action, dim_global=D_global)
    # model = QNetwork(ctx_dim=128, node_hidden=64, pool="mean")
    model = QNetwork(ctx_dim=128, node_hidden=64, pool="mean", num_actions=max_deg)

    dummy_sf   = jnp.zeros((1, D_state), dtype=jnp.float32)
    dummy_af   = jnp.zeros((1, max_deg, D_action), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, max_deg), dtype=jnp.bool_)
    dummy_gf   = jnp.zeros((1, global_dim), dtype=jnp.float32)
    init_params = model.init(key, dummy_sf, dummy_af, dummy_mask, dummy_gf)

    # Merge pretrained if provided
    if pretrain_ckpt:
        with open(pretrain_ckpt, 'rb') as f:
            pretrained = pickle.load(f)
        params = init_params
        params = adapt_pretrained_zeroinit(params, pretrained, init_params)
    else:
        params = freeze(init_params)
    target_params = params

    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    opt_state = opt.init(params)


    # For percentage of pickups
    # pickup_count = 0
    # done_count = 0

    rollout = get_batched_rollout_q(model, obs_fn_batch,
                                    jnp.array(fixed_starts), jnp.array(fixed_pickups),
                                    num_steps=num_steps)
    
    # Set up logger and fixed validation batch
    logger = TrainingLogger()
    key, vkey = random.split(key)
    init_states_val = init_state_fn()
    batch_val, _  = rollout(env, init_states_val, vkey, params, 0.0)
    obs_val       = obs_fn_batch(batch_val.state)
    T_val, B_val  = batch_val.action.shape
    N_val         = T_val * B_val
    sf_val = obs_val['state_feats'].reshape((N_val, -1))
    af_val = obs_val['action_feats'].reshape((N_val, max_deg, D_action))
    mask_val = batch_val.state.neighbor_mask.reshape((N_val, max_deg))
    gf_val = obs_val['global_feats'].reshape((N_val, global_dim))
    act_val = batch_val.action.reshape((N_val,))

    # Metrics buffers
    loss_history = []
    update_norms = []
    q_stabilities = []
    q_prev_val = None

    def train_step(params, target_params, opt_state,
                   sf, af, mask, gf,
                   act, rew, sf2, af2, mask2, gf2, 
                   done, freeze_mask: bool):
        def loss_fn(p, tp):
            q = model.apply(p, sf, af, mask, gf)               # [N, max_deg]
            q_taken = jnp.take_along_axis(q, act[:,None], -1).squeeze(-1)
            qn = model.apply(tp, sf2, af2, mask2, gf2)
            max_qn = jnp.max(qn, axis=-1)
            td = rew + gamma * max_qn * (1.0 - done)
            td_tgt = jax.lax.stop_gradient(td)
            return jnp.mean((q_taken - td_tgt)**2)

        loss, grads = jax.value_and_grad(loss_fn)(params, target_params)
        # during early epochs, freeze Dense_0 and Dense_1
        # freeze pretrained layers if freeze_mask True
        grads = lax.cond(
            freeze_mask,
            lambda g: mask_grads(g, ['Dense_0', 'Dense_1']),
            lambda g: g,
            grads
        )
        # grads, grad_norm = optax.clip_by_global_norm(1.0)(grads)
        updates, new_opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    train_step = jax.jit(train_step, static_argnames=('freeze_mask',))

    buffer = ReplayBuffer(
        max_size=100_000,
        state_dim=D_state,
        max_deg=max_deg,
        action_dim=D_action,
        global_dim=global_dim
    )

    # Training epochs
    states = init_state_fn()
    for ep in range(1, epochs+1):
        freeze_mask = (ep <= freeze_epochs)
        # t0 = time.time()
        eps = epsilon_start + (epsilon_end - epsilon_start) * (ep/epochs)
        key, subkey = random.split(key)
        batch, states = rollout(env, states, subkey, params, eps)

        # Compute observations
        obs = obs_fn_batch(batch.state)
        obs2 = obs_fn_batch(batch.next_state)
        sf = obs['state_feats']         # [T,B,D_state]
        af = obs['action_feats']        # [T,B,max_deg,3]
        mask = batch.state.neighbor_mask
        gf = obs['global_feats']

        sf2 = obs2['state_feats']
        af2 = obs2['action_feats']
        mask2 = batch.next_state.neighbor_mask
        gf2 = obs2['global_feats']

        # Flatten
        T,B = batch.action.shape
        N   = T*B
        sf = sf.reshape((N,-1));    
        af  = af.reshape((N, max_deg, D_action))
        mask = mask.reshape((N,max_deg))
        gf = gf.reshape((N, global_dim))

        act = batch.action.reshape((N,))
        rew = batch.reward.reshape((N,))
        sf2 = sf2.reshape((N,-1))  
        af2 = af2.reshape((N,max_deg,D_action))
        mask2 = mask2.reshape((N,max_deg))
        gf2 = gf2.reshape((N, global_dim))
        done = batch.done.reshape((N,))
        waits   = batch.wait.reshape((T * B,)).tolist()
        # Removed unused variable "travels"
        logger.log(float(jnp.sum(rew)), waits)

        # Clone params before epoch updates
        params_before = params

        sf_np, af_np, mask_np, gf_np = map(np.asarray, (sf, af, mask, gf))
        act_np, rew_np      = map(np.asarray, (act, rew))
        sf2_np, af2_np, mask2_np, gf2_np = map(np.asarray, (sf2, af2, mask2, gf2))
        done_np = np.asarray(done)

        for i in range(sf_np.shape[0]):
            buffer.add(
                sf_np[i],  af_np[i],  mask_np[i],  gf_np[i],
                act_np[i], rew_np[i],
                sf2_np[i], af2_np[i], mask2_np[i], gf2_np[i],
                done_np[i]
            )

        # — sample & train from buffer —
        if buffer.size >= batch_size:
            for _ in range(4):   # you can increase this to e.g. 4 updates per rollout
                batch_sample = buffer.sample(batch_size)
                params, opt_state, loss = train_step(
                    params, target_params, opt_state,
                    **batch_sample,
                    freeze_mask=(ep <= freeze_epochs)
                )
                # Update target network parameters via Polyak averaging 
                τ = 0.005
                target_params = jax.tree_util.tree_map(
                    lambda p, tp: τ*p + (1-τ)*tp,
                    params, target_params
                )
            loss_history.append(loss.item())

        # ---- metrics ----
        # Compute validation Q-values for metrics
        q_vals_val = model.apply(params, sf_val, af_val, mask_val, gf_val)  # [N_val, max_deg]

        # Update norm: ||params - params_before||_2
        leaves_new, _ = tree_util.tree_flatten(params)
        leaves_old, _ = tree_util.tree_flatten(params_before)
        total_sq = 0.0
        for p_new, p_old in zip(leaves_new, leaves_old):
            diff = (p_new - p_old).ravel()
            total_sq += jnp.sum(diff * diff)
        update_norms.append(jnp.sqrt(total_sq).item())

        # Q-value stability
        q_sel_val = jnp.take_along_axis(q_vals_val, act_val[:,None], axis=1).squeeze()
        if q_prev_val is None:
            q_prev_val = q_sel_val
            q_stabilities.append(0.0)
        else:
            stab = jnp.mean(jnp.abs(q_sel_val - q_prev_val)).item()
            q_stabilities.append(stab)
            q_prev_val = q_sel_val

        logger.log_metrics(
            losses=loss_history,
            update_norms=update_norms,
            q_stabilities=q_stabilities
        )

        # # Count pickups
        # per_env_picked  = jnp.any(batch.pickup,  axis=0)
        # per_env_done = jnp.any(batch.done, axis=0)
        # n_picked  = int(jnp.sum(per_env_picked))
        # n_done   = int(jnp.sum(per_env_done))
        # pickup_count  += n_picked
        # done_count   += n_done

        # pickup_rate = 100.0 * pickup_count / done_count
        # epoch_rate = n_picked / n_done

        # dt = time.time() - t0
        # print(f"Ep {ep}/{epochs} loss={loss:.4f} t={dt:.2f}s - Pickup Rate: {pickup_rate:.2f}% - Epoch Rate: {epoch_rate:.2f}%")

    logger.save_plots(out_dir="plots/1layer_paramtuning")   # writes reward_per_episode.png and avg_wait_per_episode.png


    # Save final params
    with open('trained_q_params.pkl','wb') as f:
        pickle.dump(params, f)
    print("Saved trained Q-network parameters.")
    return params
