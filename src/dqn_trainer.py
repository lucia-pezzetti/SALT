import pickle
from timing_decorator import timeit
from typing import NamedTuple, Callable, Sequence, Dict, Tuple
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax, tree_util
import optax
from flax import linen as nn
from flax import struct
from flax.core.frozen_dict import freeze, unfreeze
from functools import partial

from taxi_env import init_env, TaxiState, TaxiEnv
from visualization_utils import TrainingLogger
from utils import estimate_returns_jit, EstimateReturnsState
from ot import emd
from scipy.optimize import linear_sum_assignment

from models.q_network import adapt_pretrained_zeroinit, mask_grads
from models.q_network import QNetwork
    

@struct.dataclass
class ReplayBuffer:
    """
    JAX-compatible, fully jit-compiled replay buffer.

    Usage:
        buffer = ReplayBuffer.create(max_size, state_dim, max_deg, action_dim, global_dim)
        buffer = buffer.add(sf, af, mask, gf, act, rew, sf2, af2, mask2, gf2, done)
        batch = buffer.sample(key, batch_size)
    """
    max_size: int
    ptr: int
    size: int
    sf: jnp.ndarray
    af: jnp.ndarray
    mask: jnp.ndarray
    gf: jnp.ndarray
    act: jnp.ndarray
    rew: jnp.ndarray
    sf2: jnp.ndarray
    af2: jnp.ndarray
    mask2: jnp.ndarray
    gf2: jnp.ndarray
    done: jnp.ndarray

    @classmethod
    def create(
        cls,
        max_size: int,
        state_dim: int,
        max_deg: int,
        action_dim: int,
        global_dim: int
    ):  # -> ReplayBuffer
        return cls(
            max_size=max_size,
            ptr=0,
            size=0,
            sf=jnp.zeros((max_size, state_dim), jnp.float32),
            af=jnp.zeros((max_size, max_deg, action_dim), jnp.float32),
            mask=jnp.zeros((max_size, max_deg), jnp.bool_),
            gf=jnp.zeros((max_size, global_dim), jnp.float32),
            act=jnp.zeros((max_size,), jnp.int32),
            rew=jnp.zeros((max_size,), jnp.float32),
            sf2=jnp.zeros((max_size, state_dim), jnp.float32),
            af2=jnp.zeros((max_size, max_deg, action_dim), jnp.float32),
            mask2=jnp.zeros((max_size, max_deg), jnp.bool_),
            gf2=jnp.zeros((max_size, global_dim), jnp.float32),
            done=jnp.zeros((max_size,), jnp.float32),
        )
    
    @jax.jit
    def add_batch(
        self,
        sf_batch: jnp.ndarray,      # [N, D_state]
        af_batch: jnp.ndarray,      # [N, max_deg, D_action]
        mask_batch: jnp.ndarray,    # [N, max_deg]
        gf_batch: jnp.ndarray,      # [N, D_global]
        act_batch: jnp.ndarray,     # [N]
        rew_batch: jnp.ndarray,     # [N]
        sf2_batch: jnp.ndarray,     # [N, D_state]
        af2_batch: jnp.ndarray,     # [N, max_deg, D_action]
        mask2_batch: jnp.ndarray,   # [N, max_deg]
        gf2_batch: jnp.ndarray,     # [N, D_global]
        done_batch: jnp.ndarray,    # [N]
    ) -> "ReplayBuffer":
        batch_size = sf_batch.shape[0]
        
        # Calculate indices for circular buffer
        indices = (jnp.arange(batch_size) + self.ptr) % self.max_size
        
        # Update all arrays at once using advanced indexing
        new_sf = self.sf.at[indices].set(sf_batch)
        new_af = self.af.at[indices].set(af_batch)
        new_mask = self.mask.at[indices].set(mask_batch)
        new_gf = self.gf.at[indices].set(gf_batch)
        new_act = self.act.at[indices].set(act_batch)
        new_rew = self.rew.at[indices].set(rew_batch)
        new_sf2 = self.sf2.at[indices].set(sf2_batch)
        new_af2 = self.af2.at[indices].set(af2_batch)
        new_mask2 = self.mask2.at[indices].set(mask2_batch)
        new_gf2 = self.gf2.at[indices].set(gf2_batch)
        new_done = self.done.at[indices].set(done_batch)
        
        # Update pointer and size
        new_ptr = (self.ptr + batch_size) % self.max_size
        new_size = jnp.minimum(self.size + batch_size, self.max_size)
        
        return self.replace(
            ptr=new_ptr,
            size=new_size,
            sf=new_sf, af=new_af, mask=new_mask, gf=new_gf,
            act=new_act, rew=new_rew,
            sf2=new_sf2, af2=new_af2, mask2=new_mask2, gf2=new_gf2,
            done=new_done,
        )

    @jax.jit
    def add(
        self,
        sf: jnp.ndarray,
        af: jnp.ndarray,
        mask: jnp.ndarray,
        gf: jnp.ndarray,
        act: jnp.ndarray,
        rew: jnp.ndarray,
        sf2: jnp.ndarray,
        af2: jnp.ndarray,
        mask2: jnp.ndarray,
        gf2: jnp.ndarray,
        done: jnp.ndarray,
    ) -> "ReplayBuffer":
        idx = self.ptr
        # Insert new elements
        new_sf    = self.sf.at[idx].set(sf)
        new_af    = self.af.at[idx].set(af)
        new_mask  = self.mask.at[idx].set(mask)
        new_gf    = self.gf.at[idx].set(gf)
        new_act   = self.act.at[idx].set(act)
        new_rew   = self.rew.at[idx].set(rew)
        new_sf2   = self.sf2.at[idx].set(sf2)
        new_af2   = self.af2.at[idx].set(af2)
        new_mask2 = self.mask2.at[idx].set(mask2)
        new_gf2   = self.gf2.at[idx].set(gf2)
        new_done  = self.done.at[idx].set(done)
        # Update pointer and size
        ptr = (idx + 1) % self.max_size
        size = jnp.minimum(self.size + 1, self.max_size)
        # Return new buffer state
        return self.replace(
            ptr=ptr,
            size=size,
            sf=new_sf,
            af=new_af,
            mask=new_mask,
            gf=new_gf,
            act=new_act,
            rew=new_rew,
            sf2=new_sf2,
            af2=new_af2,
            mask2=new_mask2,
            gf2=new_gf2,
            done=new_done,
        )

    @partial(jax.jit, static_argnums=(2,))
    def sample_fixed(self, key: jnp.ndarray, batch_size: int) -> dict:
        # Sample with replacement to avoid shape issues
        indices = jax.random.randint(key, (batch_size,), 0, self.size)
        
        # Sample from each buffer component
        batch_sample = {
            'sf': self.sf[indices],
            'af': self.af[indices],
            'mask': self.mask[indices],
            'gf': self.gf[indices],
            'act': self.act[indices],
            'rew': self.rew[indices],
            'sf2': self.sf2[indices],
            'af2': self.af2[indices],
            'mask2': self.mask2[indices],
            'gf2': self.gf2[indices],
            'done': self.done[indices],
        }
        
        return batch_sample

# --- Batch definition: trajectories of N agents ---
class Batch(NamedTuple):
    state: TaxiState        # [T, B]
    action: jnp.ndarray     # [T, B]
    reward: jnp.ndarray     # [T, B]
    next_state: TaxiState   # [T, B]
    done: jnp.ndarray       # [T, B]
    pickup: jnp.ndarray     # [T, B]
    wait: jnp.ndarray       # [T, B]
    travel: jnp.ndarray     # [T, B]

def emd_assignment_optimized(R_jax, B):
    """
    Optimized EMD assignment with minimal conversions
    """
    # Single conversion to NumPy for EMD computation
    R_np = np.asarray(R_jax)
    
    # Prepare EMD inputs (NumPy)
    a = np.ones(B) / B  # uniform distribution over sources
    b = np.ones(B) / B  # uniform distribution over targets
    M = -R_np  # cost matrix (negative rewards)
    
    # Compute EMD transport plan
    F = emd(a, b, M, numItermax=1000)
    
    # Extract assignment from transport plan
    col_idx = np.argmax(F, axis=1)
    
    # Single conversion back to JAX
    return jnp.array(col_idx)


# --- Training loop ---
def train(
    env: TaxiEnv,
    init_state_fn: Callable[[], TaxiState],
    obs_fn_batch: Callable[[TaxiState], Dict[str, jnp.ndarray]],
    key: jnp.ndarray,
    fixed_starts: Sequence[int],
    fixed_pickups: Sequence[int],
    estimate_state: EstimateReturnsState,
    # init_states: TaxiState = None,
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
    batched_init = jax.vmap(
                partial(init_env, neighbor_mask_static=neighbor_mask_static),
                in_axes=(0, 0, 0),     # split keys, start_idxs, pickup_idxs
                out_axes=(0, 0),
            )
    
    def make_init_env_fn(batched_init, base_key):
        """Factory function that pre-splits keys"""
        def init_env_fn(starts: jnp.ndarray, pickups: jnp.ndarray):
            # Use a deterministic key split based on input size
            num_envs = starts.shape[0]
            rng_keys = jax.random.split(base_key, num_envs)
            return batched_init(rng_keys, starts, pickups)
        return init_env_fn

    # Create once outside training loop
    base_key = jax.random.PRNGKey(42)
    init_env_fn = make_init_env_fn(batched_init, base_key)

    @jax.jit
    # @timeit
    def maybe_reset(
        state: TaxiState,
        key: jnp.ndarray,
        init_state: TaxiState = None
    ) -> Tuple[TaxiState, jnp.ndarray]:
        """
        Vectorized reset: for each instance in the batch, if state.done is True,
        replace it with a freshly initialized state, otherwise keep the existing one.
        Returns the new batch-state and an updated PRNGKey.
        """
        B = state.current_node.shape[0]     # num_agents
        keys = random.split(key, B + 1)
        new_key, subkeys = keys[0], keys[1:]

        # Batch-init B new states (states and new keys)
        def do_reset(_):
            resets, _ = batched_init(
                subkeys,
                init_state.current_node,
                init_state.pickup_node
            )
            return resets
        
        use_init = init_state is not None
        resets  = jax.lax.cond(
            use_init,
            do_reset,
            lambda _: state,  # identity case
            operand=None
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
        *, num_steps: int = 128
    ):
        @jax.jit
        # @timeit
        def rollout(env: TaxiEnv,
                    init_states: TaxiState,
                    key: jnp.ndarray,
                    params: dict,
                    epsilon: float
                ):
            keys = random.split(key, num_steps + 1)
            carry0 = (init_states, init_states)

            def step_fn(carry: Tuple[TaxiState, TaxiState], k):
                state, orig_states = carry
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
                next_s, _ = maybe_reset(next_s, k3, orig_states)
                new_carry = (next_s, orig_states)
                return new_carry, (state, action, rew, next_s, done, pickup, wait, travel)

            (final_state, _), traj = jax.lax.scan(step_fn, carry0, keys)
            s, a, r, s2, d, p, waits, travels = traj
            batch = Batch(s, a, r, s2, d, p, waits, travels)
            return batch, final_state

        return rollout

    
    # Initialize network and parameters
    max_deg = env.max_deg
    global_dim = env.global_state_dim
    freeze_epochs = 0.1 * epochs if pretrain_ckpt is not None else 0
    
    D_state  = 6        # 2(curr_xy)+2(pick_xy)+2(delta_xy)
    D_action = 2        # (travel_time, wait_time)
    
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


    rollout = get_batched_rollout_q(model, obs_fn_batch,
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

    @partial(jax.jit, static_argnames=('freeze_mask',))
    # @timeit
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
    
    # train_step = jax.jit(train_step, static_argnames=('freeze_mask',))

    buffer = ReplayBuffer.create(
        max_size=100_000,
        state_dim=D_state,
        max_deg=max_deg,
        action_dim=D_action,
        global_dim=global_dim
    )

    # Training epochs
    states = init_state_fn()
    for ep in range(1, epochs+1):
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

        buffer = buffer.add_batch(
            sf, af, mask, gf,        # Keep as JAX arrays
            act, rew,
            sf2, af2, mask2, gf2,
            done
        )

        # — sample & train from buffer —
        if buffer.size >= batch_size:
            for _ in range(4):
                key, subkey = jax.random.split(key)
                batch_sample = buffer.sample_fixed(subkey, batch_size)  # Returns JAX arrays
                params, opt_state, loss = train_step(
                    params, target_params, opt_state,
                    **batch_sample,  # No conversion needed
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

        # --- OT assignment ---
        # split one key for sampling
        key, subkey = random.split(key)
        # generate 2*num_agents subkeys
        all_subkeys = random.split(subkey, 2 * B)
        start_keys, pickup_keys = jnp.split(all_subkeys, 2)

        # sample uniformly from the pool
        new_starts  = jnp.array([random.choice(k, fixed_starts, ())
                                for k in start_keys])    # shape [B]
        new_pickups = jnp.array([random.choice(k, fixed_pickups, ())
                         for k in pickup_keys])   # shape [B]

        R = estimate_returns_jit(
            env,
            params,
            model,
            obs_fn_batch,
            init_env_fn,
            new_starts,
            new_pickups,
            estimate_state,
            rollout_steps=10
        )
        # Assign new starts and pickups to the batch using optimal transport
        # a = np.ones((B,)) / B  # uniform distribution over starts
        # b = np.ones((B,)) / B  # uniform distribution over pickups    
        # M = -np.asarray(R)

        # F = emd(a, b, M, numItermax=1000)
        # col_idx = np.argmax(F, axis=1)
        # _, col_idx = linear_sum_assignment(-R)
        col_idx = emd_assignment_optimized(R, B)
        # col_idx is a 1D array of indices that maps each start to a pickup
        pickups = new_pickups[col_idx]
        # jax.debug.print("Epoch {}: new starts={}, new_pickups={}, pickups={}", ep, new_starts, new_pickups, pickups)


        # Assign new starts and pickups to the batch
        states, _ = batched_init(start_keys, new_starts, pickups)

        # if ep > 19000:
        #     jax.debug.print("Epoch {}: new starts={}, pickups={}", ep, new_starts, pickups)
        if ep % 100 == 0:
            jax.clear_caches()

    logger.save_plots(out_dir="plots/6layers_100offset")   # writes reward_per_episode.png and avg_wait_per_episode.png


    # Save final params
    # jax.debug.print("Saving final parameters to 'trained_q_params.pkl'...")
    with open('trained_q_params.pkl','wb') as f:
        pickle.dump(params, f)
    # print("Saved trained Q-network parameters.")
    return params