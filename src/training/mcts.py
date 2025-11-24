import mctx
import haiku as hk
import jax
import jax.numpy as jnp
from jax import random as jax_random
from jax import jit, vmap, grad, profiler
from functools import partial

import numpy as np
import optax
import chex
from typing import Optional, Sequence

# --- Import your Taxi environment ---
from taxi_env import init_env, TaxiState

# map activation names if you still use a value network
activation_dict = {"relu": jax.nn.relu, "silu": jax.nn.silu, "elu": jax.nn.elu}

# --- Value network ---
class V_function(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config['num_hidden_units']
        self.num_hidden_layers = config['num_hidden_layers']
        self.activation = activation_dict[config['activation']]
        
    def __call__(self, obs):
        x = jnp.ravel(obs)
        for _ in range(self.num_hidden_layers):
            x = self.activation(hk.Linear(
                self.num_hidden_units,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                b_init=hk.initializers.Constant(0.0)
            )(x))
        return hk.Linear(
            1,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
            b_init=hk.initializers.Constant(0.0)
        )(x)[0]

class GraphAwareVFunction(hk.Module):
    
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.hidden_dim = config['num_hidden_units']
        self.num_layers = config['num_hidden_layers'] 
        self.activation = activation_dict[config['activation']]
        self.node_embedding_dim = config.get('node_embedding_dim', 64)
        self.max_nodes = config.get('max_nodes', 100)
        self.cycle_length = config.get('cycle_length', 90)  # Default cycle length
        
    def __call__(self, obs, adj_list=None, road_types=None):
        # # Node embeddings
        # current_pos = obs[..., 0].astype(jnp.int32)
        # pickup_pos = obs[..., 1].astype(jnp.int32)

        # # Learnable node embeddings
        # node_embeddings = hk.Embed(
        #     self.max_nodes, 
        #     self.node_embedding_dim,
        #     w_init=hk.initializers.TruncatedNormal(stddev=0.02)
        # )
        # current_emb = node_embeddings(current_pos)
        # pickup_emb = node_embeddings(pickup_pos)

        """
        Enhanced value function for lat/lon-based observations.
        Input: obs [..., 5] = [current_pos(2), pickup_pos(2), time(1)]
        """
        # Extract features from the 5-dimensional observation
        current_pos = obs[..., 0:2]      # [..., 2] - current position (lat/lon)
        pickup_pos = obs[..., 2:4]       # [..., 2] - pickup position (lat/lon)
        time = obs[..., 4:5]             # [..., 1] - time
        
        # Normalize time feature
        time = time / self.cycle_length  # Normalize by typical cycle length
        
        # Normalize distance (assuming max distance is around 1.0 for normalized coordinates)
        # distance = distance / 1.0  # Could be made configurable
        
        # # Normalize angle to [-1, 1] range
        # angle = angle / jnp.pi  # Convert from [-π, π] to [-1, 1]

        # Combine all features
        features = jnp.concatenate([
            current_pos,    # [..., 2] - current position
            pickup_pos,     # [..., 2] - pickup position
            time            # [..., 1] - time
        ], axis=-1)  # Total: [..., 5]
        
        # Layer normalization for stability
        features = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(features)

        # Process through MLP
        x = features
        for i in range(self.num_layers):
            residual = x if x.shape[-1] == self.hidden_dim else None
            x = hk.Linear(
                self.hidden_dim,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                b_init=hk.initializers.Constant(0.0)
            )(x)
            x = self.activation(x)
            # Layer norm after activation
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            if residual is not None:
                x = x + residual  # Residual connection
                
        # Final layer with small initialization for stability
        output = hk.Linear(
            1,
            w_init=hk.initializers.VarianceScaling(0.1, "fan_in", "truncated_normal"),
            b_init=hk.initializers.Constant(0.0)
        )(x).squeeze(-1)
        
        # Clip output
        return jnp.clip(output, -5000.0, 5000.0)

def epsilon_greedy_qvalue_prior(env, V_apply, V_target_params, obs_fn_batch, state: TaxiState, 
                                         epsilon: float, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Epsilon-greedy action selection with Q-value prior.
    """
    mask = state.neighbor_mask
    
    # Compute next states and rewards for all actions
    def step_action(action_idx):
        return env.step(state, action_idx)
    
    action_indices = jnp.arange(env.max_deg)
    next_states, rewards, dones, infos = vmap(step_action)(action_indices)
    
    # Get observations for all next states
    next_obs = obs_fn_batch(next_states)
    
    # Compute values for all next states
    next_values = vmap(V_apply, in_axes=(None, 0))(V_target_params, next_obs)

    # Compute Q-values: Q(s,a) = r + gamma * V(s') * (1 - done)
    qvalues = rewards + env.gamma * next_values * (1.0 - dones.astype(float))
    
    # Clip and mask invalid actions
    qvalues = jnp.clip(qvalues, -1e6, 1e6)
    qvalues = jnp.where(mask, qvalues, -jnp.inf)
    
    # Find best action
    best_action = jnp.argmax(qvalues)
    
    # Create epsilon-greedy logits
    num_valid = jnp.sum(mask)
    uniform_prob = epsilon / jnp.maximum(num_valid, 1.0)  # Avoid division by zero
    greedy_prob = uniform_prob + (1.0 - epsilon)
    
    # # Simple uniform prior over valid actions (much faster than Q-value computation)
    # uniform_prob = 1.0 / jnp.maximum(num_valid, 1.0)
    
    # Create logits: uniform for valid actions, -inf for invalid
    logits = jnp.where(mask, jnp.log(uniform_prob + 1e-8), -jnp.inf)
    # Add extra probability to best action
    logits = logits.at[best_action].set(jnp.log(greedy_prob + 1e-8))

    # Debug: print Q-value components to understand the variation
    qmax = jnp.max(jnp.where(mask, qvalues, -jnp.inf))
    qmin = jnp.min(jnp.where(mask, qvalues, jnp.inf))
    reward_max = jnp.max(rewards)
    reward_min = jnp.min(rewards)
    next_val_max = jnp.max(next_values)
    next_val_min = jnp.min(next_values)
    any_done = jnp.any(dones)
    
    # jax.debug.print(
    #     "[prior] num_valid={nv} qmax={qmax:.2f} qmin={qmin:.2f} rmax={rmax:.2f} rmin={rmin:.2f} Vmax={vmax:.2f} Vmin={vmin:.2f} any_done={done}",
    #     nv=jnp.sum(mask),
    #     qmax=qmax,
    #     qmin=qmin,
    #     rmax=reward_max,
    #     rmin=reward_min,
    #     vmax=next_val_max,
    #     vmin=next_val_min,
    #     done=any_done,
    # )
    
    return logits

# Batch version of the efficient implementation
def batch_epsilon_greedy_qvalue_prior(env, V_apply, V_params, obs_fn_batch, 
                                               states, epsilon: float, keys):
    """Efficient batch version"""
    return vmap(epsilon_greedy_qvalue_prior, in_axes=(None, None, None, None, 0, None, 0))(
        env, V_apply, V_params, obs_fn_batch, states, epsilon, keys)



# ot matching function using the estimated returns as costs
def estimate_returns_batch(keys, V_apply, obs_fn_batch, batch_init, 
                          V_params, starts, pickups):
    
    N = len(starts) # number of starts == number of pickups
    
    # Create all combinations of starts and pickups
    start_grid, pickup_grid = jnp.meshgrid(starts, pickups, indexing='ij')
    start_flat = start_grid.flatten()
    pickup_flat = pickup_grid.flatten()
    
    # Initialize all combinations
    all_states, _ = batch_init(keys, start_flat, pickup_flat)
    
    # Get observations for all states
    obs = obs_fn_batch(all_states)
    
    # Ensure obs has the correct shape [batch_size, obs_dim]
    # jnp.atleast_2d ensures 2D: [obs_dim] -> [1, obs_dim], [batch, obs_dim] -> [batch, obs_dim]
    # This handles the case where obs_fn_batch might return a 1D array
    obs = jnp.atleast_2d(obs)
    
    # Get value estimates directly from the learned value function
    batch_V = vmap(V_apply, in_axes=(None, 0))
    returns = batch_V(V_params, obs.astype(float))
    
    # Reshape to [num_starts, num_pickups] matrix
    return returns.reshape(N, N)


# --- Init function (vectorized) ---
def get_init_fn(env, config, obs_fn_single):
    # vmap init_env over (key, start, pickup)
    batch_init = vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0))

    def init_fn(key):
        # split keys
        keys = jax.random.split(key, 3*config['batch_size'])
        subkeys_starts = keys[:config['batch_size']]
        subkeys_pickups = keys[config['batch_size']:2*config['batch_size']]
        subkeys = keys[2*config['batch_size']:]
        # fixed starts & pickups from config
        starts = jnp.stack([jax_random.choice(k, env.fixed_starts) for k in subkeys_starts])
        pickups = jnp.stack([jax_random.choice(k, env.fixed_pickups) for k in subkeys_pickups])
        # initialize states
        env_states, _ = batch_init(subkeys, starts, pickups)

        # get a sample obs for net init
        dummy_state, _ = init_env(
            jax_random.PRNGKey(0), starts[0], pickups[0], env.neighbor_mask_static
        )
        dummy_obs = obs_fn_single(dummy_state)
        dummy_mask = dummy_state.neighbor_mask

        # build value network
        # V_net = hk.without_apply_rng(hk.transform(lambda obs: V_function(config)(obs)))
        V_net = hk.without_apply_rng(hk.transform(lambda obs: GraphAwareVFunction(config)(obs)))
        key, sk = jax.random.split(key)
        V_params = V_net.init(sk, dummy_obs)
        V_target_params = V_params  # for now, same as online params
        V_func = V_net.apply

        V_opt = optax.adamw(
            config['V_alpha'],
            eps=config['eps_adam'],
            b1=config['b1_adam'],
            b2=config['b2_adam'],
            weight_decay=config['wd_adam']
        )
        V_opt_init    = V_opt.init
        V_opt_update  = V_opt.update
        get_V_params  = optax.apply_updates

        V_opt_state = V_opt_init(V_params)

        return key, env_states, V_func, V_opt_state, V_opt_update, get_V_params, V_target_params

    return init_fn

# --- Recurrent fn for tree-search ---
def get_recurrent_fn(
    env,
    V_apply,
    obs_fn_batch,
    epsilon_schedule_fn=None,
    curriculum_steps: int = 100_000,
):
    # pre‑compute next_hop[src, tgt] - the neighbor of i that lies on some shortest path to j.
    hop_dist = jnp.array(env.hop_distances)        # Keep on GPU, shape [N, N]
    N = hop_dist.shape[0]
    next_hop_np = np.zeros((N, N), dtype=int)  # Still need numpy for the loop
    
    # # Precompute action lookup table for faster node_to_action conversion
    # action_lookup_np = np.zeros((N, N), dtype=int)
    
    for src in range(N):
        # gather all 1‑hop neighbors of `src`
        neighs = jnp.where(hop_dist[src] == 1)[0]  # Keep on GPU
        for tgt in range(N):
            d = hop_dist[src, tgt]
            if d > 0:
                # pick the first neighbor whose dist-to-target is d-1
                for nei in neighs:
                    if hop_dist[nei, tgt] == d - 1:
                        next_hop_np[src, tgt] = nei
                        break
            else:
                # same node ⇒ action can be arbitrary (we'll never call it)
                next_hop_np[src, tgt] = src
        
    # print(f"next_hop_np shape: {next_hop_np.shape}")  # Only print shape, not full array
    next_hop = jax.device_put(jnp.array(next_hop_np, dtype=jnp.int32))  # JAX array, shape [N, N]

    # 2) vectorized env.step and V:
    batch_step = vmap(env.step, in_axes=(0, 0))
    batch_V    = vmap(V_apply, in_axes=(None, 0))

    @jax.jit
    def rollout_value_fn(states):
        """
        True discounted return of following the shortest-path policy until pickup.
        """
        # how many steps max?
        hop_dists     = env.hop_distances[states.current_node, states.pickup_node]  # [B]
        max_steps = jnp.max(hop_dists)                                          # scalar

        B = states.current_node.shape[0]
        init_R     = jnp.zeros(B, dtype=jnp.float32)
        init_disc  = jnp.ones(B, dtype=jnp.float32)
        init_done  = jnp.zeros(B, dtype=bool)

        # carry = (step, states, R, discount, done_mask)
        init_carry = (0, states, init_R, init_disc, init_done)

        def cond_fn(carry):
            step, _, _, _, done = carry
            # continue while we haven't hit max_steps AND at least one trajectory still alive
            return jnp.logical_and(step < max_steps, jnp.any(~done))

        def body_fn(carry):
            step, states, R, discount, done = carry


            # look up shortest-path neighbor for each instance in the batch:
            next_nodes = next_hop[ states.current_node, states.pickup_node ]   # [B] neighbor node ids

            # map neighbor node ids to action indices expected by env.step (slot in adj_list)
            def node_to_action(curr_node, target_node):
                nbrs = env.adj_list[curr_node]  # [max_deg]
                # find index where adj_list[curr_node, j] == target_node
                # size=1 guarantees a value even if padded; mask ensures only valid are used
                pos = jnp.where(nbrs == target_node, size=1)[0][0]
                return jnp.int32(pos)

            actions = jax.vmap(node_to_action)(states.current_node, next_nodes)  # [B]

            # jax.debug.print(
            #     "[rollout s={step}] curr={curr} tgt={tgt} next_hop={nh} action={act} valid?={valid} hop(curr,tgt)={hd} hop(nh,tgt)={hdn}",
            #     step=step,
            #     curr=states.current_node,
            #     tgt=states.pickup_node,
            #     nh=next_nodes,
            #     act=actions,
            #     valid=env.neighbor_mask_static[states.current_node, actions],
            #     hd=env.hop_distances[states.current_node, states.pickup_node],
            #     hdn=env.hop_distances[next_nodes, states.pickup_node],
            # )

            next_s, rew, term, _ = batch_step(states, actions)

            # accumulate only for still‑alive ones
            R        = R + discount * rew
            discount = discount * env.gamma * (1.0 - term)
            done     = done | term

            # jax.debug.print(
            #     "[rollout s={step}] rew={r} term={term} disc={disc}",
            #     step=step,
            #     r=rew,
            #     term=term,
            #     disc=discount,
            # )

            return (step + 1, next_s, R, discount, done)

        _, _, final_R, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        return final_R  # [B]
        
        # # Use precomputed distances as negative returns (shorter = better)
        # curr, target = states.current_node, states.pickup_node  # shape [B]
        # rem_dist = env.distances[curr, target]  # shape [B]
        
        # Convert distance to approximate return (negative cost)
        # Scale by typical travel time to make it comparable to learned values
        # return -rem_dist / 60.0  # shape [B]

    def recurrent_fn(params, key, actions, states):
        V_params   = params["V"]
        step_count = params.get("step", 0)
        ε          = epsilon_schedule_fn(step_count) if epsilon_schedule_fn else 0.1
        # Curriculum learning disabled
        # alpha      = jnp.clip(step_count / float(curriculum_steps), 0.0, 1.0)
        alpha = 1.0

        # 1) actual environment step
        next_states, rewards, terminals, _ = batch_step(states, actions)
        obs = obs_fn_batch(next_states)

        # 2) rollout return under shortest-path policy (curriculum learning disabled)
        rollout_vals = rollout_value_fn(next_states)  # shape [B]

        # 3) your learned V
        V_learned = batch_V(V_params, obs)            # [B] or [B,1]
        V_learned = jnp.squeeze(V_learned)            # make sure [B]
        V_learned = jnp.where(terminals, 0.0, V_learned)

        # 4) curriculum mix (disabled - use only learned V)
        value = (1.0 - alpha) * rollout_vals + alpha * V_learned
        # value = V_learned  # Use only learned value function

        # 5) ε‑greedy prior (unchanged)
        B = next_states.current_node.shape[0]
        keys = jax.random.split(key, B)
        pi_logits = batch_epsilon_greedy_qvalue_prior(
            env, V_apply, V_params, obs_fn_batch, next_states, ε, keys
        )

        return mctx.RecurrentFnOutput(
            reward       = rewards,
            discount     = (1.0 - terminals) * env.gamma,
            prior_logits = pi_logits,
            value        = value
        ), next_states

    return recurrent_fn

# def get_recurrent_fn(env, V_apply, obs_fn_batch, epsilon_schedule_fn=None, curriculum_steps: int = 100_000):
#     batch_step = vmap(env.step, in_axes=(0, 0))
#     batch_V    = vmap(V_apply,   in_axes=(None, 0))

#     # ------------------------------------------------------------------
#     # Rollout value via shortest‐path heuristic:
#     # for each next_state, look up the precomputed distance from
#     # its current node to its assigned pickup (in env.fixed_pickups),
#     # then negate to turn a “cost” into a return.
#     def rollout_value_fn(states):
#         curr, target = states.current_node, states.pickup_node                # shape [B]
#         rem_dist = env.distances[curr, target]    # shape [B]
    
#         return -rem_dist                          # shape [B]
#     # ------------------------------------------------------------------
 
#     def recurrent_fn(params, key, actions, states):
#         V_params = params.get("V", None)
 
#         current_step = params.get("step", 0)
#         epsilon = epsilon_schedule_fn(current_step) if epsilon_schedule_fn else 0.1
#         next_states, rewards, terminals, _ = batch_step(states, actions)
#         obs = obs_fn_batch(next_states)

#         # compute value via shortest‐path rollout
#         rollout_vals = rollout_value_fn(next_states)             # [B]
#         V_learned = batch_V(V_params, obs)            # [B] or [B,1]
#         alpha = jnp.clip(current_step / float(curriculum_steps), 0.0, 1.0)

#         # 4) curriculum mix (optional)
#         value = (1.0 - alpha) * rollout_vals + alpha * V_learned

#         # 5) ε‑greedy prior (unchanged)
#         B = next_states.current_node.shape[0]
#         keys = jax.random.split(key, B)
#         pi_logits = batch_epsilon_greedy_qvalue_prior(
#             env, V_apply, params.get("V"), obs_fn_batch, next_states, epsilon, keys)

#         return mctx.RecurrentFnOutput(
#             reward      = rewards,
#             discount    = (1.0 - terminals) * env.gamma,
#             prior_logits= pi_logits,
#             value       = value
#         ), next_states
    
#     return recurrent_fn

# --- Main training loop builder ---
def get_agent_loop(env, config, obs_fn_batch, V_apply, recurrent_fn, V_opt_update, get_V_params, epsilon_schedule_fn=None):
    @jax.jit
    def batch_loss(V_params, value_targets, obs, mask):
        per_sample = vmap(lambda vp, vt, o: (V_apply(vp, o) - vt) ** 2,
                          in_axes=(None, 0, 0))(V_params, value_targets, obs)
        mask = mask.astype(per_sample.dtype)
        normalizer = jnp.maximum(1.0, jnp.sum(mask))
        return jnp.sum(per_sample * mask) / normalizer

    loss_grad = jax.jit(grad(batch_loss, argnums=0))
    batch_step = vmap(env.step, in_axes=(0, 0))
    batch_V = vmap(V_apply, in_axes=(None, 0))
    batch_reset = vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0))

    def loop_fn(state_dict, _):
        obs = obs_fn_batch(state_dict['env_states'])
        V_params = state_dict["V_params"]
        V_target_params = state_dict["V_target_params"]
        active_mask = jnp.logical_not(state_dict['env_states'].done)
        
        # Get policy logits and value
        epsilon = epsilon_schedule_fn(state_dict['opt_t']) if epsilon_schedule_fn else 0.1
        # Generate Q-value based epsilon-greedy prior for root
        batch_size = state_dict['env_states'].current_node.shape[0]
        
        # Optimize random key splitting - split once and reuse
        state_dict['key'], subkey1, subkey2 = jax.random.split(state_dict['key'], 3)
        keys = jax.random.split(subkey1, batch_size)
        
        root_prior = batch_epsilon_greedy_qvalue_prior(
            env, V_apply, V_params, obs_fn_batch, state_dict['env_states'], epsilon, keys)
        
        V = batch_V(V_params, obs)
        
        # Use learned policy as root prior
        root = mctx.RootFnOutput(
            prior_logits=root_prior,
            value=V,
            embedding=state_dict['env_states']
        )

        # Define inv_mask before using it in debug print
        inv_mask = jnp.logical_not(state_dict['env_states'].neighbor_mask)

        # jax.debug.print(
        #     "[root] any_invalid={ai} num_invalid={ni} prior_has_mass_on_invalid={mi} sum_prior_on_valid={sv}",
        #     ai=jnp.any(inv_mask),
        #     ni=jnp.sum(inv_mask),
        #     mi=jnp.any(jnp.isfinite(root_prior) & inv_mask),
        #     sv=jnp.sum(jnp.where(~inv_mask, jnp.exp(root_prior), 0.0)),
        # )

        # Use subkey2 for tree search
        sk = subkey2
        
        # Combine parameters for recurrent function
        params = {"V": V_target_params, "step": state_dict['opt_t']}
        
        po = mctx.muzero_policy(
            params=params,
            rng_key=sk,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config['num_simulations'],
            invalid_actions=inv_mask,
            # max_num_considered_actions=env.max_deg,       # for gumbel_muzero_policy
            qtransform=partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=config['use_mixed_value'],
                value_scale=config['value_scale']
            )
        )
        # extract search targets
        search_val = po.search_tree.node_values[:, po.search_tree.ROOT_INDEX]
        value_targets = jax.lax.stop_gradient(search_val)
        
        # For done agents, set value target to 0 (they're done, no future reward)
        # IMPORTANT: Include done states in training with target 0 so value function learns V(terminal) = 0
        value_targets = jnp.where(active_mask, value_targets, 0.0)

        jax.debug.print(
            "[targets] mean={mt:.4f} std={st:.4f} min={mn:.4f} max={mx:.4f} V_pred_mean={mp:.4f}",
            mt=jnp.mean(value_targets), st=jnp.std(value_targets),
            mn=jnp.min(value_targets), mx=jnp.max(value_targets),
            mp=jnp.mean(V),
        )

        # Include done states in training with target 0 to teach value function that V(terminal) = 0
        # Use all states (not just active_mask) so done states are included
        loss = batch_loss(V_params, value_targets, obs, jnp.ones_like(active_mask))
        V_grads = loss_grad(V_params, value_targets, obs, jnp.ones_like(active_mask))
        
        # Debug: print loss components and mask info
        num_done = jnp.sum(state_dict['env_states'].done.astype(jnp.int32))
        num_active = jnp.sum(active_mask.astype(jnp.int32))
        jax.debug.print(
            "MCTS Loss | loss={loss:.6f} active={active}/{total} done={done} value_targets={targets} V_pred={vpred}",
            loss=loss,
            active=num_active,
            total=active_mask.shape[0],
            done=num_done,
            targets=value_targets,
            vpred=batch_V(V_params, obs)
        )

        # Update value function
        V_updates, state_dict['V_opt_state'] = V_opt_update(V_grads, state_dict['V_opt_state'], V_params)
        state_dict['V_params'] = get_V_params(V_params, V_updates)
        
        # Fixed target update logic
        state_dict['opt_t'] += 1
        should_update = (state_dict["opt_t"] % config['target_update_frequency']) == 0
        state_dict["V_target_params"] = jax.tree_util.tree_map(
            lambda new, old: jnp.where(should_update, new, old),
            state_dict['V_params'],
            state_dict["V_target_params"]
        )

        # take action & step
        actions = po.action
        # jax.debug.print("Current state: {currs}, actions taken: {actions}", currs=state_dict['env_states'].current_node, actions=actions)
        
        # Capture state before step to check if agents were already done
        was_already_done = state_dict['env_states'].done
        
        state_dict['env_states'], rewards, terminals, info = batch_step(state_dict['env_states'], actions)
        
        # jax.debug.print(
        #     "MCTS | current_node={curr} pickup_node={pickup} reward={rew} done={done}",
        #     curr=state_dict['env_states'].current_node,
        #     pickup=state_dict['env_states'].pickup_node,
        #     rew=rewards,
        #     done=terminals
        # )
        # elap sed = info['travel'] + info['wait']  # or info['elapsed'] if you have it
        # jax.debug.print(
        #     "[step] rew={r} term={t} travel={tr} wait={wa} elapsed={el} step_gamma={g} time_gamma={tg}",
        #     r=rewards, t=terminals, tr=info['travel'], wa=info['wait'], el=elapsed,
        #     g=env.gamma,
        #     tg=(env.gamma ** elapsed),
        # )

        # Check if agents reached pickup (should keep them still, not reset)
        reached_pickup = terminals & (state_dict['env_states'].current_node == state_dict['env_states'].pickup_node)
        
        # Check if agents were already done before this step (to avoid double-counting episodes)
        new_episode_termination = terminals & ~was_already_done

        # reset environments - only reset episodes that terminated but didn't reach pickup (e.g., timeout, invalid)
        # For agents that reached pickup, keep them still at pickup location (avoid truncation bias)
        state_dict["key"], subkey = jax.random.split(state_dict["key"])
        subkeys = jax.random.split(subkey, num=config['batch_size'])
        should_reset = terminals & ~reached_pickup  # Reset for timeout/invalid, but not for reaching pickup
        jax.debug.print(
            "[reset] term={term} reached_pickup={rp} should_reset={sr} resets={nres}",
            term=terminals, rp=reached_pickup, sr=should_reset,
            nres=jnp.sum(should_reset.astype(jnp.int32)),
        )
        state_dict['env_states'] = jax.tree_util.tree_map(
            lambda reset, current: jnp.where(
                jnp.reshape(should_reset, [should_reset.shape[0]] + [1] * (len(current.shape) - 1)),
                reset,  # Use reset state for terminated episodes that didn't reach pickup
                current  # Use current state for continuing episodes or agents that reached pickup
            ),
            batch_reset(subkeys, state_dict['last_start'], state_dict['env_states'].pickup_node)[0],
            state_dict["env_states"]
        )

        state_dict['episode_travel'] = state_dict['episode_travel'] + info['travel']
        state_dict['episode_wait'] = state_dict['episode_wait'] + info['wait']
        # Track total episode time (travel + wait) for completed episodes
        state_dict['episode_total_time'] = state_dict['episode_travel'] + state_dict['episode_wait']
        episode_total_travel = state_dict['episode_travel']
        episode_total_wait = state_dict['episode_wait']
        episode_total_time = state_dict['episode_total_time']

        completed_time_increment = jnp.sum(jnp.where(new_episode_termination, episode_total_time, 0.0))
        completed_travel_increment = jnp.sum(jnp.where(new_episode_termination, episode_total_travel, 0.0))
        completed_wait_increment = jnp.sum(jnp.where(new_episode_termination, episode_total_wait, 0.0))
        completed_episodes_increment = jnp.sum(new_episode_termination.astype(jnp.float32))

        term_mask = new_episode_termination
        term_count = jnp.sum(term_mask)
        term_sum = jnp.sum(jnp.where(term_mask, state_dict['episode_return'], 0.0))
        jax.debug.print(
            "[avg_return] term_count={tc} batch_mean={bm:.4f} true_mean_over_terms={tm:.4f}",
            tc=term_count,
            bm=jnp.mean(jnp.where(term_mask, state_dict['episode_return'], 0.0)),
            tm=term_sum / jnp.maximum(1.0, term_count),
        )

        # update statistics - only count new episode terminations (not already-done agents)
        # Episode return accumulates for all agents (adding zero doesn't change it for done agents)
        state_dict.update({
            'episode_return': state_dict['episode_return'] + rewards,
            'avg_return': jnp.where(
                jnp.any(new_episode_termination),  # Only update on new terminations
                (state_dict['avg_return'] * config['avg_return_smoothing'] + 
                 jnp.mean(jnp.where(new_episode_termination, state_dict['episode_return'], 0.0)) * (1.0 - config['avg_return_smoothing'])),
                state_dict['avg_return']
            ),
            'num_episodes': jnp.where(jnp.any(new_episode_termination), state_dict['num_episodes'] + jnp.sum(new_episode_termination), state_dict['num_episodes']),
            'completed_total_time': state_dict['completed_total_time'] + completed_time_increment,
            'completed_travel_time': state_dict['completed_travel_time'] + completed_travel_increment,
            'completed_wait_time': state_dict['completed_wait_time'] + completed_wait_increment,
            'completed_episodes': state_dict['completed_episodes'] + completed_episodes_increment,
        })
        # Reset episode statistics only for new terminations (when episode first completes)
        state_dict['episode_return'] = jnp.where(new_episode_termination, 0, state_dict['episode_return'])
        state_dict['episode_travel'] = jnp.where(new_episode_termination, 0, state_dict['episode_travel'])
        state_dict['episode_wait'] = jnp.where(new_episode_termination, 0, state_dict['episode_wait'])
        state_dict['episode_total_time'] = jnp.where(new_episode_termination, 0, state_dict['episode_total_time'])

        state_dict['avg_travel'] = jnp.where(
            state_dict['num_episodes'] > 0,
            episode_total_travel / state_dict['num_episodes'],
            0.0
        )
        state_dict['avg_wait'] = jnp.where(
            state_dict['num_episodes'] > 0,
            episode_total_wait / state_dict['num_episodes'],
            0.0
        )
        state_dict['avg_total_time'] = jnp.where(
            state_dict['num_episodes'] > 0,
            episode_total_time / state_dict['num_episodes'],
            0.0
        )

        # visitation counts
        # flatten current_node indices
        nodes = state_dict['env_states'].current_node  # [batch_size]
        counts = jnp.bincount(nodes, length=env.num_nodes)
        state_dict['visit_counts'] += counts
        state_dict['cumulative_visits'] += counts

        # loss
        state_dict['loss'] = loss

        state_dict['key'], _ = jax.random.split(state_dict['key'])

        return state_dict, (info, po)

    @jit
    def run_loop(state_dict):
        state_dict, (metrics, po) = jax.lax.scan(loop_fn, state_dict, None, length=config['eval_frequency'])

        # Log the search tree if needed
        # if state_dict['opt_t'] % 500 == 0:
        #     first_tree = jax.tree_util.tree_map(lambda x: x[0], po.search_tree)
        #     last_tree = jax.tree_util.tree_map(lambda x: x[-1], po.search_tree)
        #     graph = convert_tree_to_graph(first_tree)
        #     fname1 = f"first_tree_search_{state_dict['opt_t']}.png"
        #     graph1 = graph.draw(fname1, prog="dot")
        #     graph = convert_tree_to_graph(last_tree)
        #     fname2 = f"last_tree_search_{state_dict['opt_t']}.png"
        #     graph2 = graph.draw(fname2, prog="dot")
        #     logger.log({
        #         "first_tree_search": wandb.Image(fname1),
        #         "last_tree_search":  wandb.Image(fname2)
        #     }, step=state_dict['opt_t'])

        eps_ot = 0.1 * (1 - state_dict['opt_t'] / config['num_steps'])

        # matching
        keys = jax.random.split(state_dict['key'], 2*config['batch_size']+1)
        subkeys_starts = keys[:config['batch_size']]
        subkeys_pickups = keys[config['batch_size']:2*config['batch_size']]
        base_key = keys[2 * config['batch_size']]          # shape (2,)
        key, eps_key, subkey1, subkey2 = jax.random.split(base_key, 4)
        subkeys  = jax.random.split(subkey1, config['batch_size']**2)
        # fixed starts & pickups from config
        starts = jnp.stack([jax_random.choice(k, env.fixed_starts) for k in subkeys_starts])
        pickups = jnp.stack([jax_random.choice(k, env.fixed_pickups) for k in subkeys_pickups])

        # jax.debug.print("starts={starts} pickups={pickups}", starts=starts, pickups=pickups)

        batch_init = vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0))

        do_random = jax_random.uniform(eps_key, shape=()) < eps_ot

        C = estimate_returns_batch(
            subkeys, V_apply, obs_fn_batch, batch_init, 
            state_dict['V_target_params'], starts, pickups
        )
        # epsilon-greedy ot matching
        _, col_idx = optax.assignment.hungarian_algorithm(-C)
        pickups = jnp.where(do_random, pickups, pickups[col_idx])

        # initialize states
        subkeys = jax.random.split(subkey2, config['batch_size'])
        state_dict['env_states'], _ = batch_init(subkeys, starts, pickups)
        state_dict['last_start'] = starts
        state_dict['key'] = key

        total_visits = jnp.sum(state_dict['visit_counts'])
        freq = jnp.where(
            total_visits > 0,
            state_dict['visit_counts'] / total_visits,
            jnp.zeros_like(state_dict['visit_counts'], dtype=jnp.float32)
        )
        completed_eps = jnp.maximum(state_dict['completed_episodes'], 1.0)
        avg_total_time = state_dict['completed_total_time'] / completed_eps
        avg_travel_time = state_dict['completed_travel_time'] / completed_eps
        avg_wait_time = state_dict['completed_wait_time'] / completed_eps

        return state_dict, {
            'loss': state_dict['loss'], 
            'avg_return': state_dict['avg_return'], 
            'visit_freq': freq, 
            'avg_wait': avg_wait_time,
            'avg_travel': avg_travel_time,
            'avg_total_time': avg_total_time,
            'starts': starts,
            'pickups': pickups
        }

    return run_loop