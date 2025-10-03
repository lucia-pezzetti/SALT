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

from taxi_env import init_env, TaxiState

# map activation names
activation_dict = {"relu": jax.nn.relu, "silu": jax.nn.silu, "elu": jax.nn.elu}

# Value network
class VFunction(hk.Module):
    
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.hidden_dim = config['num_hidden_units']
        self.num_layers = config['num_hidden_layers'] 
        self.activation = activation_dict[config['activation']]
        self.node_embedding_dim = config.get('node_embedding_dim', 64)
        self.max_nodes = config.get('max_nodes', 100)
        self.cycle_length = config.get('cycle_length', 200)  # Default cycle length
        
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
        Input: obs [..., 9] = [current_pos(2), pickup_pos(2), relative_pos(2), distance(1), angle(1), time(1)]
        """
        # Extract features from the 9-dimensional observation
        current_pos = obs[..., 0:2]      # [..., 2] - current position (lat/lon)
        pickup_pos = obs[..., 2:4]       # [..., 2] - pickup position (lat/lon)
        relative_pos = obs[..., 4:6]     # [..., 2] - direction vector
        distance = obs[..., 6:7]         # [..., 1] - distance to pickup
        angle = obs[..., 7:8]            # [..., 1] - direction angle
        time = obs[..., 8:9]    
        
        # Normalize time feature
        time = time / self.cycle_length  # Normalize by typical cycle length
        
        # Normalize distance (assuming max distance is around 1.0 for normalized coordinates)
        distance = distance / 1.0  # Could be made configurable
        
        # Normalize angle to [-1, 1] range
        angle = angle / jnp.pi  # Convert from [-π, π] to [-1, 1]

        # Combine all features
        features = jnp.concatenate([
        #     current_emb, pickup_emb, time
        # ], axis=-1)
            current_pos,    # [..., 2] - current position
            pickup_pos,     # [..., 2] - pickup position
            relative_pos,   # [..., 2] - direction vector
            distance,       # [..., 1] - distance
            angle,          # [..., 1] - angle
            time            # [..., 1] - time
        ], axis=-1)  # Total: [..., 9]
        
        # Layer normalization
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
                
        # Final layer
        output = hk.Linear(
            1,
            w_init=hk.initializers.VarianceScaling(0.1, "fan_in", "truncated_normal"),
            b_init=hk.initializers.Constant(0.0)
        )(x).squeeze(-1)
        
        # Clip output
        return jnp.clip(output, -100.0, 100.0)


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
    uniform_prob = epsilon / jnp.maximum(num_valid, 1.0)
    greedy_prob = uniform_prob + (1.0 - epsilon)
    
    # Create logits
    logits = jnp.where(mask, jnp.log(uniform_prob + 1e-8), -jnp.inf)
    logits = logits.at[best_action].set(jnp.log(greedy_prob + 1e-8))
    
    return logits

def batch_epsilon_greedy_qvalue_prior(env, V_apply, V_params, obs_fn_batch, 
                                               states, epsilon: float, keys):
    """Efficient batch version"""
    return vmap(epsilon_greedy_qvalue_prior, in_axes=(None, None, None, None, 0, None, 0))(
        env, V_apply, V_params, obs_fn_batch, states, epsilon, keys)



# ot matching function using the estimated returns as costs
def estimate_returns_batch(keys, V_apply, obs_fn_batch, batch_init, 
                          V_params, starts, pickups):
    
    N = len(starts)
    
    # Create all combinations of starts and pickups
    start_grid, pickup_grid = jnp.meshgrid(starts, pickups, indexing='ij')
    start_flat = start_grid.flatten()
    pickup_flat = pickup_grid.flatten()
    
    # Initialize all combinations
    all_states, _ = batch_init(keys, start_flat, pickup_flat)
    
    # Get observations for all states
    obs = obs_fn_batch(all_states)
    
    # Get value estimates directly from the learned value function
    batch_V = vmap(V_apply, in_axes=(None, 0))
    returns = batch_V(V_params, obs.astype(float))
    
    # Reshape to [num_starts, num_pickups] matrix
    return returns.reshape(N, N)


# Init function
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
        V_net = hk.without_apply_rng(hk.transform(lambda obs: VFunction(config)(obs)))
        key, sk = jax.random.split(key)
        V_params = V_net.init(sk, dummy_obs)
        V_target_params = V_params
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

# Recurrent fn for tree-search
def get_recurrent_fn(
    env,
    V_apply,
    obs_fn_batch,
    epsilon_schedule_fn=None,
    curriculum_steps: int = 100_000,
):
    # pre‑compute next_hop[src, tgt] - the neighbor of i that lies on some shortest path to j.
    hop_dist = jnp.array(env.hop_distances)
    N = hop_dist.shape[0]
    next_hop_np = np.zeros((N, N), dtype=int)
    
    for src in range(N):
        # gather all 1-hop neighbors of `src`
        neighs = jnp.where(hop_dist[src] == 1)[0]
        for tgt in range(N):
            d = hop_dist[src, tgt]
            if d > 0:
                # find the first neighbor whose dist-to-target is d-1
                for nei in neighs:
                    if hop_dist[nei, tgt] == d - 1:
                        next_hop_np[src, tgt] = nei
                        break
            else:
                # same node ⇒ action can be arbitrary
                next_hop_np[src, tgt] = src
        
    next_hop = jax.device_put(jnp.array(next_hop_np, dtype=jnp.int32))

    # vectorized env.step and V
    batch_step = vmap(env.step, in_axes=(0, 0))
    batch_V    = vmap(V_apply, in_axes=(None, 0))

    @jax.jit
    def rollout_value_fn(states):
        """
        True discounted return of following the shortest-path policy until pickup.
        """
        # how many steps max?
        hop_dists     = env.hop_distances[states.current_node, states.pickup_node]
        max_steps = jnp.max(hop_dists)

        B = states.current_node.shape[0]
        init_R     = jnp.zeros(B, dtype=jnp.float32)
        init_disc  = jnp.ones(B, dtype=jnp.float32)
        init_done  = jnp.zeros(B, dtype=bool)

        # carry = (step, states, R, discount, done_mask)
        init_carry = (0, states, init_R, init_disc, init_done)

        def cond_fn(carry):
            step, _, _, _, done = carry
            # continue while we haven't hit max_steps and at least one trajectory still alive
            return jnp.logical_and(step < max_steps, jnp.any(~done))

        def body_fn(carry):
            step, states, R, discount, done = carry

            # look up shortest-path neighbor for each instance in the batch
            next_nodes = next_hop[ states.current_node, states.pickup_node ]

            # map neighbor node ids to action indices expected by env.step (slot in adj_list)
            def node_to_action(curr_node, target_node):
                nbrs = env.adj_list[curr_node]
                # find index where adj_list[curr_node, j] == target_node
                pos = jnp.where(nbrs == target_node, size=1)[0][0]
                return jnp.int32(pos)

            actions = jax.vmap(node_to_action)(states.current_node, next_nodes)

            next_s, rew, term, _ = batch_step(states, actions)

            # accumulate only for still‑alive ones
            R        = R + discount * rew
            discount = discount * env.gamma * (1.0 - term)
            done     = done | term

            return (step + 1, next_s, R, discount, done)

        _, _, final_R, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        return final_R

    def recurrent_fn(params, key, actions, states):
        V_params   = params["V"]
        step_count = params.get("step", 0)
        ε          = epsilon_schedule_fn(step_count) if epsilon_schedule_fn else 0.1
        alpha      = jnp.clip(step_count / float(curriculum_steps), 0.0, 1.0)

        # actual environment step
        next_states, rewards, terminals, _ = batch_step(states, actions)
        obs = obs_fn_batch(next_states)

        # rollout return under shortest-path policy
        rollout_vals = rollout_value_fn(next_states)  # shape [B]

        # your learned V
        V_learned = batch_V(V_params, obs)            # [B] or [B,1]
        V_learned = jnp.squeeze(V_learned)            # make sure [B]
        V_learned = jnp.where(terminals, 0.0, V_learned)

        # curriculum mix
        value = (1.0 - alpha) * rollout_vals + alpha * V_learned

        # ε‑greedy prior
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

# Main training loop builder
def get_agent_loop(env, config, obs_fn_batch, V_apply, recurrent_fn, V_opt_update, get_V_params, epsilon_schedule_fn=None):
    @jax.jit
    def batch_loss(V_params, value_targets, obs):
        return jnp.mean(vmap(lambda vp, vt, o: (V_apply(vp, o) - vt) ** 2, 
                     in_axes=(None, 0, 0))(V_params, value_targets, obs))
    loss_grad = jax.jit(grad(batch_loss))
    batch_step = vmap(env.step, in_axes=(0, 0))
    batch_V = vmap(V_apply, in_axes=(None, 0))
    batch_reset = vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0))

    def loop_fn(state_dict, _):
        obs = obs_fn_batch(state_dict['env_states'])
        V_params = state_dict["V_params"]
        V_target_params = state_dict["V_target_params"]
        
        # Get policy logits and value
        epsilon = epsilon_schedule_fn(state_dict['opt_t']) if epsilon_schedule_fn else 0.1
        # Generate Q-value based epsilon-greedy prior for root
        batch_size = state_dict['env_states'].current_node.shape[0]
        
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
        sk = subkey2
        
        # Combine parameters for recurrent function
        params = {"V": V_target_params, "step": state_dict['opt_t']}

        inv_mask = jnp.logical_not(state_dict['env_states'].neighbor_mask)
        
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

        loss = batch_loss(V_params, value_targets, obs)
        V_grads = loss_grad(V_params, value_targets, obs)

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
        state_dict['env_states'], rewards, terminals, info = batch_step(state_dict['env_states'], actions)
        # jax.debug.print("Next states: {next_states}, pickups: {pickups}, done: {done}", next_states=state_dict['env_states'].current_node, pickups=state_dict['env_states'].pickup_node, done=terminals)

        # reset environments that are done
        state_dict["key"], subkey = jax.random.split(state_dict["key"])
        subkeys = jax.random.split(subkey, num=config['batch_size'])
        state_dict['env_states'] = jax.tree_util.tree_map(
            lambda reset, current: jnp.where(
                jnp.reshape(terminals, [terminals.shape[0]] + [1] * (len(current.shape) - 1)),
                reset,
                current
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

        # update statistics
        state_dict.update({
            'episode_return': state_dict['episode_return'] + rewards,
            'avg_return': jnp.where(
                terminals,
                (state_dict['avg_return'] * config['avg_return_smoothing'] + 
                 state_dict['episode_return'] * (1.0 - config['avg_return_smoothing'])),
                state_dict['avg_return']
            ),
            'num_episodes': jnp.where(terminals, state_dict['num_episodes'] + 1, state_dict['num_episodes']),
        })
        state_dict['episode_return'] = jnp.where(terminals, 0, state_dict['episode_return'])
        state_dict['episode_travel'] = jnp.where(terminals, 0, state_dict['episode_travel'])
        state_dict['episode_wait'] = jnp.where(terminals, 0, state_dict['episode_wait'])
        state_dict['episode_total_time'] = jnp.where(terminals, 0, state_dict['episode_total_time'])

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

        freq = state_dict['visit_counts'] / state_dict['visit_counts'].sum()
        return state_dict, {
            'loss': state_dict['loss'], 
            'avg_return': state_dict['avg_return'], 
            'visit_freq': freq, 
            'avg_wait': jnp.mean(metrics['wait']), 
            'avg_travel': jnp.mean(metrics['travel']), 
            'avg_total_time': jnp.mean(state_dict['avg_total_time']),
            'starts': starts,
            'pickups': pickups
        }

    return run_loop