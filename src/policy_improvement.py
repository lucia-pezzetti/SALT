import mctx
import haiku as hk
import jax
import jax.numpy as jnp
from jax import random as jax_random
from jax import jit, vmap, grad
from functools import partial

import optax

# --- Import your Taxi environment ---
from taxi_env import init_env, TaxiState

# map activation names if you still use a value network
activation_dict = {"relu": jax.nn.relu, "silu": jax.nn.silu, "elu": jax.nn.elu}

# --- Value network only (keep it if you want to learn a value fn) ---
class V_function(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config['num_hidden_units']
        self.num_hidden_layers = config['num_hidden_layers']
        self.activation = activation_dict[config['activation']]
    def __call__(self, obs):
        x = jnp.ravel(obs)
        for _ in range(self.num_hidden_layers):
            x = self.activation(hk.Linear(self.num_hidden_units)(x))
        return hk.Linear(1)(x)[0]
    

class pi_function(hk.Module):
    def __init__(self, config, num_actions, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config['num_hidden_units']  # Fixed: use dict access
        self.num_hidden_layers = config['num_hidden_layers']  # Fixed: use dict access
        self.activation_function = activation_dict[config['activation']]  # Fixed: use dict access
        self.num_actions = num_actions

    def __call__(self, obs, mask):
        x = jnp.ravel(obs)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        pi_logit = hk.Linear(self.num_actions)(x)
        # apply mask to logits
        masked_logits = jnp.where(mask, pi_logit, -jnp.inf)
        return masked_logits
    

# --- Deterministic shortest-path prior ---
def shortest_path_prior(env, state: TaxiState) -> jnp.ndarray:
    curr = state.current_node
    pickup = state.pickup_node
    neighbors = env.adj_list[curr]          # [max_deg]
    dists = env.distances[neighbors, pickup]  # [max_deg]
    # mask out invalid edges
    mask = state.neighbor_mask              # [max_deg]
    dists = jnp.where(mask, dists, jnp.inf)  # set invalid to inf
    best = jnp.argmin(dists)
    logits = jnp.full((env.max_deg,), -jnp.inf)
    return logits.at[best].set(0.0)     # after softmax, best has prob 1.0, others 0.0

batch_prior = vmap(shortest_path_prior, in_axes=(None, 0))

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
        dummy_obs = obs_fn_single(dummy_state).astype(float)
        dummy_mask = dummy_state.neighbor_mask.astype(float)

        # build value network
        V_net = hk.without_apply_rng(hk.transform(lambda obs: V_function(config)(obs.astype(float))))
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

        # build policy network
        pi_net = hk.without_apply_rng(hk.transform(lambda obs, mask: pi_function(config, env.max_deg)(obs.astype(float), mask)))
        key, subkey = jax.random.split(key)
        pi_params = pi_net.init(subkey, dummy_obs, dummy_mask)
        pi_func = pi_net.apply

        pi_opt = optax.adamw(
            config['pi_alpha'],
            eps=config['eps_adam'],
            b1=config['b1_adam'],
            b2=config['b2_adam'],
            weight_decay=config['wd_adam']
        )
        pi_opt_init    = pi_opt.init
        pi_opt_update  = pi_opt.update
        get_pi_params  = optax.apply_updates
        pi_opt_state = pi_opt_init(pi_params)

        return key, env_states, V_func, pi_func, V_opt_state, pi_opt_state, V_opt_update, pi_opt_update, get_V_params, V_target_params, get_pi_params, pi_params

    return init_fn

# --- Recurrent fn for tree-search ---
def get_recurrent_fn(env, V_apply, pi_func, obs_fn_batch):
    batch_step = vmap(env.step, in_axes=(0, 0))
    batch_pi_func = vmap(pi_func, in_axes=(None, 0, 0))
    batch_V = vmap(V_apply, in_axes=(None, 0))
    
    def recurrent_fn(params, key, actions, states):
        V_params = params["V"]
        pi_params = params["pi"]
        # step env
        next_states, rewards, terminals, _ = batch_step(states, actions)
        obs = obs_fn_batch(next_states)
        # compute value & prior
        V = batch_V(V_params, obs)
        pi_logits = batch_pi_func(pi_params, obs.astype(float), next_states.neighbor_mask)
        return mctx.RecurrentFnOutput(
            reward=rewards,
            discount=(1.0 - terminals) * env.gamma,
            prior_logits=pi_logits,
            value=V
        ), next_states
    return recurrent_fn

# Actor-Critic loss function
def get_AC_loss(pi_func, V_func):
    def AC_loss(pi_params, V_params, pi_target, V_target, obs, mask):
        pi_logits = pi_func(pi_params, obs.astype(float), mask)
        V = V_func(V_params, obs.astype(float))
        pi_log_probs = jax.nn.log_softmax(pi_logits)

        # KL divergence loss for policy
        pi_loss = -jnp.sum(pi_target * pi_log_probs)
        # MSE loss for value
        V_loss = (V_target - V) ** 2

        return pi_loss + V_loss
    return AC_loss

# --- Main training loop builder ---
def get_agent_loop(env, config, obs_fn_batch, V_apply, pi_func, recurrent_fn, V_opt_update, pi_opt_update, get_V_params, get_pi_params):
    batch_loss = lambda pi_params, V_params, policy_targets, value_targets, obs, mask: \
        jnp.mean(vmap(get_AC_loss(pi_func, V_apply), in_axes=(None, None, 0, 0, 0, 0))(pi_params, V_params, policy_targets, value_targets, obs, mask))
    loss_grad = grad(batch_loss, argnums=(0, 1))
    batch_step = vmap(env.step, in_axes=(0, 0))
    batch_V = vmap(V_apply, in_axes=(None, 0))
    batch_pi_func = vmap(pi_func, in_axes=(None, 0, 0))
    batch_reset = vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0))

    def loop_fn(state_dict, _):
        obs = obs_fn_batch(state_dict['env_states'])
        
        # Get current policy parameters
        pi_params = state_dict["pi_params"]
        V_params = state_dict["V_params"]

        # target params for value estimation
        V_target_params = state_dict["V_target_params"]
        
        # Get policy logits and value
        sp_logits = batch_prior(env, state_dict['env_states'])
        pi_logits = batch_pi_func(pi_params, obs.astype(float), state_dict['env_states'].neighbor_mask)

        # alpha = jnp.clip(training_progress, 0.0, 1.0)
        # mixed_logits = (1.0 - alpha) * sp_logits + alpha * pi_logits

        V = batch_V(V_target_params, obs)
        
        # Use learned policy as root prior instead of shortest path
        root = mctx.RootFnOutput(
            prior_logits=pi_logits,  # Use learned policy as prior
            value=V,
            embedding=state_dict['env_states']
        )

        state_dict['key'], sk = jax.random.split(state_dict['key'])
        
        # Combine parameters for recurrent function
        params = {"V": V_target_params, "pi": pi_params}

        inv_mask = jnp.logical_not(state_dict['env_states'].neighbor_mask).astype(jnp.float32)
        
        po = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=sk,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config['num_simulations'],
            invalid_actions=inv_mask,
            max_num_considered_actions=env.max_deg,
            qtransform=partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=config['use_mixed_value'],
                value_scale=config['value_scale']
            )
        )
        # extract search targets
        search_policy = po.action_weights
        search_val = po.search_tree.node_values[:, po.search_tree.ROOT_INDEX]

        # compute gradients
        loss = batch_loss(pi_params, V_params, search_policy, search_val, obs, state_dict['env_states'].neighbor_mask)
        pi_grads, V_grads = loss_grad(pi_params, V_params, search_policy, search_val, obs, state_dict['env_states'].neighbor_mask)

        # update policy
        pi_updates, state_dict['pi_opt_state'] = pi_opt_update(pi_grads, state_dict['pi_opt_state'], pi_params)

        state_dict['pi_params'] = get_pi_params(pi_params, pi_updates)

        # update value function
        V_updates, state_dict['V_opt_state'] = V_opt_update(V_grads, state_dict['V_opt_state'], V_params)

        # apply updates
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
        state_dict['env_states'], rewards, terminals, info = batch_step(state_dict['env_states'], actions)

        # print for debugging
        # jax.debug.print("Current state: {}, actions: {}, rewards: {}, reached pickup: {}, wait: {}", 
        #                state_dict['env_states'].current_node[0], actions[0], rewards[0], terminals[0], info['wait'][0])

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
        episode_total_travel = state_dict['episode_travel']
        episode_total_wait = state_dict['episode_wait']

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

        # visitation counts
        # flatten current_node indices
        nodes = state_dict['env_states'].current_node  # [batch_size]
        counts = jnp.bincount(nodes, length=env.num_nodes)
        state_dict['visit_counts'] += counts

        # loss
        state_dict['loss'] = loss

        state_dict['key'], _ = jax.random.split(state_dict['key'])

        return state_dict, info

    @jit
    def run_loop(state_dict):
        state_dict, metrics = jax.lax.scan(loop_fn, state_dict, None, length=config['eval_frequency'])

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

        # jax.debug.print("Matching results: starts: {}, pickups: {}", starts, pickups)

        # initialize states
        subkeys = jax.random.split(subkey2, config['batch_size'])
        state_dict['env_states'], _ = batch_init(subkeys, starts, pickups)
        state_dict['last_start'] = starts
        state_dict['key'] = key

        freq = state_dict['visit_counts'] / state_dict['visit_counts'].sum()
        return state_dict, {'loss': state_dict['loss'], 'avg_return': state_dict['avg_return'], 'visit_freq': freq, 'avg_wait': jnp.mean(metrics['wait']), 'avg_travel': jnp.mean(metrics['travel'])}

    return run_loop