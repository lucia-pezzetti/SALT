import mctx
import haiku as hk
import jax
import jax.numpy as jnp
from jax import random as jax_random
from jax import jit, vmap, grad
from functools import partial

import numpy as np
import optax
import chex
import pygraphviz
from typing import Optional, Sequence
import wandb

# --- Import your Taxi environment ---
from taxi_env import init_env, TaxiState

def convert_tree_to_graph(
        tree: mctx.Tree,
        action_labels: Optional[Sequence[str]] = None,
        batch_index: int = 0
    ) -> pygraphviz.AGraph:
    """Converts a search tree into a Graphviz AGraph, materializing JAX arrays first."""

    if tree.node_values.ndim == 3:
        # take only the final step
        tree = tree.replace(
        node_values         = tree.node_values        [-1],
        node_visits         = tree.node_visits        [-1],
        children_index      = tree.children_index     [-1],
        children_rewards    = tree.children_rewards   [-1],
        children_discounts  = tree.children_discounts [-1],
        children_prior_logits = tree.children_prior_logits[-1],
        )

    chex.assert_rank(tree.node_values, 2)
    B, N = tree.node_values.shape
    _, _, A = tree.children_index.shape

    node_values   = np.array(jax.device_get(tree.node_values   [batch_index]))  # shape [num_nodes]
    node_visits   = np.array(jax.device_get(tree.node_visits   [batch_index]))
    children_idxs = np.array(jax.device_get(tree.children_index[batch_index]))  # shape [num_nodes, num_actions]
    orig_ids      = np.array(jax.device_get(tree.embeddings.current_node[batch_index]))
    children_rew  = np.array(jax.device_get(tree.children_rewards [batch_index]))  # [num_nodes, num_actions]
    children_dis  = np.array(jax.device_get(tree.children_discounts[batch_index]))
    prior_logits  = np.array(jax.device_get(tree.children_prior_logits[batch_index]))  # [num_nodes, num_actions]

    if action_labels is None:
        action_labels = list(range(A))
    elif len(action_labels) != A:
        raise ValueError(f"Wrong number of action_labels: got {len(action_labels)}, expected {A}")

    def node_to_str(i: int) -> str:
        return (
        f"{i}\n"
        f"Original ID: {orig_ids[i]}\n"
        f"Reward: {0.0:.2f}\n" 
        f"Discount: {1.0:.2f}\n"
        f"Value: {node_values[i]:.2f}\n"
        f"Visits: {node_visits[i]}"
        )

    def edge_to_str(parent, a):
        q_arr = np.array(jax.device_get(tree.qvalues(jax.numpy.array([parent]))[batch_index]))
        p_arr = np.array(jax.device_get(jax.nn.softmax(prior_logits[parent])))
        return f"{action_labels[a]}\nQ: {q_arr[a]:.2f}\np: {p_arr[a]:.2f}"

    graph = pygraphviz.AGraph(directed=True)

    # root
    graph.add_node(0, label=node_to_str(0), color="green")

    # expand out
    for parent in range(node_values.shape[0]):
        for a in range(A):
            child = int(children_idxs[parent, a])
            if child >= 0:
                graph.add_node(
                child,
                label=(
                    f"{child}\n"
                    f"Original ID: {orig_ids[child]}\n"
                    f"Reward: {children_rew[parent,a]:.2f}\n"
                    f"Discount: {children_dis[parent,a]:.2f}\n"
                    f"Value: {node_values[child]:.2f}\n"
                    f"Visits: {node_visits[child]}"
                ),
                color="red"
                )
                graph.add_edge(parent, child, label=edge_to_str(parent, a))

    return graph

# map activation names if you still use a value network
activation_dict = {"relu": jax.nn.relu, "silu": jax.nn.silu, "elu": jax.nn.elu}

# --- Value network only (keep it if you want to learn a value fn) ---
class V_function(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config['num_hidden_units']
        self.num_hidden_layers = config['num_hidden_layers']
        self.activation = activation_dict[config['activation']]
        self.init_bias = config.get('init_bias', 10.0)
    def __call__(self, obs):
        x = jnp.ravel(obs)
        for _ in range(self.num_hidden_layers):
            x = self.activation(hk.Linear(self.num_hidden_units,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(self.init_bias))(x))
        return hk.Linear(
            1,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(self.init_bias))(x)[0]
    
class GraphAwareVFunction(hk.Module):
    """Value function that understands graph structure and road types"""
    
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.hidden_dim = config['num_hidden_units']
        self.num_layers = config['num_hidden_layers'] 
        self.activation = activation_dict[config['activation']]
        self.node_embedding_dim = config.get('node_embedding_dim', 64)
        self.max_nodes = config.get('max_nodes', 100)
        
    def __call__(self, obs, adj_list=None, road_types=None):
        # Node embeddings
        current_pos = obs[..., 0].astype(jnp.int32)
        pickup_pos = obs[..., 1].astype(jnp.int32)

        # Learnable node embeddings
        node_embeddings = hk.Embed(self.max_nodes, self.node_embedding_dim)
        current_emb = node_embeddings(current_pos)
        pickup_emb = node_embeddings(pickup_pos)
        
        # global features: time
        time = obs[..., -1:]

        # Combine features
        features = jnp.concatenate([
            current_emb, pickup_emb,
            time
        ], axis=-1)  # [B, 2*node_embedding_dim + 1]

        # Process through MLP with residual connections
        x = features
        for i in range(self.num_layers):
            residual = x if x.shape[-1] == self.hidden_dim else None
            x = hk.Linear(self.hidden_dim)(x)
            x = self.activation(x)
            if residual is not None:
                x = x + residual  # Residual connection
                
        return hk.Linear(1)(x).squeeze(-1)  # Output value for each state
    

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
    next_values = vmap(V_apply, in_axes=(None, 0))(V_target_params, next_obs.astype(float))

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
    
    # Start with uniform probability for valid actions, -inf for invalid
    logits = jnp.where(mask, jnp.log(uniform_prob + 1e-8), -jnp.inf)
    # Add extra probability to best action
    logits = logits.at[best_action].set(jnp.log(greedy_prob + 1e-8))
    
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
        V_net = hk.without_apply_rng(hk.transform(lambda obs: GraphAwareVFunction(config)(obs.astype(float))))
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
def get_recurrent_fn(env, V_apply, obs_fn_batch, epsilon_schedule_fn=None):
    batch_step = vmap(env.step, in_axes=(0, 0))
    batch_V = vmap(V_apply, in_axes=(None, 0))
    
    def recurrent_fn(params, key, actions, states):
        V_params = params.get("V", None)

        current_step = params.get("step", 0)
        epsilon = epsilon_schedule_fn(current_step) if epsilon_schedule_fn else 0.1

        # step env
        next_states, rewards, terminals, _ = batch_step(states, actions)
        obs = obs_fn_batch(next_states)

        # compute value 
        V_learned = batch_V(V_params, obs)
        # V_learned = jnp.where(terminals, 0.0, V_learned)

        # Generate Q-value based epsilon-greedy prior
        batch_size = next_states.current_node.shape[0]
        keys = jax.random.split(key, batch_size)
        pi_logits = batch_epsilon_greedy_qvalue_prior(
            env, V_apply, V_params, obs_fn_batch, next_states, epsilon, keys)
        
        return mctx.RecurrentFnOutput(
            reward=rewards,
            discount=(1.0 - terminals) * env.gamma,
            prior_logits=pi_logits,
            value=V_learned
        ), next_states
    return recurrent_fn

# --- Main training loop builder ---
def get_agent_loop(env, config, obs_fn_batch, V_apply, recurrent_fn, V_opt_update, get_V_params, epsilon_schedule_fn=None):
    batch_loss = lambda V_params, value_targets, obs: \
        jnp.mean(vmap(lambda vp, vt, o: (V_apply(vp, o.astype(float)) - vt) ** 2, 
                     in_axes=(None, 0, 0))(V_params, value_targets, obs))
    loss_grad = grad(batch_loss)
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
        state_dict['key'], subkey = jax.random.split(state_dict['key'])
        keys = jax.random.split(subkey, batch_size)
        root_prior = batch_epsilon_greedy_qvalue_prior(
            env, V_apply, V_params, obs_fn_batch, state_dict['env_states'], epsilon, keys)
        
        V = batch_V(V_params, obs)
        
        # Use learned policy as root prior
        root = mctx.RootFnOutput(
            prior_logits=root_prior,
            value=V,
            embedding=state_dict['env_states']
        )

        state_dict['key'], sk = jax.random.split(state_dict['key'])
        
        # Combine parameters for recurrent function
        params = {"V": V_target_params, "step": state_dict['opt_t']}

        inv_mask = jnp.logical_not(state_dict['env_states'].neighbor_mask).astype(jnp.float32)
        
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

        loss = batch_loss(V_params, search_val, obs)
        V_grads = loss_grad(V_params, search_val, obs)

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
        state_dict['env_states'], rewards, terminals, info = batch_step(state_dict['env_states'], actions)

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

        return state_dict, (info, po)

    @jit
    def run_loop(state_dict):
        state_dict, (metrics, po) = jax.lax.scan(loop_fn, state_dict, None, length=config['eval_frequency'])

        # Log the search tree if needed
        # if state_dict['opt_t'] % 5000 == 0:
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
        return state_dict, {'loss': state_dict['loss'], 'avg_return': state_dict['avg_return'], 'visit_freq': freq, 'avg_wait': jnp.mean(metrics['wait']), 'avg_travel': jnp.mean(metrics['travel'])}

    return run_loop