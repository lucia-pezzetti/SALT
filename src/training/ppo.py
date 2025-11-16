import haiku as hk
import jax
import jax.numpy as jnp
from jax import random as jax_random
from jax import jit, vmap, grad, profiler
from functools import partial
import numpy as onp
import optax

# --- Import your Taxi environment ---
from taxi_env import init_env, TaxiState
from utils import offline_shortest_path_action, offline_shortest_path_action_batch

# map activation names
activation_dict = {"relu": jax.nn.relu, "silu": jax.nn.silu, "elu": jax.nn.elu}

# Import GNN networks (optional)
try:
    from training.gnn_networks import GNPPolicyNetwork, GNNValueNetwork
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

# PPO policy network
class PPOPolicyNetwork(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.hidden_dim = config['num_hidden_units']
        self.num_layers = config['num_hidden_layers'] 
        self.activation = activation_dict[config['activation']]
        self.max_deg = config.get('max_deg', 8)
        self.cycle_length = config.get('cycle_length', 200)
        
    def __call__(self, obs):
        """
        Policy network for lat/lon-based observations with optional neighbor info.
        Input: obs [..., D] where D = 5 (base) or 5 + 2*max_deg (with neighbor info)
          Base: [current_pos(2), pickup_pos(2), time(1)]
          With neighbors: [current_pos(2), pickup_pos(2), time(1), neighbor_travel_times(max_deg), neighbor_distances(max_deg)]
        """
        obs_dim = obs.shape[-1]
        has_neighbor_info = obs_dim > 5
        
        # Extract base features
        current_pos = obs[..., 0:2]      # [..., 2] - current position (lat/lon)
        pickup_pos = obs[..., 2:4]       # [..., 2] - pickup position (lat/lon)
        time = obs[..., 4:5]             # [..., 1] - time
        
        # Normalize time feature
        time = time / self.cycle_length

        # Combine base features
        features_list = [current_pos, pickup_pos, time]
        
        # Add neighbor information if present
        if has_neighbor_info:
            neighbor_travel_times = obs[..., 5:5+self.max_deg]  # [..., max_deg]
            neighbor_distances = obs[..., 5+self.max_deg:5+2*self.max_deg]  # [..., max_deg]
            features_list.extend([neighbor_travel_times, neighbor_distances])
        
        features = jnp.concatenate(features_list, axis=-1)
        
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
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            if residual is not None:
                x = x + residual  # Residual connection
                
        # Final layer for action logits
        return hk.Linear(
            self.max_deg,
            w_init=hk.initializers.VarianceScaling(0.1, "fan_in", "truncated_normal"),
            b_init=hk.initializers.Constant(0.0)
        )(x)

# PPO value network
class PPOValueNetwork(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.hidden_dim = config['num_hidden_units']
        self.num_layers = config['num_hidden_layers'] 
        self.activation = activation_dict[config['activation']]
        self.max_deg = config.get('max_deg', 8)
        self.cycle_length = config.get('cycle_length', 200)
        self.init_bias = config.get('V_init_bias', 0.0)  # Initialize with config-specified bias
        
    def __call__(self, obs):
        """
        Value network for lat/lon-based observations with optional neighbor info.
        Input: obs [..., D] where D = 5 (base) or 5 + 2*max_deg (with neighbor info)
          Base: [current_pos(2), pickup_pos(2), time(1)]
          With neighbors: [current_pos(2), pickup_pos(2), time(1), neighbor_travel_times(max_deg), neighbor_distances(max_deg)]
        """
        obs_dim = obs.shape[-1]
        has_neighbor_info = obs_dim > 5
        
        # Extract base features
        current_pos = obs[..., 0:2]      # [..., 2] - current position (lat/lon)
        pickup_pos = obs[..., 2:4]       # [..., 2] - pickup position (lat/lon)
        time = obs[..., 4:5]             # [..., 1] - time
        
        # Normalize time feature
        time = time / self.cycle_length

        # Combine base features
        features_list = [current_pos, pickup_pos, time]
        
        # Add neighbor information if present
        if has_neighbor_info:
            neighbor_travel_times = obs[..., 5:5+self.max_deg]  # [..., max_deg]
            neighbor_distances = obs[..., 5+self.max_deg:5+2*self.max_deg]  # [..., max_deg]
            features_list.extend([neighbor_travel_times, neighbor_distances])
        
        features = jnp.concatenate(features_list, axis=-1)
        
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
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            if residual is not None:
                x = x + residual  # Residual connection
                
        # Final layer for value
        return hk.Linear(
            1,
            w_init=hk.initializers.VarianceScaling(0.1, "fan_in", "truncated_normal"),
            b_init=hk.initializers.Constant(0.0)
        )(x).squeeze(-1)

# PPO Experience Buffer
class PPOExperienceBuffer:
    def __init__(self, buffer_size, obs_dim, max_deg):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.max_deg = max_deg
        self.reset()
    
    def reset(self):
        self.observations = jnp.zeros((self.buffer_size, self.obs_dim))
        self.actions = jnp.zeros(self.buffer_size, dtype=jnp.int32)
        self.rewards = jnp.zeros(self.buffer_size)
        self.values = jnp.zeros(self.buffer_size)
        self.log_probs = jnp.zeros(self.buffer_size)
        self.advantages = jnp.zeros(self.buffer_size)
        self.returns = jnp.zeros(self.buffer_size)
        self.masks = jnp.zeros((self.buffer_size, self.max_deg))
        self.dones = jnp.zeros(self.buffer_size, dtype=bool)
        self.weights = jnp.zeros(self.buffer_size)
        self.ptr = 0
        self.size = 0
    
    def add(self, obs, action, reward, value, log_prob, mask, done):
        """Add a single experience to the buffer"""
        idx = self.ptr
        self.observations = self.observations.at[idx].set(obs)
        self.actions = self.actions.at[idx].set(action)
        self.rewards = self.rewards.at[idx].set(reward)
        self.values = self.values.at[idx].set(value)
        self.log_probs = self.log_probs.at[idx].set(log_prob)
        self.masks = self.masks.at[idx].set(mask)
        self.dones = self.dones.at[idx].set(done)
        self.weights = self.weights.at[idx].set(1.0 - float(done))
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def add_batch(self, obs, actions, rewards, values, log_probs, masks, dones):
        """Add a batch of experiences to the buffer"""
        batch_size = obs.shape[0]
        for i in range(batch_size):
            self.add(obs[i], actions[i], rewards[i], values[i], log_probs[i], masks[i], dones[i])
    
    def get_batch(self, batch_size, key):
        """Get a random batch from the buffer"""
        if self.size < batch_size:
            batch_size = self.size
        
        indices = jax.random.choice(
            key, 
            self.size, 
            shape=(batch_size,), 
            replace=False
        )
        
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices],
            'advantages': self.advantages[indices],
            'returns': self.returns[indices],
            'masks': self.masks[indices],
            'dones': self.dones[indices],
            'weights': self.weights[indices]
        }

# PPO Loss Functions
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE) - optimized vectorized version.
    
    This is a vectorized GAE computation that compiles much faster than loop-based versions.
    It computes advantages for a trajectory, respecting episode boundaries via the dones flag.
    """
    # Pad with zero for next values
    next_values = jnp.concatenate([values[1:], jnp.array([0.0])])
    
    # Compute TD errors
    deltas = rewards + gamma * next_values * (1 - dones) - values
    
    # Compute GAE
    def gae_step(carry, delta_done):
        delta, done = delta_done
        gae = delta + gamma * lam * (1 - done) * carry
        return gae, gae
    
    # Reverse the inputs
    deltas_reversed = jnp.flip(deltas)
    dones_reversed = jnp.flip(dones)
    
    _, advantages_reversed = jax.lax.scan(gae_step, 0.0, (deltas_reversed, dones_reversed))
    
    # Reverse back to original order
    advantages = jnp.flip(advantages_reversed)
    
    return advantages

def policy_loss_fn(policy_apply, policy_params, batch, old_log_probs, clip_ratio=0.2, entropy_coef=0.01):
    """Compute policy loss (actor loss) with entropy bonus"""
    logits = policy_apply(policy_params, batch['observations'])
    
    masks = batch['masks']
    
    masked_logits = jnp.where(masks, logits, -1e8)
    log_probs = jax.nn.log_softmax(masked_logits)
    selected_log_probs = jnp.sum(log_probs * jax.nn.one_hot(batch['actions'], masked_logits.shape[-1]), axis=-1)

    # Use raw advantages
    adv = batch['advantages']
    weights = batch.get('weights', jnp.ones_like(adv))
    weight_sum = jnp.sum(weights) + 1e-8
    
    # PPO policy loss with clipping
    log_ratio = selected_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    clipped_ratio = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)
    surrogate = jnp.minimum(ratio * adv, clipped_ratio * adv)
    policy_loss = -jnp.sum(surrogate * weights) / weight_sum
    
    # Entropy loss for exploration
    probs = jax.nn.softmax(masked_logits)
    entropy = -jnp.sum(jnp.sum(log_probs * probs, axis=-1) * weights) / weight_sum
    entropy_loss = -entropy_coef * entropy
    
    total_policy_loss = policy_loss + entropy_loss
    
    return total_policy_loss, {
        'policy_loss': policy_loss,
        'entropy_loss': entropy_loss,
        'entropy': entropy
    }

def value_loss_fn(value_apply, value_params, batch, value_coef=0.5, value_clip_ratio=0.2):
    """Compute value loss (critic loss) with clipping for stability"""
    values = value_apply(value_params, batch['observations'])
    returns = batch['returns']
    weights = batch.get('weights', jnp.ones_like(returns))
    weight_sum = jnp.sum(weights) + 1e-8
    
    value_loss = jnp.sum(((values - returns) ** 2) * weights) / weight_sum
    
    return value_coef * value_loss, {
        'value_loss': value_loss
    }

def ppo_loss_fn(policy_apply, value_apply, policy_params, value_params, batch, old_log_probs, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
    """Compute combined PPO loss"""
    policy_loss, policy_info = policy_loss_fn(policy_apply, policy_params, batch, old_log_probs, clip_ratio, entropy_coef)
    value_loss, value_info = value_loss_fn(value_apply, value_params, batch, value_coef)
    
    total_loss = policy_loss + value_loss
    
    return total_loss, {
        **policy_info,
        **value_info,
        'total_loss': total_loss
    }

# Behavioral Cloning Pretraining Function
def pretrain_policy_on_shortest_path(
    env, 
    policy_params, 
    policy_apply, 
    policy_opt, 
    policy_opt_state,
    obs_fn_batch,
    config,
    key,
    num_pretrain_steps=1000,
    pretrain_batch_size=64,
    pretrain_lr=None,
    eval_starts=None,
    eval_pickups=None,
    log_fn=None,
    value_params=None,
    value_apply=None,
    value_opt=None,
    value_opt_state=None
):
    """
    Behavioral Cloning pretraining: train policy to imitate expert shortest path actions
    
    Args:
        env: TaxiEnv instance
        policy_params: Current policy parameters
        policy_apply: Policy network apply function
        policy_opt: Policy optimizer
        policy_opt_state: Policy optimizer state
        obs_fn_batch: Batch observation function
        config: Configuration dictionary
        key: JAX random key
        num_pretrain_steps: Number of pretraining steps
        pretrain_batch_size: Batch size for pretraining
        pretrain_lr: Learning rate for pretraining (if None, uses config policy_lr)
        eval_starts: Optional array of start nodes for evaluation
        eval_pickups: Optional array of pickup nodes for evaluation
        log_fn: Optional logging function(log_dict, step) for metrics
        value_params: Optional value network parameters for value regression
        value_apply: Optional value network apply function
        value_opt: Optional value optimizer
        value_opt_state: Optional value optimizer state
    
    Returns:
        Updated policy_params, policy_opt_state, final_key, (optional value_params, value_opt_state)
    """
    from utils import smart_greedy_next_hop_batch
    
    # Use separate pretraining optimizer if specified
    if pretrain_lr is not None and pretrain_lr != config.get('policy_lr', 1e-4):
        pretrain_policy_opt = optax.chain(
            optax.clip_by_global_norm(config.get('max_grad_norm', 0.5)),
            optax.adamw(
                pretrain_lr,
                eps=config.get('eps_adam', 1e-5),
                b1=config.get('b1_adam', 0.9),
                b2=config.get('b2_adam', 0.999),
                weight_decay=config.get('wd_adam', 1e-4)
            )
        )
        pretrain_policy_opt_state = pretrain_policy_opt.init(policy_params)
        if value_params is not None:
            pretrain_value_opt = optax.chain(
                optax.clip_by_global_norm(config.get('max_grad_norm', 0.5)),
                optax.adamw(
                    pretrain_lr,
                    eps=config.get('eps_adam', 1e-5),
                    b1=config.get('b1_adam', 0.9),
                    b2=config.get('b2_adam', 0.999),
                    weight_decay=config.get('wd_adam', 1e-4)
                )
            )
            pretrain_value_opt_state = pretrain_value_opt.init(value_params)
        else:
            pretrain_value_opt = None
            pretrain_value_opt_state = None
    else:
        pretrain_policy_opt = policy_opt
        pretrain_policy_opt_state = policy_opt_state
        pretrain_value_opt = value_opt
        pretrain_value_opt_state = value_opt_state
    
    batch_init = jax.jit(vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0)))
    batch_step = jax.jit(vmap(env.step, in_axes=(0, 0)))
    
    gamma = config.get('gamma', 0.99)
    entropy_coef = config.get('bc_entropy_coef', 0.01)
    value_regression_enabled = value_params is not None and value_apply is not None
    value_coef = config.get('bc_value_coef', 0.5) if value_regression_enabled else 0.0
    
    # Evaluation function
    def evaluate_policy_performance(policy_params, starts, pickups):
        """Evaluate policy on fixed start-pickup pairs and compute BC accuracy"""
        if starts is None or pickups is None or len(starts) == 0:
            return None
        
        eval_keys = jax.random.split(jax.random.PRNGKey(42), len(starts))
        eval_states, _ = batch_init(eval_keys, starts, pickups)
        
        total_rewards = jnp.zeros(len(starts))
        total_steps = jnp.zeros(len(starts))
        completed = jnp.zeros(len(starts), dtype=bool)
        correct_actions = 0
        total_actions = 0
        
        # Run episodes
        for step in range(env.max_steps):
            active_mask = ~completed
            if bool(jnp.any(active_mask)):
                obs = obs_fn_batch(eval_states)
                logits = policy_apply(policy_params, obs)
                neighbor_masks = eval_states.neighbor_mask
                masked_logits = jnp.where(neighbor_masks, logits, -1e8)
                actions = jnp.argmax(masked_logits, axis=-1)
                
                # Compute expert actions
                expert_actions = smart_greedy_next_hop_batch(
                    eval_states.current_node,
                    eval_states.pickup_node,
                    env.adj_list,
                    env.travel_times,
                    env.distances,
                    neighbor_masks
                )
                
                # Count correct actions
                correct_actions += jnp.sum(jnp.where(active_mask, actions == expert_actions, 0))
                total_actions += jnp.sum(active_mask)
                
                next_states, rewards, terminals, info = batch_step(eval_states, actions)
                total_rewards = jnp.where(active_mask, total_rewards + rewards, total_rewards)
                total_steps = jnp.where(active_mask, total_steps + 1, total_steps)
                completed = completed | terminals
                eval_states = next_states
                
                if bool(jnp.all(completed)):
                    break
            else:
                break
        
        bc_accuracy = float(correct_actions / total_actions) if total_actions > 0 else 0.0
        
        return {
            'avg_reward': float(jnp.mean(total_rewards)),
            'avg_steps': float(jnp.mean(total_steps)),
            'completion_rate': float(jnp.mean(completed.astype(float))),
            'bc_accuracy': bc_accuracy
        }
    
    def compute_param_update_magnitude(params_before, params_after):
        """Compute L2 norm of parameter updates"""
        updates = jax.tree_util.tree_map(lambda a, b: a - b, params_after, params_before)
        param_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), updates)
        # Sum of squared norms
        total_magnitude = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(updates)))
        return float(total_magnitude)
    
    def pretrain_step(policy_params, policy_opt_state, key, value_params=None, value_opt_state=None):
        """Single BC pretraining step"""
        # Generate random starts and pickups
        key, key1, key2 = jax.random.split(key, 3)
        subkeys1 = jax.random.split(key1, pretrain_batch_size)
        subkeys2 = jax.random.split(key2, pretrain_batch_size)
        
        starts = jnp.stack([jax.random.choice(k, env.fixed_starts) for k in subkeys1])
        pickups = jnp.stack([jax.random.choice(k, env.fixed_pickups) for k in subkeys2])
        
        # Initialize environment states
        key, key_init = jax.random.split(key)
        init_keys = jax.random.split(key_init, pretrain_batch_size)
        env_states, _ = batch_init(init_keys, starts, pickups)
        
        # Collect expert trajectories: (state, mask, expert-action, return)
        max_rollout_steps = config.get('pretrain_rollout_steps', 20)
        
        # Store trajectories
        trajectories = [[] for _ in range(pretrain_batch_size)]
        
        current_states = env_states
        for step in range(max_rollout_steps):
            active_mask = ~current_states.done
            if not bool(jnp.any(active_mask)):
                break
            
            # Compute expert actions
            neighbor_masks = current_states.neighbor_mask
            expert_actions = smart_greedy_next_hop_batch(
                current_states.current_node,
                current_states.pickup_node,
                env.adj_list,
                env.travel_times,
                env.distances,
                neighbor_masks
            )
            
            # Take step using expert actions
            next_states, rewards, terminals, _ = batch_step(current_states, expert_actions)
            
            # Store data
            for i in range(pretrain_batch_size):
                if not bool(current_states.done[i]) and not bool(terminals[i]):
                    trajectories[i].append({
                        'state': jax.tree_util.tree_map(lambda x: x[i], current_states),
                        'mask': neighbor_masks[i],
                        'action': expert_actions[i],
                        'reward': rewards[i]
                    })
            
            # Update
            current_states = next_states
            
            # Stop if all complete
            if bool(jnp.all(terminals | current_states.done)):
                break
        
        # Flatten trajectories and compute returns
        all_states = []
        all_masks = []
        all_expert_actions = []
        all_returns = []
        
        for traj in trajectories:
            if len(traj) == 0:
                continue
            # Compute returns
            returns = jnp.zeros(len(traj))
            current_return = 0.0
            for i in range(len(traj) - 1, -1, -1):
                current_return = traj[i]['reward'] + gamma * current_return
                returns = returns.at[i].set(current_return)
            
            # Store all data from this trajectory
            for i, step_data in enumerate(traj):
                all_states.append(step_data['state'])
                all_masks.append(step_data['mask'])
                all_expert_actions.append(step_data['action'])
                all_returns.append(returns[i])
        
        if len(all_states) == 0:
            # No data collected, return unchanged
            return (policy_params, policy_opt_state, key, 
                    {'policy_loss': 0.0, 'bc_accuracy': 0.0, 'value_loss': 0.0, 
                     'ce_loss': 0.0, 'entropy_loss': 0.0, 'entropy': 0.0},
                    value_params, value_opt_state)
        
        # Concatenate collected data
        collected_states = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *all_states)
        collected_masks = jnp.stack(all_masks)
        collected_expert_actions = jnp.array(all_expert_actions)
        collected_returns = jnp.array(all_returns)
        
        # Limit to batch_size to prevent memory issues
        num_collected = collected_states.current_node.shape[0]
        if num_collected > pretrain_batch_size:
            key, sample_key = jax.random.split(key)
            indices = jax.random.choice(
                sample_key, 
                num_collected, 
                shape=(pretrain_batch_size,),
                replace=False
            )
            collected_states = jax.tree_util.tree_map(lambda x: x[indices], collected_states)
            collected_masks = collected_masks[indices]
            collected_expert_actions = collected_expert_actions[indices]
            collected_returns = collected_returns[indices]
        
        # Get observations for collected states
        obs = obs_fn_batch(collected_states)
        
        # Define BC loss function
        def bc_loss_fn(p):
            policy_logits = policy_apply(p, obs)  # [batch_size, max_deg]
            masked_logits = jnp.where(collected_masks, policy_logits, -1e-8)
            log_probs = jax.nn.log_softmax(masked_logits)  # [batch_size, max_deg]
            expert_log_probs = jnp.sum(
                log_probs * jax.nn.one_hot(collected_expert_actions, env.max_deg),
                axis=-1
            )
            ce_loss = -jnp.mean(expert_log_probs)
            
            probs = jax.nn.softmax(masked_logits)
            entropy = -jnp.mean(jnp.sum(log_probs * probs, axis=-1))
            entropy_loss = -entropy_coef * entropy
            
            total_loss = ce_loss + entropy_loss
            return total_loss, {'ce_loss': ce_loss, 'entropy_loss': entropy_loss, 'entropy': entropy}
        
        # Compute policy gradients and update policy
        (policy_loss, policy_info), policy_grads = jax.value_and_grad(
            bc_loss_fn, has_aux=True
        )(policy_params)
        updates, new_policy_opt_state = pretrain_policy_opt.update(policy_grads, policy_opt_state, policy_params)
        new_policy_params = optax.apply_updates(policy_params, updates)
        
        # Optionally update value network
        value_loss = 0.0
        new_value_params = value_params
        new_value_opt_state = value_opt_state
        if value_regression_enabled:
            def value_loss_fn(v):
                values = value_apply(v, obs)  # [batch_size]
                return jnp.mean((values - collected_returns) ** 2)
            
            value_loss, value_grads = jax.value_and_grad(value_loss_fn)(value_params)
            value_updates, new_value_opt_state = pretrain_value_opt.update(value_grads, value_opt_state, value_params)
            new_value_params = optax.apply_updates(value_params, value_updates)
        
        # Compute BC accuracy
        policy_logits = policy_apply(new_policy_params, obs)
        masked_logits = jnp.where(collected_masks, policy_logits, -1e8)
        predicted_actions = jnp.argmax(masked_logits, axis=-1)
        bc_accuracy = jnp.mean((predicted_actions == collected_expert_actions).astype(float))
        
        return (new_policy_params, new_policy_opt_state, key, 
                {'policy_loss': policy_loss, 'bc_accuracy': bc_accuracy, 
                 'value_loss': value_loss, **policy_info},
                new_value_params, new_value_opt_state)
    
    # Run pretraining steps
    current_policy_params = policy_params
    current_policy_opt_state = pretrain_policy_opt_state
    current_value_params = value_params
    current_value_opt_state = pretrain_value_opt_state
    current_key = key
    
    # Evaluation frequency
    eval_frequency = config.get('pretrain_eval_frequency', 100)
    
    print(f"Starting BC pretraining: {num_pretrain_steps} steps with batch size {pretrain_batch_size}")
    if value_regression_enabled:
        print(f"Value regression enabled with coefficient {value_coef}")
    if eval_starts is not None and eval_pickups is not None:
        print(f"Will evaluate policy every {eval_frequency} steps on {len(eval_starts)} evaluation pairs")
    
    for step in range(num_pretrain_steps):
        params_before = current_policy_params
        result = pretrain_step(
            current_policy_params, current_policy_opt_state, current_key,
            current_value_params, current_value_opt_state
        )
        current_policy_params, current_policy_opt_state, current_key, step_metrics, current_value_params, current_value_opt_state = result
        
        # Compute parameter update magnitude
        param_update_mag = compute_param_update_magnitude(params_before, current_policy_params)
        
        # Prepare logging dictionary
        log_dict = {
            'pretrain/step': step,
            'pretrain/policy_loss': float(step_metrics['policy_loss']),
            'pretrain/bc_accuracy': float(step_metrics['bc_accuracy']),
            'pretrain/ce_loss': float(step_metrics.get('ce_loss', 0.0)),
            'pretrain/entropy_loss': float(step_metrics.get('entropy_loss', 0.0)),
            'pretrain/entropy': float(step_metrics.get('entropy', 0.0)),
            'pretrain/param_update_magnitude': param_update_mag,
        }
        
        if value_regression_enabled:
            log_dict['pretrain/value_loss'] = float(step_metrics['value_loss'])
        
        # Evaluate policy performance
        eval_metrics = None
        if eval_starts is not None and eval_pickups is not None:
            if step % eval_frequency == 0 or step == num_pretrain_steps - 1:
                eval_metrics = evaluate_policy_performance(current_policy_params, eval_starts, eval_pickups)
                if eval_metrics:
                    log_dict.update({
                        'pretrain/eval_avg_reward': eval_metrics['avg_reward'],
                        'pretrain/eval_avg_steps': eval_metrics['avg_steps'],
                        'pretrain/eval_completion_rate': eval_metrics['completion_rate'],
                        'pretrain/eval_bc_accuracy': eval_metrics['bc_accuracy'],
                    })
        
        if log_fn:
            log_fn(log_dict, step)
        
        if step % 100 == 0 or step == num_pretrain_steps - 1:
            print(f"BC Pretraining step {step}/{num_pretrain_steps}: "
                  f"Policy loss = {float(step_metrics['policy_loss']):.6f}, "
                  f"BC accuracy = {float(step_metrics['bc_accuracy']):.4f}, "
                  f"Entropy = {float(step_metrics.get('entropy', 0.0)):.4f}")
            if value_regression_enabled:
                print(f"  Value loss = {float(step_metrics['value_loss']):.6f}")
            if eval_metrics:
                print(f"  Eval - Avg reward: {eval_metrics['avg_reward']:.4f}, "
                      f"Avg steps: {eval_metrics['avg_steps']:.2f}, "
                      f"Completion: {eval_metrics['completion_rate']:.2%}, "
                      f"BC accuracy: {eval_metrics['bc_accuracy']:.2%}")
    
    print("BC Pretraining completed!")
    
    if value_regression_enabled:
        return current_policy_params, current_policy_opt_state, current_key, current_value_params, current_value_opt_state
    else:
        return current_policy_params, current_policy_opt_state, current_key

# PPO Training Functions
def get_ppo_init_fn(env, config, obs_fn_single):
    """Initialize PPO networks and optimizers"""
    
    def init_fn(key):
        dummy_state, _ = init_env(
            jax_random.PRNGKey(0), 
            env.fixed_starts[0], 
            env.fixed_pickups[0], 
            env.neighbor_mask_static
        )
        dummy_obs = obs_fn_single(dummy_state)
        dummy_obs_batched = jnp.expand_dims(dummy_obs, axis=0)
        
        use_gnn = config.get('use_gnn', False) and GNN_AVAILABLE
        if use_gnn:
            # GNN networks
            print("Using GNN architecture for policy and value networks")
            config['num_nodes'] = env.num_nodes
            policy_net = hk.without_apply_rng(hk.transform(lambda obs: GNPPolicyNetwork(config)(obs)))
            key, sk = jax.random.split(key)
            policy_params = policy_net.init(sk, dummy_obs_batched)
            policy_apply = policy_net.apply
            
            value_net = hk.without_apply_rng(hk.transform(lambda obs: GNNValueNetwork(config)(obs)))
            key, sk = jax.random.split(key)
            value_params = value_net.init(sk, dummy_obs_batched)
            value_apply = value_net.apply
        else:
            # MLP networks
            if use_gnn and not GNN_AVAILABLE:
                print("Warning: GNN requested but not available, using MLP instead")
            policy_net = hk.without_apply_rng(hk.transform(lambda obs: PPOPolicyNetwork(config)(obs)))
            key, sk = jax.random.split(key)
            policy_params = policy_net.init(sk, dummy_obs_batched)
            policy_apply = policy_net.apply
            
            value_net = hk.without_apply_rng(hk.transform(lambda obs: PPOValueNetwork(config)(obs)))
            key, sk = jax.random.split(key)
            value_params = value_net.init(sk, dummy_obs_batched)
            value_apply = value_net.apply
        
        # Learning rate schedules
        policy_lr_schedule = optax.linear_schedule(
            init_value=config.get('policy_lr', 1e-4),
            end_value=config.get('policy_lr', 1e-4) * 0.1,
            transition_steps=config.get('num_steps', 50000)
        )
        
        value_lr_schedule = optax.linear_schedule(
            init_value=config.get('value_lr', 1e-4),
            end_value=config.get('value_lr', 1e-4) * 0.1,
            transition_steps=config.get('num_steps', 50000)
        )
        
        # Optimizers
        policy_opt = optax.chain(
            optax.clip_by_global_norm(config.get('max_grad_norm', 0.5)),
            optax.adamw(
                policy_lr_schedule,
                eps=config.get('eps_adam', 1e-5),
                b1=config.get('b1_adam', 0.9),
                b2=config.get('b2_adam', 0.999),
                weight_decay=config.get('wd_adam', 1e-4)
            )
        )
        
        value_opt = optax.chain(
            optax.clip_by_global_norm(config.get('max_grad_norm', 0.5)),
            optax.adamw(
                value_lr_schedule,
                eps=config.get('eps_adam', 1e-5),
                b1=config.get('b1_adam', 0.9),
                b2=config.get('b2_adam', 0.999),
                weight_decay=config.get('wd_adam', 1e-4)
            )
        )
        
        policy_opt_state = policy_opt.init(policy_params)
        value_opt_state = value_opt.init(value_params)
        
        return (key, policy_params, value_params, policy_apply, value_apply, 
                policy_opt, value_opt, policy_opt_state, value_opt_state)
    
    return init_fn

def get_ppo_agent_loop(env, config, obs_fn_batch, obs_fn_single, policy_apply, value_apply, 
                      policy_opt, value_opt, epsilon_schedule_fn=None, eval_starts=None, eval_pickups=None, bc_policy_params=None):
    """Create PPO training loop with proper loss computation and evaluation
    
    After BC pretraining:
    - Uses small clip=0.1, entropy≈0.01-0.02
    - Adds KL penalty to frozen BC policy (decays to 0)
    """
    
    # Batch operations
    batch_step = jax.jit(vmap(env.step, in_axes=(0, 0)))
    batch_reset = jax.jit(vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0)))        
    
    # KL penalty schedule
    use_kl_penalty = config.get('use_kl_penalty', False) and bc_policy_params is not None
    if use_kl_penalty:
        kl_penalty_initial = config.get('kl_penalty_initial', 0.1)
        kl_penalty_decay_steps = config.get('kl_penalty_decay_steps', config.get('num_steps', 50000))
        
        def kl_penalty_schedule(step):
            """KL penalty schedule: decays linearly from initial to 0"""
            progress = jnp.clip(step / kl_penalty_decay_steps, 0.0, 1.0)
            return kl_penalty_initial * (1.0 - progress)
    else:
        def kl_penalty_schedule(step):
            return 0.0
    
    # Entropy coefficient schedule
    entropy_decay_enabled = config.get('entropy_decay_enabled', True)
    if entropy_decay_enabled:
        entropy_coef_initial = config.get('entropy_coef_initial', config.get('entropy_coef', 0.01))
        entropy_coef_final = config.get('entropy_coef_final', 0.001)
        entropy_decay_steps = config.get('entropy_decay_steps', config.get('num_steps', 50000))
        
        def entropy_coef_schedule(step):
            progress = jnp.clip(step / entropy_decay_steps, 0.0, 1.0)
            return entropy_coef_initial * (1.0 - progress) + entropy_coef_final * progress
    else:
        fixed_entropy_coef = config.get('ppo_entropy_coef', config.get('entropy_coef', 0.01))
        def entropy_coef_schedule(step):
            return fixed_entropy_coef
    
    # Evaluation set
    if eval_starts is None or eval_pickups is None:
        raise ValueError("Fixed evaluation set must be provided")
    
    # Evaluation function
    def evaluate_policy_on_fixed_set(policy_params, starts, pickups):
        """Evaluate policy on fixed start-pickup pairs"""
        eval_keys = jax.random.split(jax.random.PRNGKey(42), len(starts))
        eval_states, _ = batch_reset(eval_keys, starts, pickups)
        
        total_rewards = jnp.zeros(len(starts))
        total_steps = jnp.zeros(len(starts))
        completed = jnp.zeros(len(starts), dtype=bool)
        
        for step in range(env.max_steps):
            # Skip completed
            active_mask = ~completed
            
            if bool(jnp.any(active_mask)):
                # Get observations and policy logits only for active
                obs = obs_fn_batch(eval_states)
                logits = policy_apply(policy_params, obs)
                
                neighbor_masks = eval_states.neighbor_mask
                masked_logits = jnp.where(neighbor_masks, logits, -1e8)
                
                actions = jnp.argmax(masked_logits, axis=-1)
                
                next_states, rewards, terminals, info = batch_step(eval_states, actions)
                
                # Note: terminals (done) is True only when reaching pickup (invalid moves error)
                # So completed == reached_pickup
                
                # Only update metrics for active episodes
                total_rewards = jnp.where(active_mask, total_rewards + rewards, total_rewards)
                total_steps = jnp.where(active_mask, total_steps + 1, total_steps)
                completed = completed | terminals
                
                # Update states
                eval_states = next_states
                
                # Break if all episodes completed
                if bool(jnp.all(completed)):
                    break
            else:
                # All completed, break early
                break
        
        num_episodes = len(starts)
        # completed is equivalent to reached_pickup since done only happens when reaching pickup
        reached_pickup = completed
        return {
            'total_rewards': total_rewards,
            'total_steps': total_steps,
            'reached_pickup': reached_pickup,
            'avg_reward': jnp.mean(total_rewards),
            'avg_steps': jnp.mean(total_steps),
            'completion_rate': jnp.mean(completed.astype(float)),
            'num_reached_pickup': jnp.sum(reached_pickup.astype(int)),
            'reached_pickup_rate': float(jnp.sum(reached_pickup.astype(float)) / num_episodes) if num_episodes > 0 else 0.0
        }
    
    # Experience buffer
    buffer = PPOExperienceBuffer(
        buffer_size=config.get('buffer_size', 10000),
        obs_dim=5,
        max_deg=env.max_deg
    )
    
    # GAE
    @jax.jit
    def compute_gae_jit(rewards, values, dones, gamma=0.99, lam=0.95, last_value=0.0):
        """
        Compute GAE with support for custom terminal state values.
        
        Args:
            rewards: [T] array of rewards
            values: [T] array of value estimates
            dones: [T] array of done flags
            gamma: discount factor
            lam: GAE lambda parameter
            last_value: value estimate for the state after the last step (0.0 if reached pickup)
        """
        # For the last step, use the provided last_value
        next_values = jnp.concatenate([values[1:], jnp.array([last_value])])
        
        # Debug: Check GAE inputs
        # Note: This will print for every trajectory, but JAX will optimize it
        num_done = jnp.sum(dones.astype(jnp.int32))
        traj_length = rewards.shape[0]
        jax.debug.print(
            "[GAE_fn] T={T} last_value={lv:.4f} num_done={nd} mean_reward={mr:.4f} mean_value={mv:.4f}",
            T=traj_length,
            lv=last_value,
            nd=num_done,
            mr=jnp.mean(rewards),
            mv=jnp.mean(values)
        )
        
        # TD errors
        deltas = rewards + gamma * next_values * (1 - dones) - values
        
        # GAE
        def gae_step(carry, delta_done):
            delta, done = delta_done
            gae = delta + gamma * lam * (1 - done) * carry
            return gae, gae
        
        # Reverse the inputs
        deltas_reversed = jnp.flip(deltas)
        dones_reversed = jnp.flip(dones)
        
        _, advantages_reversed = jax.lax.scan(gae_step, 0.0, (deltas_reversed, dones_reversed))
        
        # Reverse
        advantages = jnp.flip(advantages_reversed)
        
        return advantages
    
    # Policy loss
    @jax.jit
    def policy_loss_jit(policy_params, batch, old_log_probs, clip_ratio=0.2, entropy_coef=0.01, 
                        bc_policy_params=None, kl_penalty_weight=0.0):
        """Policy loss with optional KL penalty to BC policy"""
        loss, info = policy_loss_fn(policy_apply, policy_params, batch, old_log_probs, clip_ratio, entropy_coef)
        
        # KL penalty to frozen BC policy if enabled
        def compute_kl(_):
            # Current policy log probs
            logits = policy_apply(policy_params, batch['observations'])
            masked_logits = jnp.where(batch['masks'], logits, -1e8)
            current_log_probs = jax.nn.log_softmax(masked_logits)
            current_probs = jax.nn.softmax(masked_logits)
            
            # BC policy log probs
            bc_logits = policy_apply(bc_policy_params, batch['observations'])
            bc_masked_logits = jnp.where(batch['masks'], bc_logits, -1e8)
            bc_log_probs = jax.nn.log_softmax(bc_masked_logits)
            
            # KL divergence: KL(current || bc) = sum(current_probs * (log(current_probs) - log(bc_probs)))
            kl_div = jnp.mean(jnp.sum(
                current_probs * (current_log_probs - bc_log_probs),
                axis=-1
            ))
            kl_penalty = kl_penalty_weight * kl_div
            new_loss = loss + kl_penalty
            new_info = {**info, 'kl_div': kl_div, 'kl_penalty': kl_penalty}
            return new_loss, new_info
        
        def skip_kl(_):
            return loss, {**info, 'kl_div': jnp.array(0.0), 'kl_penalty': jnp.array(0.0)}
        
        # KL only if bc_policy_params exists and weight > 0
        bc_available = bc_policy_params is not None
        pred = jnp.logical_and(jnp.array(bc_available), kl_penalty_weight > 0.0)
        loss, info = jax.lax.cond(pred, compute_kl, skip_kl, operand=None)
        
        return loss, info
    
    # Value loss
    @jax.jit
    def value_loss_jit(value_params, batch, value_coef=0.5, value_clip_ratio=0.2):
        """Value loss"""
        return value_loss_fn(value_apply, value_params, batch, value_coef, value_clip_ratio)
    
    # PPO loss
    @jax.jit
    def ppo_loss_jit(policy_params, value_params, batch, old_log_probs, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        """PPO loss"""
        return ppo_loss_fn(policy_apply, value_apply, policy_params, value_params, batch, old_log_probs, clip_ratio, value_coef, entropy_coef)
    
    def loop_fn(state_dict, _):
        """PPO"""
        # Get current observations
        obs = obs_fn_batch(state_dict['env_states'])
        
        # Policy logits
        policy_logits = policy_apply(state_dict['policy_params'], obs)
        
        # Action masking
        neighbor_mask = state_dict['env_states'].neighbor_mask
        
        masked_logits = jnp.where(
            neighbor_mask,
            policy_logits,
            -1e8
        )
        
        state_dict['key'], sample_key = jax.random.split(state_dict['key'], 2)
        sample_keys = jax.random.split(sample_key, config['batch_size'])
        actions = jax.vmap(jax.random.categorical)(sample_keys, masked_logits)

        # Log probabilities
        log_probs = jax.nn.log_softmax(masked_logits)
        selected_log_probs = jnp.sum(log_probs * jax.nn.one_hot(actions, masked_logits.shape[-1]), axis=-1)
        
        # Values
        values = value_apply(state_dict['value_params'], obs)
        
        # Environment step
        next_states, rewards, terminals, info = batch_step(state_dict['env_states'], actions)
        
        # Debug: Check episode termination behavior
        num_terminated = jnp.sum(terminals.astype(jnp.int32))
        jax.debug.print(
            "[step] step={step} terminated={term} mean_reward={r:.4f} mean_value={v:.4f}",
            step=state_dict['opt_t'],
            term=num_terminated,
            r=jnp.mean(rewards),
            v=jnp.mean(values)
        )
        
        # Experiences
        # Note: terminals (done) is True only when reaching pickup (invalid moves error)
        experiences = {
            'observations': obs,
            'actions': actions,
            'rewards': rewards,
            'values': values,
            'log_probs': selected_log_probs,
            'masks': neighbor_mask,
            'dones': terminals
        }
        # No reset logic - agents stay at pickup when done (invalid moves will error)
        updated_env_states = next_states
        
        # Update statistics
        episode_return = state_dict['episode_return'] + rewards
        
        # Calculate mean return only for terminated episodes (when they reach pickup)
        terminated_returns = jnp.where(terminals, episode_return, 0.0)
        num_terminations = jnp.sum(terminals.astype(jnp.int32))
        mean_terminated_return = jnp.where(
            num_terminations > 0,
            jnp.sum(terminated_returns) / num_terminations,
            0.0
        )
        avg_return = state_dict['avg_return']
        num_episodes = jnp.where(jnp.any(terminals), state_dict['num_episodes'] + jnp.sum(terminals.astype(jnp.int32)), state_dict['num_episodes'])
        
        new_state_dict = {
            **state_dict,
            'env_states': updated_env_states,
            'episode_return': episode_return,
            'avg_return': avg_return,
            'num_episodes': num_episodes,
            'opt_t': state_dict['opt_t'] + 1,
        }
        
        return new_state_dict, {
            'loss': 0.0,
            'avg_return': jnp.mean(new_state_dict['avg_return']),
            'num_episodes': jnp.sum(new_state_dict['num_episodes']),
            'experiences': experiences,
        }
    
    def run_loop(state_dict):
        """PPO"""
        # Clear buffer before collecting new data
        buffer.reset()
        
        # Environment steps
        state_dict, metrics = jax.lax.scan(loop_fn, state_dict, None, length=config.get('eval_frequency', 100))
        
        # Experiences
        all_experiences = {
            'observations': metrics['experiences']['observations'],
            'actions': metrics['experiences']['actions'],
            'rewards': metrics['experiences']['rewards'],
            'values': metrics['experiences']['values'],
            'log_probs': metrics['experiences']['log_probs'],
            'masks': metrics['experiences']['masks'],
            'dones': metrics['experiences']['dones']
        }
        
        # Flatten experiences
        batch_size = all_experiences['observations'].shape[1]
        num_steps = all_experiences['observations'].shape[0]
        
        obs_flat = all_experiences['observations'].reshape(-1, all_experiences['observations'].shape[-1])
        actions_flat = all_experiences['actions'].reshape(-1)
        rewards_flat = all_experiences['rewards'].reshape(-1)
        values_flat = all_experiences['values'].reshape(-1)
        log_probs_flat = all_experiences['log_probs'].reshape(-1)
        masks_flat = all_experiences['masks'].reshape(-1, all_experiences['masks'].shape[-1])
        dones_flat = all_experiences['dones'].reshape(-1)
        # Weight experiences: exclude done experiences (no learning signal after reaching pickup)
        weights_flat = jnp.ones_like(dones_flat, dtype=jnp.float32)

        # Compute last values for GAE computation
        # Get final states after the rollout
        final_states = state_dict['env_states']
        final_obs = obs_fn_batch(final_states)
        final_values = value_apply(state_dict['value_params'], final_obs)  # [batch_size]
        
        # For each trajectory, check if it ended (done=True at the last step)
        # If done=True, episode ended (reached pickup), so last_value should be 0
        # Otherwise, episode is still ongoing, so use the value estimate
        last_done = all_experiences['dones'][-1, :]  # [batch_size]
        
        # Set last_value to 0 if done, otherwise use the computed value estimate
        last_values = jnp.where(last_done, 0.0, final_values)  # [batch_size]
        
        # Debug: Check GAE last value computation
        num_done_at_end = jnp.sum(last_done.astype(jnp.int32))
        mean_final_value = jnp.mean(final_values)
        mean_last_value = jnp.mean(last_values)
        jax.debug.print(
            "[GAE] num_done_at_end={nd} mean_final_value={fv:.4f} mean_last_value={lv:.4f}",
            nd=num_done_at_end,
            fv=mean_final_value,
            lv=mean_last_value
        )

        # GAE
        def compute_gae_single_trajectory(rewards, values, dones, last_value):
            """GAE with last_value parameter"""
            return compute_gae_jit(
                rewards, values, dones,
                gamma=config.get('gamma', 0.99),
                lam=config.get('gae_lambda', 0.95),
                last_value=last_value
            )
        
        # GAE - pass last_values for each trajectory
        advantages_per_env = jax.vmap(compute_gae_single_trajectory, in_axes=(1, 1, 1, 0))(
            all_experiences['rewards'],
            all_experiences['values'],
            all_experiences['dones'],
            last_values  # [batch_size]
        )
        
        # Transpose and flatten
        advantages_flat = advantages_per_env.T.reshape(-1)
        returns_flat = advantages_flat + all_experiences['values'].reshape(-1)
        
        # Debug: Check advantages and returns
        mean_advantage = jnp.mean(advantages_flat)
        std_advantage = jnp.std(advantages_flat)
        mean_return = jnp.mean(returns_flat)
        std_return = jnp.std(returns_flat)
        min_adv = jnp.min(advantages_flat)
        max_adv = jnp.max(advantages_flat)
        jax.debug.print(
            "[advantages] mean={ma:.4f} std={sa:.4f} min={min:.4f} max={max:.4f} mean_return={mr:.4f} std_return={sr:.4f}",
            ma=mean_advantage,
            sa=std_advantage,
            min=min_adv,
            max=max_adv,
            mr=mean_return,
            sr=std_return
        )

        total_entries = obs_flat.shape[0]
        
        # Debug: Check experience statistics before adding to buffer
        num_done_in_buffer = jnp.sum(dones_flat.astype(jnp.int32))
        jax.debug.print(
            "[buffer] total_entries={te} num_done={nd} buffer_size={bs}",
            te=total_entries,
            nd=num_done_in_buffer,
            bs=buffer.buffer_size
        )
        
        buffer.observations = buffer.observations.at[:total_entries].set(obs_flat)
        buffer.actions = buffer.actions.at[:total_entries].set(actions_flat)
        buffer.rewards = buffer.rewards.at[:total_entries].set(rewards_flat)
        buffer.values = buffer.values.at[:total_entries].set(values_flat)
        buffer.log_probs = buffer.log_probs.at[:total_entries].set(log_probs_flat)
        buffer.masks = buffer.masks.at[:total_entries].set(masks_flat)
        buffer.dones = buffer.dones.at[:total_entries].set(dones_flat)
        buffer.advantages = buffer.advantages.at[:total_entries].set(advantages_flat)
        buffer.returns = buffer.returns.at[:total_entries].set(returns_flat)
        buffer.weights = buffer.weights.at[:total_entries].set(weights_flat)
        buffer.size = min(total_entries, buffer.buffer_size)
        buffer.ptr = buffer.size % buffer.buffer_size
        valid_count = int(jnp.sum(weights_flat).item())
        filtered_count = total_entries - valid_count
        filter_rate = float(filtered_count / total_entries) if total_entries > 0 else 0.0
        
        # PPO
        total_policy_loss = 0.0
        total_value_loss = 0.0
        min_buffer_size = config.get('min_buffer_size', 32)
        
        # Initialize KL metrics
        kl_div = 0.0
        kl_penalty_val = 0.0
        policy_entropy = 0.0
        
        if buffer.size >= min_buffer_size:
            # Debug: Training started
            jax.debug.print("[training] buffer_size={bs} min_buffer_size={mbs} training_active", 
                          bs=buffer.size, mbs=min_buffer_size)
            
            # PPO
            for epoch in range(config.get('ppo_epochs', 4)):
                # Get batch for PPO update with proper random key
                state_dict['key'], batch_key = jax.random.split(state_dict['key'])
                ppo_batch = buffer.get_batch(config.get('ppo_batch_size', 64), batch_key)
                
                # Normalize
                batch_advantages = ppo_batch['advantages']
                batch_weights = ppo_batch.get('weights', jnp.ones_like(batch_advantages))
                weight_sum = jnp.sum(batch_weights) + 1e-8
                advantage_mean = jnp.sum(batch_advantages * batch_weights) / weight_sum
                advantage_var = jnp.sum(((batch_advantages - advantage_mean) ** 2) * batch_weights) / weight_sum
                advantage_std = jnp.sqrt(advantage_var + 1e-8)
                batch_advantages_normalized = (batch_advantages - advantage_mean) / (advantage_std + 1e-8)
                ppo_batch_normalized = {
                    **ppo_batch,
                    'advantages': batch_advantages_normalized,
                    'weights': batch_weights
                }
                
                # KL penalty weight
                kl_weight = kl_penalty_schedule(state_dict['opt_t'])
                
                # BC hyperparameters
                clip_ratio = config.get('ppo_clip_ratio', config.get('clip_ratio', 0.2))
                # Entropy schedule
                entropy_coef = entropy_coef_schedule(state_dict['opt_t'])
                
                # Policy
                bc_params_for_kl = bc_policy_params if bc_policy_params is not None else state_dict['policy_params']

                (policy_loss, policy_info), policy_grads = jax.value_and_grad(
                    lambda p: policy_loss_jit(
                        p, ppo_batch_normalized, ppo_batch_normalized['log_probs'],
                        clip_ratio=clip_ratio,
                        entropy_coef=entropy_coef,
                        bc_policy_params=bc_params_for_kl,
                        kl_penalty_weight=kl_weight
                    ), has_aux=True
                )(state_dict['policy_params'])
                
                policy_updates, policy_opt_state = policy_opt.update(policy_grads, state_dict['policy_opt_state'], state_dict['policy_params'])
                state_dict['policy_params'] = optax.apply_updates(state_dict['policy_params'], policy_updates)
                state_dict['policy_opt_state'] = policy_opt_state
                
                # Value
                (value_loss, value_info), value_grads = jax.value_and_grad(
                    lambda v: value_loss_jit(
                        v, ppo_batch, value_coef=config.get('value_coef', 0.5), value_clip_ratio=config.get('value_clip_ratio', 0.2)
                    ), has_aux=True
                )(state_dict['value_params'])
                
                value_updates, value_opt_state = value_opt.update(value_grads, state_dict['value_opt_state'], state_dict['value_params'])
                state_dict['value_params'] = optax.apply_updates(state_dict['value_params'], value_updates)
                state_dict['value_opt_state'] = value_opt_state
                
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                
                # Debug: Check losses per epoch
                jax.debug.print(
                    "[losses] epoch={ep} policy_loss={pl:.6f} value_loss={vl:.6f} entropy={ent:.6f}",
                    ep=epoch,
                    pl=policy_loss,
                    vl=value_loss,
                    ent=policy_info.get('entropy', 0.0)
                )
                
                # KL metrics
                if 'kl_div' in policy_info:
                    kl_div = float(policy_info['kl_div'])
                    kl_penalty_val = float(policy_info.get('kl_penalty', 0.0))
                
                # Entropy
                policy_entropy = float(policy_info.get('entropy', 0.0))
                
        # Reset
        state_dict["key"], subkey = jax.random.split(state_dict["key"])
        subkeys = jax.random.split(subkey, config['batch_size'])
        starts = jnp.stack([jax_random.choice(k, env.fixed_starts) for k in subkeys])
        pickups = jnp.stack([jax_random.choice(k, env.fixed_pickups) for k in subkeys])
        
        batch_init = vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0))
        state_dict['env_states'], _ = batch_init(subkeys, starts, pickups)
        state_dict['last_start'] = starts
        state_dict['key'] = subkey
        
        # Evaluation
        eval_results = evaluate_policy_on_fixed_set(state_dict['policy_params'], eval_starts, eval_pickups)

        # Metrics
        aggregated_metrics = {
            'policy_loss': total_policy_loss,
            'value_loss': total_value_loss,
            'total_loss': total_policy_loss + total_value_loss,
            'avg_return': jnp.mean(metrics['avg_return']),
            'num_episodes': jnp.sum(metrics['num_episodes']),
            'buffer_size': buffer.size,
            'buffer_utilization': buffer.size / buffer.buffer_size,
            'learning_active': buffer.size >= min_buffer_size,
            'filtered_experiences': int(filtered_count),
            'filter_rate': filter_rate,
            
            # KL penalty metrics
            'kl/kl_div': kl_div if use_kl_penalty else 0.0,
            'kl/kl_penalty': kl_penalty_val if use_kl_penalty else 0.0,
            'kl/kl_weight': float(kl_penalty_schedule(state_dict['opt_t'])),
            
            # Entropy coefficient
            'entropy/entropy_coef': float(entropy_coef_schedule(state_dict['opt_t'])),
            'entropy/policy_entropy': policy_entropy,
            
            # Evaluation
            'eval/avg_reward': float(eval_results['avg_reward']),
            'eval/avg_steps': float(eval_results['avg_steps']),
            'eval/num_reached_pickup': int(eval_results['num_reached_pickup']),
            'eval/reached_pickup_rate': float(eval_results['reached_pickup_rate']),
            'eval/total_rewards': eval_results['total_rewards'].tolist(),
            'eval/total_steps': eval_results['total_steps'].tolist(),
            'eval/reached_pickup': eval_results['reached_pickup'].tolist()
        }
        
        return state_dict, aggregated_metrics
    
    return run_loop
