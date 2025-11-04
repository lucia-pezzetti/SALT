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
from utils import smart_greedy_next_hop, smart_greedy_next_hop_batch

# map activation names
activation_dict = {"relu": jax.nn.relu, "silu": jax.nn.silu, "elu": jax.nn.elu}

# --- PPO Policy Network ---
class PPOPolicyNetwork(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config['num_hidden_units']
        self.num_hidden_layers = config['num_hidden_layers']
        self.activation = activation_dict[config['activation']]
        self.max_deg = config['max_deg']  # Maximum degree of nodes
        
    def __call__(self, obs):
        x = jnp.ravel(obs)
        for _ in range(self.num_hidden_layers):
            x = self.activation(hk.Linear(
                self.num_hidden_units,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                b_init=hk.initializers.Constant(0.0)
            )(x))
        # Output logits for all possible actions
        return hk.Linear(
            self.max_deg,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
            b_init=hk.initializers.Constant(0.0)
        )(x)

# --- PPO Value Network ---
class PPOValueNetwork(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config['num_hidden_units']
        self.num_hidden_layers = config['num_hidden_layers']
        self.activation = activation_dict[config['activation']]
        self.init_bias = config.get('V_init_bias', 0.0)
        
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
            b_init=hk.initializers.Constant(self.init_bias)
        )(x)[0]

# --- Graph-Aware PPO Networks ---
class GraphAwarePPOPolicyNetwork(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.hidden_dim = config['num_hidden_units']
        self.num_layers = config['num_hidden_layers'] 
        self.activation = activation_dict[config['activation']]
        self.max_deg = config.get('max_deg', 8)
        self.cycle_length = config.get('cycle_length', 200)
        
    def __call__(self, obs):
        """
        Policy network for lat/lon-based observations.
        Input: obs [..., 5] = [current_pos(2), pickup_pos(2), time(1)]
        """
        # Extract features from the 5-dimensional observation
        current_pos = obs[..., 0:2]      # [..., 2] - current position (lat/lon)
        pickup_pos = obs[..., 2:4]       # [..., 2] - pickup position (lat/lon)
        # relative_pos = obs[..., 4:6]     # [..., 2] - direction vector
        # distance = obs[..., 6:7]         # [..., 1] - distance to pickup
        # angle = obs[..., 7:8]            # [..., 1] - direction angle
        time = obs[..., 4:5]             # [..., 1] - time
        
        # Normalize time feature
        time = time / self.cycle_length
        
        # # Normalize distance
        # distance = distance / 1.0
        
        # # Normalize angle to [-1, 1] range
        # angle = angle / jnp.pi

        # Combine all features
        features = jnp.concatenate([
            current_pos,    # [..., 2]
            pickup_pos,     # [..., 2]
            time            # [..., 1]
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
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            if residual is not None:
                x = x + residual  # Residual connection
                
        # Final layer for action logits
        return hk.Linear(
            self.max_deg,
            w_init=hk.initializers.VarianceScaling(0.1, "fan_in", "truncated_normal"),
            b_init=hk.initializers.Constant(0.0)
        )(x)

class GraphAwarePPOValueNetwork(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.hidden_dim = config['num_hidden_units']
        self.num_layers = config['num_hidden_layers'] 
        self.activation = activation_dict[config['activation']]
        self.cycle_length = config.get('cycle_length', 200)
        self.init_bias = config.get('V_init_bias', 0.0)  # Initialize with config-specified bias
        
    def __call__(self, obs):
        """
        Value network for lat/lon-based observations.
        Input: obs [..., 5] = [current_pos(2), pickup_pos(2), time(1)]
        """
        # Extract features from the 5-dimensional observation
        current_pos = obs[..., 0:2]      # [..., 2] - current position (lat/lon)
        pickup_pos = obs[..., 2:4]       # [..., 2] - pickup position (lat/lon)
        # relative_pos = obs[..., 4:6]     # [..., 2] - direction vector
        # distance = obs[..., 6:7]         # [..., 1] - distance to pickup
        # angle = obs[..., 7:8]            # [..., 1] - direction angle
        time = obs[..., 4:5]             # [..., 1] - time
        
        # Normalize time feature
        time = time / self.cycle_length
        
        # # Normalize distance
        # distance = distance / 1.0
        
        # # Normalize angle to [-1, 1] range
        # angle = angle / jnp.pi

        # Combine all features
        features = jnp.concatenate([
            current_pos,    # [..., 2]
            pickup_pos,     # [..., 2]
            # relative_pos,   # [..., 2]
            # distance,       # [..., 1]
            # angle,          # [..., 1]
            time            # [..., 1]
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
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            if residual is not None:
                x = x + residual  # Residual connection
                
        # Final layer for value
        return hk.Linear(
            1,
            w_init=hk.initializers.VarianceScaling(0.1, "fan_in", "truncated_normal"),
            b_init=hk.initializers.Constant(self.init_bias)
        )(x).squeeze(-1)

# --- PPO Experience Buffer ---
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
            'dones': self.dones[indices]
        }

# --- PPO Loss Functions ---
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
    
    # Compute GAE using scan (much faster than loops)
    def gae_step(carry, delta_done):
        delta, done = delta_done
        gae = delta + gamma * lam * (1 - done) * carry
        return gae, gae
    
    # Reverse the inputs for proper GAE computation
    deltas_reversed = jnp.flip(deltas)
    dones_reversed = jnp.flip(dones)
    
    _, advantages_reversed = jax.lax.scan(gae_step, 0.0, (deltas_reversed, dones_reversed))
    
    # Reverse back to original order
    advantages = jnp.flip(advantages_reversed)
    
    return advantages

def policy_loss_fn(policy_apply, policy_params, batch, old_log_probs, clip_ratio=0.2, entropy_coef=0.01):
    """Compute policy loss (actor loss) with entropy bonus"""
    logits = policy_apply(policy_params, batch['observations'])
    
    # Fix shape mismatch between policy logits and masks
    masks = batch['masks']
    if masks.shape[-1] != logits.shape[-1]:
        # Pad the masks to match policy logits shape
        raise ValueError(f"Shape mismatch between masks and logits: {masks.shape} != {logits.shape}")
    
    masked_logits = jnp.where(masks, logits, -1e8)
    log_probs = jax.nn.log_softmax(masked_logits)
    selected_log_probs = jnp.sum(log_probs * jax.nn.one_hot(batch['actions'], masked_logits.shape[-1]), axis=-1)

    # Use raw advantages (no normalization inside loss function)
    adv = batch['advantages']
    
    # PPO policy loss with clipping
    log_ratio = selected_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    clipped_ratio = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)
    policy_loss = -jnp.mean(jnp.minimum(ratio * adv, clipped_ratio * adv))
    
    # Entropy loss for exploration
    probs = jax.nn.softmax(masked_logits)
    entropy = -jnp.mean(jnp.sum(log_probs * probs, axis=-1))
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
    
    # Standard value loss without clipping for now
    value_loss = jnp.mean((values - returns) ** 2)
    
    return value_coef * value_loss, {
        'value_loss': value_loss
    }

def ppo_loss_fn(policy_apply, value_apply, policy_params, value_params, batch, old_log_probs, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
    """Compute combined PPO loss (for backward compatibility)"""
    policy_loss, policy_info = policy_loss_fn(policy_apply, policy_params, batch, old_log_probs, clip_ratio, entropy_coef)
    value_loss, value_info = value_loss_fn(value_apply, value_params, batch, value_coef)
    
    total_loss = policy_loss + value_loss
    
    return total_loss, {
        **policy_info,
        **value_info,
        'total_loss': total_loss
    }

# --- Behavioral Cloning Pretraining Function ---
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
    Behavioral Cloning pretraining: train policy to imitate expert shortest path actions.
    
    Builds dataset of (state, valid-action-mask, expert-action, return).
    Trains with masked cross-entropy on expert actions.
    Optionally regresses value head to returns.
    Keeps entropy regularization to avoid over-confidence.
    
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
    
    # Use separate pretraining optimizer if specified, otherwise use regular policy optimizer
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
    
    # Create evaluation function if eval sets provided
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
        
        # Run episodes with proper early stopping
        for step in range(env.max_steps):
            active_mask = ~completed
            if bool(jnp.any(active_mask)):
                obs = obs_fn_batch(eval_states)
                logits = policy_apply(policy_params, obs)
                neighbor_masks = eval_states.neighbor_mask
                masked_logits = jnp.where(neighbor_masks, logits, -1e8)
                actions = jnp.argmax(masked_logits, axis=-1)
                
                # Compute expert actions for BC accuracy
                expert_actions = smart_greedy_next_hop_batch(
                    eval_states.current_node,
                    eval_states.pickup_node,
                    env.adj_list,
                    env.travel_times,
                    env.distances,
                    neighbor_masks
                )
                
                # Count correct actions (only for active episodes)
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
        """Single BC pretraining step
        
        Builds dataset of (state, valid-action-mask, expert-action, return).
        Trains with masked cross-entropy on expert actions.
        Optionally regresses value head to returns.
        Includes entropy regularization.
        """
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
        # For each trajectory, collect (state, mask, action, reward) and compute returns
        max_rollout_steps = config.get('pretrain_rollout_steps', 20)
        
        # Store trajectories per episode (list of lists)
        trajectories = [[] for _ in range(pretrain_batch_size)]
        
        current_states = env_states
        for step in range(max_rollout_steps):
            active_mask = ~current_states.done
            if not bool(jnp.any(active_mask)):
                break
            
            # Compute expert (shortest path) actions for active states
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
            
            # Store data for active episodes (before terminal)
            for i in range(pretrain_batch_size):
                if not bool(current_states.done[i]) and not bool(terminals[i]):
                    trajectories[i].append({
                        'state': jax.tree_util.tree_map(lambda x: x[i], current_states),
                        'mask': neighbor_masks[i],
                        'action': expert_actions[i],
                        'reward': rewards[i]
                    })
            
            # Update for next iteration
            current_states = next_states
            
            # Stop if all episodes complete
            if bool(jnp.all(terminals | current_states.done)):
                break
        
        # Flatten trajectories and compute returns per trajectory
        all_states = []
        all_masks = []
        all_expert_actions = []
        all_returns = []
        
        for traj in trajectories:
            if len(traj) == 0:
                continue
            # Compute returns backward through trajectory
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
        
        # Define BC loss function with masked cross-entropy and entropy regularization
        def bc_loss_fn(p):
            policy_logits = policy_apply(p, obs)  # [batch_size, max_deg]
            # Apply masking to logits
            masked_logits = jnp.where(collected_masks, policy_logits, -1e-8)
            # Compute log probabilities
            log_probs = jax.nn.log_softmax(masked_logits)  # [batch_size, max_deg]
            # Select log prob of expert action
            expert_log_probs = jnp.sum(
                log_probs * jax.nn.one_hot(collected_expert_actions, env.max_deg),
                axis=-1
            )
            # Masked cross-entropy loss (negative log likelihood)
            ce_loss = -jnp.mean(expert_log_probs)
            
            # Entropy regularization to avoid over-confidence
            probs = jax.nn.softmax(masked_logits)
            entropy = -jnp.mean(jnp.sum(log_probs * probs, axis=-1))
            entropy_loss = -entropy_coef * entropy
            
            total_loss = ce_loss + entropy_loss
            return total_loss, {'ce_loss': ce_loss, 'entropy_loss': entropy_loss, 'entropy': entropy}
        
        # Compute policy gradients and update
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
        
        # Evaluate policy performance if evaluation set provided
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
        
        # Log metrics
        if log_fn:
            log_fn(log_dict, step)
        
        # Print progress
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

# --- PPO Training Functions ---
def get_ppo_init_fn(env, config, obs_fn_single):
    """Initialize PPO networks and optimizers"""
    
    def init_fn(key):
        # Create dummy observation for network initialization
        dummy_state, _ = init_env(
            jax_random.PRNGKey(0), 
            env.fixed_starts[0], 
            env.fixed_pickups[0], 
            env.neighbor_mask_static
        )
        dummy_obs = obs_fn_single(dummy_state)
        
        # Initialize policy network
        policy_net = hk.without_apply_rng(hk.transform(lambda obs: GraphAwarePPOPolicyNetwork(config)(obs)))
        key, sk = jax.random.split(key)
        policy_params = policy_net.init(sk, dummy_obs)
        policy_apply = policy_net.apply
        
        # Initialize value network
        value_net = hk.without_apply_rng(hk.transform(lambda obs: GraphAwarePPOValueNetwork(config)(obs)))
        key, sk = jax.random.split(key)
        value_params = value_net.init(sk, dummy_obs)
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
    
    # Curriculum learning schedule: alpha decays from initial_alpha to final_alpha over decay_steps
    # alpha = probability of using shortest path action (1.0 = always SP, 0.0 = always learned policy)
    curriculum_enabled = config.get('curriculum_learning', False)
    if curriculum_enabled:
        initial_alpha = config.get('curriculum_initial_alpha', 1.0)  # Start with 100% SP
        final_alpha = config.get('curriculum_final_alpha', 0.0)     # End with 0% SP
        hold_steps = config.get('curriculum_hold_steps', config.get('num_steps', 50000) * 0.3)  # Keep SP for this many steps
        decay_steps = config.get('curriculum_decay_steps', config.get('num_steps', 50000))
        
        def curriculum_schedule(step):
            """Linear decay schedule for curriculum learning with hold period"""
            # Hold at initial_alpha for hold_steps, then decay linearly
            # Use jnp.where for JAX compatibility
            effective_step = step - hold_steps
            effective_decay_steps = jnp.maximum(decay_steps - hold_steps, 1)  # Ensure positive
            progress = jnp.clip(effective_step / effective_decay_steps, 0.0, 1.0)
            decayed_alpha = initial_alpha * (1.0 - progress) + final_alpha * progress
            # If step < hold_steps, return initial_alpha, otherwise return decayed_alpha
            return jnp.where(step < hold_steps, initial_alpha, decayed_alpha)
    else:
        def curriculum_schedule(step):
            """No curriculum learning - always return 0.0 (use learned policy)"""
            return 0.0
    
    # KL penalty schedule for BC policy (decays to 0)
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
    
    # Setup fixed evaluation set
    if eval_starts is None or eval_pickups is None:
        # Use default evaluation set from environment
        eval_starts = env.fixed_starts[:min(5, len(env.fixed_starts))]  # Use first 5 starts
        eval_pickups = env.fixed_pickups[:min(5, len(env.fixed_pickups))]  # Use first 5 pickups
    
    # print(f"Fixed evaluation set: {len(eval_starts)} starts, {len(eval_pickups)} pickups")
    # print(f"Evaluation starts: {eval_starts}")
    # print(f"Evaluation pickups: {eval_pickups}")
    
    # Create evaluation function (without JIT to avoid boolean conversion issues)
    def evaluate_policy_on_fixed_set(policy_params, starts, pickups):
        """Evaluate policy on fixed start-pickup pairs"""
        # Initialize evaluation environments
        eval_keys = jax.random.split(jax.random.PRNGKey(42), len(starts))
        eval_states, _ = batch_reset(eval_keys, starts, pickups)
        
        total_rewards = jnp.zeros(len(starts))
        total_steps = jnp.zeros(len(starts))
        completed = jnp.zeros(len(starts), dtype=bool)
        
        # Run episodes with proper early stopping
        for step in range(env.max_steps):
            # Skip completed episodes
            active_mask = ~completed
            
            # Convert JAX boolean to Python boolean for control flow
            if bool(jnp.any(active_mask)):
                # Get observations and policy logits only for active episodes
                obs = obs_fn_batch(eval_states)
                logits = policy_apply(policy_params, obs)
                
                # Apply action masking
                neighbor_masks = eval_states.neighbor_mask
                masked_logits = jnp.where(neighbor_masks, logits, -1e8)
                
                # Select actions greedily (no exploration during evaluation)
                actions = jnp.argmax(masked_logits, axis=-1)
                
                # Take environment step
                next_states, rewards, terminals, info = batch_step(eval_states, actions)
                
                # Only update metrics for active episodes
                total_rewards = jnp.where(active_mask, total_rewards + rewards, total_rewards)
                total_steps = jnp.where(active_mask, total_steps + 1, total_steps)
                completed = completed | terminals
                
                # Update states
                eval_states = next_states
                
                # Break if all episodes completed (convert JAX boolean to Python boolean)
                if bool(jnp.all(completed)):
                    break
            else:
                # All episodes completed, break early
                break
        
        return {
            'total_rewards': total_rewards,
            'total_steps': total_steps,
            'completed': completed,
            'avg_reward': jnp.mean(total_rewards),
            'avg_steps': jnp.mean(total_steps),
            'completion_rate': jnp.mean(completed.astype(float))
        }
    
    # Create experience buffer (outside JAX compilation)
    buffer = PPOExperienceBuffer(
        buffer_size=config.get('buffer_size', 10000),
        obs_dim=5,  # Observation dimension
        max_deg=env.max_deg
    )
    
    # JIT-compiled functions for PPO updates
    @jax.jit
    def compute_gae_jit(rewards, values, dones, gamma=0.99, lam=0.95):
        """JIT-compiled GAE computation - optimized vectorized version"""
        # Pad with zero for next values
        next_values = jnp.concatenate([values[1:], jnp.array([0.0])])
        
        # Compute TD errors
        deltas = rewards + gamma * next_values * (1 - dones) - values
        
        # Compute GAE using scan (much faster than loops)
        def gae_step(carry, delta_done):
            delta, done = delta_done
            gae = delta + gamma * lam * (1 - done) * carry
            return gae, gae
        
        # Reverse the inputs for proper GAE computation
        deltas_reversed = jnp.flip(deltas)
        dones_reversed = jnp.flip(dones)
        
        _, advantages_reversed = jax.lax.scan(gae_step, 0.0, (deltas_reversed, dones_reversed))
        
        # Reverse back to original order
        advantages = jnp.flip(advantages_reversed)
        
        return advantages
    
    @jax.jit
    def policy_loss_jit(policy_params, batch, old_log_probs, clip_ratio=0.2, entropy_coef=0.01, 
                        bc_policy_params=None, kl_penalty_weight=0.0):
        """JIT-compiled policy loss computation with optional KL penalty to BC policy"""
        loss, info = policy_loss_fn(policy_apply, policy_params, batch, old_log_probs, clip_ratio, entropy_coef)
        
        # Add KL penalty to frozen BC policy if enabled
        # Check if we should compute KL (kl_penalty_weight > 0 implies bc_policy_params is not None)
        # Use jax.lax.cond to handle the conditional computation
        def compute_kl(_):
            # Compute current policy log probs
            logits = policy_apply(policy_params, batch['observations'])
            masked_logits = jnp.where(batch['masks'], logits, -1e8)
            current_log_probs = jax.nn.log_softmax(masked_logits)
            current_probs = jax.nn.softmax(masked_logits)
            
            # Compute BC policy log probs
            bc_logits = policy_apply(bc_policy_params, batch['observations'])
            bc_masked_logits = jnp.where(batch['masks'], bc_logits, -1e8)
            bc_log_probs = jax.nn.log_softmax(bc_masked_logits)
            
            # Compute KL divergence: KL(current || bc) = sum(current_probs * (log(current_probs) - log(bc_probs)))
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
        
        # Condition: compute KL only if bc_policy_params exists and weight > 0
        bc_available = bc_policy_params is not None
        pred = jnp.logical_and(jnp.array(bc_available), kl_penalty_weight > 0.0)
        loss, info = jax.lax.cond(pred, compute_kl, skip_kl, operand=None)
        
        return loss, info
    
    @jax.jit
    def value_loss_jit(value_params, batch, value_coef=0.5, value_clip_ratio=0.2):
        """JIT-compiled value loss computation"""
        return value_loss_fn(value_apply, value_params, batch, value_coef, value_clip_ratio)
    
    @jax.jit
    def ppo_loss_jit(policy_params, value_params, batch, old_log_probs, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        """JIT-compiled combined PPO loss computation (for backward compatibility)"""
        return ppo_loss_fn(policy_apply, value_apply, policy_params, value_params, batch, old_log_probs, clip_ratio, value_coef, entropy_coef)
    
    def loop_fn(state_dict, _):
        """PPO training step"""
        # Get current observations
        obs = obs_fn_batch(state_dict['env_states'])
        
        # Get policy logits and sample actions
        policy_logits = policy_apply(state_dict['policy_params'], obs)
        
        # Apply action masking
        neighbor_mask = state_dict['env_states'].neighbor_mask
        
        # Fix shape mismatch between policy logits and neighbor mask
        if neighbor_mask.shape[-1] != policy_logits.shape[-1]:
            # Pad the neighbor mask to match policy logits shape
            # padding = policy_logits.shape[-1] - neighbor_mask.shape[-1]
            # neighbor_mask = jnp.pad(neighbor_mask, (0, padding), constant_values=False)
            raise ValueError(f"Shape mismatch between policy logits and neighbor mask: {policy_logits.shape} != {neighbor_mask.shape}")
        
        masked_logits = jnp.where(
            neighbor_mask,
            policy_logits,
            -1e8
        )
        
        # Curriculum learning: compute shortest path actions and mix with learned policy
        state_dict['key'], sample_key = jax.random.split(state_dict['key'])
        sample_keys = jax.random.split(sample_key, config['batch_size'])
        
        if curriculum_enabled:
            # Get current curriculum alpha (probability of using SP)
            current_alpha = curriculum_schedule(state_dict['opt_t'])
            
            # Compute shortest path actions for current states
            sp_actions = smart_greedy_next_hop_batch(
                state_dict['env_states'].current_node,
                state_dict['env_states'].pickup_node,
                env.adj_list,
                env.travel_times,
                env.distances,
                neighbor_mask
            )
            
            # Sample learned policy actions
            policy_actions = jax.vmap(jax.random.categorical)(sample_keys, masked_logits)
            
            # Mix actions based on curriculum alpha: with probability alpha use SP, else use learned policy
            # Generate random values for mixing decision
            state_dict['key'], mix_key = jax.random.split(state_dict['key'])
            mix_keys = jax.random.split(mix_key, config['batch_size'])
            mix_probs = jax.random.uniform(mix_keys)
            use_sp = mix_probs < current_alpha
            
            # Select actions: use SP if use_sp is True, otherwise use learned policy
            actions = jnp.where(use_sp, sp_actions, policy_actions)
            
            # For logging: track which actions came from SP
            sp_action_fraction = jnp.mean(use_sp.astype(float))
        else:
            # No curriculum learning - just use learned policy
            actions = jax.vmap(jax.random.categorical)(sample_keys, masked_logits)
            current_alpha = 0.0
            sp_action_fraction = 0.0
        
        # Debug prints for multi-agent behavior
        # jax.debug.print("=== MULTI-AGENT DEBUG ===")
        # jax.debug.print("Agent positions: {positions}", positions=state_dict['env_states'].current_node)
        # jax.debug.print("Agent pickups: {pickups}", pickups=state_dict['env_states'].pickup_node)
        # jax.debug.print("Agent actions: {actions}", actions=actions)
        # jax.debug.print("Agent starts: {starts}", starts=state_dict['last_start'])
        
        # Get log probabilities - always use learned policy log_probs (even for SP actions)
        # This is important for PPO training: we need policy log_probs for all actions
        log_probs = jax.nn.log_softmax(masked_logits)
        selected_log_probs = jnp.sum(log_probs * jax.nn.one_hot(actions, masked_logits.shape[-1]), axis=-1)
        
        # Get values
        values = value_apply(state_dict['value_params'], obs)
        
        # Take environment step
        next_states, rewards, terminals, info = batch_step(state_dict['env_states'], actions)
        
        # Debug prints for step results
        # jax.debug.print("Step results:")
        # jax.debug.print("  Next positions: {next_pos}", next_pos=next_states.current_node)
        # jax.debug.print("  Rewards: {rewards}", rewards=rewards)
        # jax.debug.print("  Terminals: {terminals}", terminals=terminals)
        # jax.debug.print("  Travel times: {travel}", travel=info['travel'])
        # jax.debug.print("  Wait times: {wait}", wait=info['wait'])
        
        # Store experiences for PPO updates (return them for collection outside JAX)
        experiences = {
            'observations': obs,
            'actions': actions,
            'rewards': rewards,
            'values': values,
            'log_probs': selected_log_probs,
            'masks': neighbor_mask,
            'dones': terminals
        }
        
        # Reset environments that are done
        state_dict["key"], subkey = jax.random.split(state_dict["key"])
        subkeys = jax.random.split(subkey, num=config['batch_size'])
        
        # Reset completed episodes, continue others
        # jax.debug.print("Reset logic:")
        # jax.debug.print("  Terminals before reset: {terminals}", terminals=terminals)
        # jax.debug.print("  Using starts: {starts}", starts=state_dict['last_start'])
        # jax.debug.print("  Using pickups: {pickups}", pickups=state_dict['env_states'].pickup_node)
        reset_states, _ = batch_reset(subkeys, state_dict['last_start'], state_dict['env_states'].pickup_node)
        # jax.debug.print("  Reset positions: {reset_pos}", reset_pos=reset_states.current_node)
        state_dict['env_states'] = jax.tree_util.tree_map(
            lambda reset, next_state: jnp.where(
                jnp.reshape(terminals, [terminals.shape[0]] + [1] * (len(next_state.shape) - 1)),
                reset,  # Use reset state for completed episodes
                next_state  # Use next_state for continuing episodes
            ),
            reset_states,
            next_states
        )
        
        # Update statistics - ensure scalar shapes
        episode_return = state_dict['episode_return'] + rewards
        avg_return = jnp.where(
            jnp.any(terminals),  # Use jnp.any to get scalar boolean
            (state_dict['avg_return'] * config.get('avg_return_smoothing', 0.99) + 
             jnp.mean(episode_return) * (1.0 - config.get('avg_return_smoothing', 0.99))),
            state_dict['avg_return']
        )
        num_episodes = jnp.where(jnp.any(terminals), state_dict['num_episodes'] + jnp.sum(terminals), state_dict['num_episodes'])
        
        # Debug episode statistics
        # jax.debug.print("Episode statistics:")
        # jax.debug.print("  Episode returns: {returns}", returns=episode_return)
        # jax.debug.print("  Avg return: {avg_return}", avg_return=avg_return)
        # jax.debug.print("  Num episodes: {num_episodes}", num_episodes=num_episodes)
        # jax.debug.print("  Episodes completed: {completed}", completed=jnp.sum(terminals))
        
        state_dict.update({
            'episode_return': jnp.where(terminals, 0, episode_return),
            'avg_return': avg_return,
            'num_episodes': num_episodes,
        })
        
        # PPO updates will be handled outside the JAX-compiled function
        
        state_dict['opt_t'] += 1
        
        return state_dict, {
            'loss': 0.0,  # Loss will be computed during updates
            'avg_return': jnp.mean(state_dict['avg_return']),  # Ensure scalar
            'num_episodes': jnp.sum(state_dict['num_episodes']),  # Ensure scalar
            'experiences': experiences,  # Return experiences for PPO updates
            'curriculum_alpha': current_alpha,  # Current curriculum learning alpha
            'sp_action_fraction': sp_action_fraction  # Fraction of actions from SP
        }
    
    def run_loop(state_dict):
        """PPO training loop with proper loss computation"""
        # CRITICAL: Clear buffer before collecting new data (on-policy requirement)
        buffer.reset()
        
        # Run environment steps
        state_dict, metrics = jax.lax.scan(loop_fn, state_dict, None, length=config.get('eval_frequency', 100))
        
        # Collect experiences from all steps
        # metrics is a dict with arrays, not a list of dicts
        all_experiences = {
            'observations': metrics['experiences']['observations'],
            'actions': metrics['experiences']['actions'],
            'rewards': metrics['experiences']['rewards'],
            'values': metrics['experiences']['values'],
            'log_probs': metrics['experiences']['log_probs'],
            'masks': metrics['experiences']['masks'],
            'dones': metrics['experiences']['dones']
        }
        
        # Add experiences to buffer (flatten batch dimensions)
        # Reshape from (steps, batch_size, ...) to (steps * batch_size, ...)
        batch_size = all_experiences['observations'].shape[1]
        num_steps = all_experiences['observations'].shape[0]
        
        # Compute GAE for each environment separately using vmap
        def compute_gae_single_trajectory(rewards, values, dones):
            """Compute GAE for a single trajectory"""
            return compute_gae_jit(
                rewards, values, dones,
                gamma=config.get('gamma', 0.99),
                lam=config.get('gae_lambda', 0.95)
            )
        
        # Apply GAE computation to each environment's trajectory
        advantages_per_env = jax.vmap(compute_gae_single_trajectory)(
            all_experiences['rewards'],      # (steps, batch_size)
            all_experiences['values'],       # (steps, batch_size) 
            all_experiences['dones']         # (steps, batch_size)
        )  # Result: (batch_size, steps)
        
        # Transpose to get (steps, batch_size) and flatten
        advantages_flat = advantages_per_env.T.reshape(-1)  # (steps * batch_size,)
        returns_flat = advantages_flat + all_experiences['values'].reshape(-1)
        
        # Vectorized buffer population: flatten (steps, batch, ...) -> (steps*batch, ...)
        obs_flat = all_experiences['observations'].reshape(-1, all_experiences['observations'].shape[-1])
        actions_flat = all_experiences['actions'].reshape(-1)
        rewards_flat = all_experiences['rewards'].reshape(-1)
        values_flat = all_experiences['values'].reshape(-1)
        log_probs_flat = all_experiences['log_probs'].reshape(-1)
        masks_flat = all_experiences['masks'].reshape(-1, all_experiences['masks'].shape[-1])
        dones_flat = all_experiences['dones'].reshape(-1)

        total_entries = obs_flat.shape[0]
        # For on-policy PPO here we reset each epoch, so no wrap-around handling needed
        buffer.observations = buffer.observations.at[:total_entries].set(obs_flat)
        buffer.actions = buffer.actions.at[:total_entries].set(actions_flat)
        buffer.rewards = buffer.rewards.at[:total_entries].set(rewards_flat)
        buffer.values = buffer.values.at[:total_entries].set(values_flat)
        buffer.log_probs = buffer.log_probs.at[:total_entries].set(log_probs_flat)
        buffer.masks = buffer.masks.at[:total_entries].set(masks_flat)
        buffer.dones = buffer.dones.at[:total_entries].set(dones_flat)
        buffer.advantages = buffer.advantages.at[:total_entries].set(advantages_flat)
        buffer.returns = buffer.returns.at[:total_entries].set(returns_flat)
        buffer.size = min(total_entries, buffer.buffer_size)
        buffer.ptr = buffer.size % buffer.buffer_size
        
        # Compute PPO loss and update networks if buffer has enough data
        total_policy_loss = 0.0
        total_value_loss = 0.0
        min_buffer_size = config.get('min_buffer_size', 32)
        
        # Initialize KL metrics
        kl_div = 0.0
        kl_penalty_val = 0.0
        
        # Debug buffer status
        # jax.debug.print("Buffer status:")
        # jax.debug.print("  Buffer size: {size}", size=buffer.size)
        # jax.debug.print("  Min buffer size: {min_size}", min_size=min_buffer_size)
        # jax.debug.print("  Learning active: {active}", active=buffer.size >= min_buffer_size)
        
        if buffer.size >= min_buffer_size:
            # PPO updates with proper separate loss functions
            # jax.debug.print("Starting PPO updates with {epochs} epochs", epochs=config.get('ppo_epochs', 4))
            for epoch in range(config.get('ppo_epochs', 4)):
                # Get batch for PPO update with proper random key
                state_dict['key'], batch_key = jax.random.split(state_dict['key'])
                ppo_batch = buffer.get_batch(config.get('ppo_batch_size', 64), batch_key)
                
                # Debug PPO batch
                # jax.debug.print("PPO Epoch {epoch}:", epoch=epoch)
                # jax.debug.print("  Batch size: {batch_size}", batch_size=ppo_batch['observations'].shape[0])
                # jax.debug.print("  Avg advantage: {avg_adv}", avg_adv=jnp.mean(ppo_batch['advantages']))
                # jax.debug.print("  Avg return: {avg_ret}", avg_ret=jnp.mean(ppo_batch['returns']))
                
                # Normalize advantages for stability (standard practice in PPO)
                # This helps with gradient stability when advantage scales vary widely
                batch_advantages = ppo_batch['advantages']
                advantage_mean = jnp.mean(batch_advantages)
                advantage_std = jnp.std(batch_advantages)
                batch_advantages_normalized = (batch_advantages - advantage_mean) / (advantage_std + 1e-8)
                ppo_batch_normalized = {**ppo_batch, 'advantages': batch_advantages_normalized}
                
                # Get KL penalty weight for current step
                kl_weight = kl_penalty_schedule(state_dict['opt_t'])
                
                # Use BC-appropriate hyperparameters if BC was used
                clip_ratio = config.get('ppo_clip_ratio', config.get('clip_ratio', 0.2))
                entropy_coef = config.get('ppo_entropy_coef', config.get('entropy_coef', 0.01))
                
                # Update policy network (actor) - OPTIMIZED: compute loss and grad together
                # Fallback BC params: if None, use current policy params (KL becomes ~0 and avoids Haiku param errors)
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
                
                # Update value network (critic) - OPTIMIZED: compute loss and grad together
                # Use original batch for value updates (not normalized advantages)
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
                
                # Store KL metrics for logging if available (use last epoch's values)
                if 'kl_div' in policy_info:
                    kl_div = float(policy_info['kl_div'])
                    kl_penalty_val = float(policy_info.get('kl_penalty', 0.0))
                
                # Debug: Print loss values
                # print(f"PPO Epoch {epoch}: Policy Loss = {policy_loss:.6f}, Value Loss = {value_loss:.6f}")
        
        # Reset environments for next iteration
        state_dict["key"], subkey = jax.random.split(state_dict["key"])
        subkeys = jax.random.split(subkey, config['batch_size'])
        starts = jnp.stack([jax_random.choice(k, env.fixed_starts) for k in subkeys])
        pickups = jnp.stack([jax_random.choice(k, env.fixed_pickups) for k in subkeys])
        
        batch_init = vmap(lambda k, s, p: init_env(k, s, p, env.neighbor_mask_static), in_axes=(0, 0, 0))
        state_dict['env_states'], _ = batch_init(subkeys, starts, pickups)
        state_dict['last_start'] = starts
        state_dict['key'] = subkey
        
        # Run evaluation on fixed set
        eval_results = evaluate_policy_on_fixed_set(state_dict['policy_params'], eval_starts, eval_pickups)
        
        # Aggregate metrics from scan results to scalars
        aggregated_metrics = {
            'policy_loss': total_policy_loss,  # Policy (actor) loss
            'value_loss': total_value_loss,    # Value (critic) loss
            'total_loss': total_policy_loss + total_value_loss,  # Combined loss
            'avg_return': jnp.mean(metrics['avg_return']),  # Average return across steps
            'num_episodes': jnp.sum(metrics['num_episodes']),  # Total episodes across steps
            'buffer_size': buffer.size,  # Current buffer size
            'buffer_utilization': buffer.size / buffer.buffer_size,  # Buffer utilization
            'learning_active': buffer.size >= min_buffer_size,  # Whether learning is active
            
            # Curriculum learning metrics
            'curriculum/alpha': float(jnp.mean(metrics['curriculum_alpha'])) if 'curriculum_alpha' in metrics else 0.0,
            'curriculum/sp_action_fraction': float(jnp.mean(metrics['sp_action_fraction'])) if 'sp_action_fraction' in metrics else 0.0,
            
            # KL penalty metrics (if BC was used)
            'kl/kl_div': kl_div if use_kl_penalty else 0.0,
            'kl/kl_penalty': kl_penalty_val if use_kl_penalty else 0.0,
            'kl/kl_weight': float(kl_penalty_schedule(state_dict['opt_t'])),
            
            # Evaluation metrics
            'eval/avg_reward': float(eval_results['avg_reward']),
            'eval/avg_steps': float(eval_results['avg_steps']),
            'eval/completion_rate': float(eval_results['completion_rate']),
            'eval/total_rewards': eval_results['total_rewards'].tolist(),  # Individual episode rewards
            'eval/total_steps': eval_results['total_steps'].tolist(),  # Individual episode steps
            'eval/completed': eval_results['completed'].tolist()  # Individual episode completion
        }
        
        return state_dict, aggregated_metrics
    
    return run_loop
