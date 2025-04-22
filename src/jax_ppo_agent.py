import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, NamedTuple

class PPOActor(eqx.Module):
    mlp: eqx.nn.MLP

    def __call__(self, obs):
        if obs.ndim == 1:
            logits = self.mlp(obs)
        else:
            logits = jax.vmap(self.mlp)(obs)
        return logits

class PPOCritic(eqx.Module):
    mlp: eqx.nn.MLP

    def __call__(self, obs):
        if isinstance(obs, tuple):
            obs = jnp.stack(obs, axis=1)

        if obs.ndim == 1:
            out = self.mlp(obs)
            return jnp.squeeze(out)  # scalar
        else:
            out = jax.vmap(self.mlp)(obs)
            return out.reshape(-1)  # flatten (batch_size, 1) -> (batch_size,)

class PPOAgent(eqx.Module):
    actor: PPOActor
    critic: PPOCritic

class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray

# @eqx.filter_value_and_grad
def ppo_loss(agent: PPOAgent, transitions: Transition, adv: jnp.ndarray, returns: jnp.ndarray, clip_eps: float):
    obs = transitions.obs  # (batch_size, obs_dim)

    if isinstance(obs, tuple):
        obs = jnp.stack(obs, axis=0)


    logits = agent.actor(obs)
    pi = jax.nn.softmax(logits)
    values = agent.critic(obs)

    # Ensure action is also batch-aligned
    actions = jnp.atleast_1d(transitions.action)

    log_pi = jax.nn.log_softmax(logits)  
    log_probs = log_pi[jnp.arange(log_pi.shape[0]), actions]
    
    ratios = jnp.exp(log_probs - transitions.log_prob)
    clipped_ratios = jnp.clip(ratios, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -jnp.mean(jnp.minimum(ratios * adv, clipped_ratios * adv))

    value_pred_clipped = values + jnp.clip(returns - values, -clip_eps, clip_eps)
    value_loss = 0.5 * jnp.mean(jnp.maximum(
        (returns - values) ** 2,
        (returns - value_pred_clipped) ** 2
    ))
    # value_loss = jnp.mean((returns - values) ** 2)
    entropy_bonus = -jnp.mean(jnp.sum(pi * jnp.log(pi + 1e-8), axis=-1))

    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
    return total_loss

def make_agent(key, obs_dim, act_dim, width=64):
    k1, k2 = jax.random.split(key)
    actor = PPOActor(eqx.nn.MLP(obs_dim, act_dim, width_size=width, depth=2, key=k1))
    critic = PPOCritic(eqx.nn.MLP(obs_dim, 1, width_size=width, depth=2, key=k2))
    return PPOAgent(actor=actor, critic=critic)

def sample_action(agent: PPOAgent, obs: jnp.ndarray, key: Any):
    logits = agent.actor(obs)
    action = jax.random.categorical(key, logits)
    log_pi = jax.nn.log_softmax(logits)
    log_prob = log_pi[action]
    value = agent.critic(obs)
    return action, log_prob, value
