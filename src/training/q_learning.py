"""
Tabular Q-Learning implementation for discrete state-action spaces.
Used when args.discrete=True to discretize time with dt=5.
"""
import jax
import jax.numpy as jnp
from jax import random as jax_random
from typing import Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

from taxi_env import TaxiState, TaxiEnv, init_env


def discretize_time(time: float, dt: float = 5.0) -> int:
    """Discretize time to the nearest multiple of dt."""
    return int(round(time / dt))


def discretize_travel_time(travel_time: float, dt: float = 5.0) -> float:
    """Discretize travel time to the nearest multiple of dt."""
    return dt * round(travel_time / dt)


def state_to_discrete_key(state: TaxiState, dt: float = 5.0) -> Tuple[int, int, int]:
    """
    Convert continuous state to discrete state key.
    Returns: (current_node, pickup_node, discrete_time)
    """
    discrete_time = discretize_time(float(state.time), dt)
    return (int(state.current_node), int(state.pickup_node), discrete_time)


class TabularQLearning:
    """
    Tabular Q-Learning agent for discrete state-action spaces.
    """
    
    def __init__(
        self,
        env: TaxiEnv,
        dt: float = 5.0,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10000,
        initial_q_value: float = 0.0,
    ):
        self.env = env
        self.dt = dt
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.initial_q_value = initial_q_value
        
        # Q-table: Dict[(current_node, pickup_node, discrete_time, action)] -> Q-value
        self.q_table: Dict[Tuple[int, int, int, int], float] = defaultdict(lambda: initial_q_value)
        
        # Track visits for statistics
        self.visit_counts: Dict[Tuple[int, int, int, int], int] = defaultdict(int)
        
        # Discretize travel times in environment
        self._discretize_travel_times()
    
    def _discretize_travel_times(self):
        """Discretize all travel times in the environment to multiples of dt."""
        # Create a discretized copy of travel_times
        self.discrete_travel_times = jnp.round(self.env.travel_times / self.dt) * self.dt
    
    def get_epsilon(self, step: int) -> float:
        """Get epsilon value for epsilon-greedy policy at current step."""
        if step >= self.epsilon_decay_steps:
            return self.epsilon_end
        progress = step / self.epsilon_decay_steps
        return self.epsilon_start * (1.0 - progress) + self.epsilon_end * progress
    
    def get_q_value(self, state: TaxiState, action: int) -> float:
        """Get Q-value for a state-action pair."""
        state_key = state_to_discrete_key(state, self.dt)
        q_key = (*state_key, action)
        return self.q_table[q_key]
    
    def set_q_value(self, state: TaxiState, action: int, value: float):
        """Set Q-value for a state-action pair."""
        state_key = state_to_discrete_key(state, self.dt)
        q_key = (*state_key, action)
        self.q_table[q_key] = value
        self.visit_counts[q_key] += 1
    
    def get_action(self, state: TaxiState, step: int, key: jnp.ndarray) -> int:
        """
        Epsilon-greedy action selection.
        
        Args:
            state: Current state
            step: Current training step (for epsilon decay)
            key: JAX random key
            
        Returns:
            Selected action
        """
        epsilon = self.get_epsilon(step)
        state_key = state_to_discrete_key(state, self.dt)
        
        # Get Q-values for all valid actions
        valid_actions = []
        q_values = []
        for action in range(self.env.max_deg):
            if state.neighbor_mask[action]:
                q_key = (*state_key, action)
                valid_actions.append(action)
                q_values.append(self.q_table[q_key])
        
        if len(valid_actions) == 0:
            return 0  # Fallback
        
        # Epsilon-greedy
        key, subkey = jax_random.split(key)
        if jax_random.uniform(subkey) < epsilon:
            # Random action
            key, subkey = jax_random.split(key)
            action_idx = jax_random.randint(subkey, (), 0, len(valid_actions))
            return valid_actions[int(action_idx)]
        else:
            # Greedy action
            best_idx = np.argmax(q_values)
            return valid_actions[best_idx]
    
    def update_q_value(
        self,
        state: TaxiState,
        action: int,
        reward: float,
        next_state: TaxiState,
        done: bool
    ):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s, a) = Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        current_q = self.get_q_value(state, action)
        
        if done:
            # Terminal state: no future value
            target = reward
        else:
            # Get max Q-value for next state
            max_next_q = float('-inf')
            for next_action in range(self.env.max_deg):
                if next_state.neighbor_mask[next_action]:
                    next_q = self.get_q_value(next_state, next_action)
                    max_next_q = max(max_next_q, next_q)
            
            if max_next_q == float('-inf'):
                max_next_q = 0.0  # No valid actions
            
            target = reward + self.gamma * max_next_q
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (target - current_q)
        self.set_q_value(state, action, new_q)
    
    def step_with_discretization(self, state: TaxiState, action: int, key: jnp.ndarray) -> Tuple[TaxiState, float, bool, dict]:
        """
        Environment step with discretized travel times.
        This replaces the normal env.step() when using discrete mode.
        """
        # Already done
        already_done = state.done
        
        # Base move
        curr = state.current_node
        nxt = self.env.adj_list[curr, action]
        
        # Use discretized travel time
        travel = float(self.discrete_travel_times[curr, action])
        
        # Check for invalid moves
        invalid = (nxt == -1)
        reach = (curr == state.pickup_node) | (nxt == state.pickup_node)
        step_n = state.step_count + 1
        done = reach
        
        # Time calculations with discretization
        t1 = state.time + travel
        
        # Signal phase & wait - match environment's traffic light logic
        period_edge = float(self.env.periods[curr, action])
        offset_edge = float(self.env.offsets[curr, action])
        green_edge = float(self.env.green_durations[curr, action])
        
        cycle = (t1 + offset_edge) % period_edge
        wait = 0.0 if cycle < green_edge else period_edge - cycle  # Match environment's wait calculation
        t2 = t1 + wait
        
        # Discretize final time
        discrete_t2 = discretize_travel_time(t2, self.dt)
        norm_time = discrete_t2 % period_edge
        
        # Reward calculation (same as original)
        total_delay = travel + wait
        dist_curr = float(self.env.distances[curr, state.pickup_node])
        dist_next = float(self.env.distances[nxt, state.pickup_node])
        dist_diff = dist_curr - dist_next
        dist_shaping = dist_diff / 60.0
        
        pickup_bonus = self.env.pickup_bonus if reach else 0.0
        reward_scaling = 1.0
        # Include distance shaping to guide agent toward pickup
        reward = -reward_scaling * total_delay / 60.0 + pickup_bonus + dist_shaping
        reward = 0.0 if already_done else reward
        
        # Neighbor mask
        nm = self.env.neighbor_mask_static[nxt]
        
        # State update
        at_pickup_before = (curr == state.pickup_node)
        reached_pickup_this_step = reach & ~already_done
        final_node = jnp.where(
            already_done,
            curr,
            jnp.where(
                reached_pickup_this_step,
                jnp.where(at_pickup_before, curr, nxt),
                nxt
            )
        )
        final_neighbor_mask = jnp.where(already_done, state.neighbor_mask, nm)
        final_time = jnp.where(already_done, state.time, jnp.array(norm_time, dtype=jnp.float32))
        final_step_count = jnp.where(already_done, state.step_count, jnp.where(done, 0, step_n))
        final_done = jnp.where(already_done, True, done)
        
        final_wait = jnp.where(already_done, 0.0, wait)
        final_travel = jnp.where(already_done, 0.0, travel)
        
        new_state = TaxiState(
            current_node=final_node,
            pickup_node=state.pickup_node,
            done=final_done,
            step_count=final_step_count,
            neighbor_mask=final_neighbor_mask,
            time=final_time
        )
        
        return new_state, reward, final_done, {"wait": final_wait, "travel": final_travel}
    
    def get_statistics(self) -> Dict:
        """Get statistics about the Q-table."""
        q_values = list(self.q_table.values())
        visit_counts = list(self.visit_counts.values())
        
        return {
            'num_states': len(self.q_table),
            'avg_q_value': np.mean(q_values) if q_values else 0.0,
            'min_q_value': np.min(q_values) if q_values else 0.0,
            'max_q_value': np.max(q_values) if q_values else 0.0,
            'avg_visits': np.mean(visit_counts) if visit_counts else 0.0,
            'max_visits': np.max(visit_counts) if visit_counts else 0.0,
        }

