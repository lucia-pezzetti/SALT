from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

from .noise import NoiseModel, NoiseConfig, Action

Pos = Tuple[int, int]

@dataclass
class GridConfig:
    h: int = 7
    w: int = 12
    horizon: int = 25
    step_cost: float = 1.0
    collision_penalty: float = 0.0
    goal_bonus: float = 10.0
    rng_seed: int = 0

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
    4: (0, 0),   # stay
}

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

class SingleAgentGoalGrid:
    """
    Single-agent goal-conditioned grid used for learning Q(s,z,a).
    Goal is a cell z=(row, last_col).
    When the agent reaches its goal it enters an absorbing state:
      - receives a one-time bonus (negative cost = -goal_bonus)
      - stays still for all remaining timesteps (no action, no noise, 0 cost)
    If the agent never reaches the goal, a terminal Manhattan-distance penalty
    is added on the last step.
    """
    def __init__(self, grid: GridConfig, noise: NoiseConfig):
        self.grid = grid
        self.noise_model = NoiseModel(noise, n_actions=len(ACTIONS))
        self.rng = np.random.default_rng(grid.rng_seed)
        self.t = 0
        self.s: Pos = (0, 0)
        self.z: Pos = (0, grid.w - 1)
        self.reached_goal = False

    def reset(self, *, start: Optional[Pos] = None, goal: Optional[Pos] = None) -> Tuple[Pos, Pos, int]:
        self.t = 0
        self.reached_goal = False
        self.s = (int(self.rng.integers(0, self.grid.h)), 0) if start is None else start
        self.z = (int(self.rng.integers(0, self.grid.h)), self.grid.w - 1) if goal is None else goal
        return self.s, self.z, self.t

    def step(self, a: Action) -> Tuple[Pos, float, bool, Dict]:
        if self.reached_goal:
            self.t += 1
            done = (self.t >= self.grid.horizon)
            return self.s, 0.0, done, {"a_exec": 4, "reached": True}

        self.noise_model.reset_timestep()
        a_exec = self.noise_model.apply(a, t=self.t, s=self.s, agent_id=0)

        dr, dc = ACTIONS[int(a_exec)]
        r = clamp(self.s[0] + dr, 0, self.grid.h - 1)
        c = clamp(self.s[1] + dc, 0, self.grid.w - 1)
        self.s = (r, c)
        self.t += 1

        done = (self.t >= self.grid.horizon)

        if self.s == self.z:
            self.reached_goal = True
            cost = -float(self.grid.goal_bonus)
            return self.s, cost, done, {"a_exec": int(a_exec), "reached": True}

        cost = float(self.grid.step_cost)
        if done:
            cost += abs(self.s[0] - self.z[0]) + abs(self.s[1] - self.z[1])
        return self.s, cost, done, {"a_exec": int(a_exec), "reached": False}

class MultiAgentGrid:
    """
    Multi-agent rollout using a shared goal-conditioned policy.
    Goals are assigned externally (e.g., via OT/assignment).
    Agents that reach their assigned goal enter an absorbing state:
      - one-time bonus (negative cost), then frozen (no action, noise, or cost).
    """
    def __init__(self, grid: GridConfig, noise: NoiseConfig, n_agents: int, targets: Optional[List[Pos]] = None):
        self.grid = grid
        self.noise_model = NoiseModel(noise, n_actions=len(ACTIONS))
        self.rng = np.random.default_rng(grid.rng_seed)
        self.n_agents = n_agents
        self.t = 0
        self.pos: List[Pos] = []
        self.goals: List[Pos] = []
        self.reached: List[bool] = [False] * n_agents
        self.targets = targets if targets is not None else [(r, grid.w - 1) for r in range(grid.h)]

    def reset(self, *, start_rows: Optional[List[int]] = None):
        self.t = 0
        self.noise_model.reset_timestep()
        self.reached = [False] * self.n_agents
        if start_rows is None:
            start_rows = [int(self.rng.integers(0, self.grid.h)) for _ in range(self.n_agents)]
        self.pos = [(int(r), 0) for r in start_rows]
        self.goals = [(0, self.grid.w - 1) for _ in range(self.n_agents)]

    def step(self, actions: List[Action]) -> Dict:
        assert len(actions) == self.n_agents
        self.noise_model.reset_timestep()

        next_pos: List[Pos] = []
        exec_actions: List[int] = []
        newly_reached: List[bool] = []
        n_active = 0

        for i, (s, a) in enumerate(zip(self.pos, actions)):
            if self.reached[i]:
                next_pos.append(s)
                exec_actions.append(4)
                newly_reached.append(False)
                continue

            n_active += 1
            a_exec = self.noise_model.apply(int(a), t=self.t, s=s, agent_id=i)
            exec_actions.append(int(a_exec))
            dr, dc = ACTIONS[int(a_exec)]
            r = clamp(s[0] + dr, 0, self.grid.h - 1)
            c = clamp(s[1] + dc, 0, self.grid.w - 1)
            next_pos.append((r, c))

            if (r, c) == tuple(self.goals[i]):
                self.reached[i] = True
                newly_reached.append(True)
            else:
                newly_reached.append(False)

        collision_cost = 0.0
        if self.grid.collision_penalty > 0.0:
            counts: Dict[Pos, int] = {}
            for s in next_pos:
                counts[s] = counts.get(s, 0) + 1
            for _, k in counts.items():
                if k >= 2:
                    collision_cost += self.grid.collision_penalty * (k - 1)

        self.pos = next_pos
        self.t += 1
        done = (self.t >= self.grid.horizon)

        n_just_reached = sum(newly_reached)
        n_still_moving = n_active - n_just_reached
        # Reached agents receive the one-time bonus and become cost-free thereafter.
        step_cost = (float(self.grid.step_cost) * n_still_moving
                     - float(self.grid.goal_bonus) * n_just_reached
                     + float(collision_cost))

        return {
            "t": self.t,
            "pos": list(self.pos),
            "exec_actions": exec_actions,
            "step_cost": step_cost,
            "done": done,
            "collision_cost": float(collision_cost),
            "newly_reached": newly_reached,
            "reached": list(self.reached),
        }
