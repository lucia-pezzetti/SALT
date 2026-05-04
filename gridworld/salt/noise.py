from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

Action = int

@dataclass
class NoiseConfig:
    kind: str
    p: float
    rng_seed: int = 0

class NoiseModel:
    """
    Applies an action 'slip' with probability p:
    - with prob (1-p): intended action
    - with prob p: a random different action
    Correlation structure depends on cfg.kind.
    """
    def __init__(self, cfg: NoiseConfig, n_actions: int = 5):
        self.cfg = cfg
        self.n_actions = n_actions
        self.rng = np.random.default_rng(cfg.rng_seed)

        if cfg.kind not in ("none", "individual", "local", "global"):
            raise ValueError(f"Unknown noise kind: {cfg.kind}")

        self._global_slip: Optional[bool] = None
        self._local_choices: Dict[Tuple[int, int, int, int], int] = {}

    def reset_timestep(self):
        self._global_slip = None
        self._local_choices.clear()

    def _slip_action(self, intended: Action) -> Action:
        choices = [a for a in range(self.n_actions) if a != intended]
        return int(self.rng.choice(choices))

    def apply(self, intended: Action, *, t: int, s: Tuple[int, int], agent_id: int) -> Action:
        kind = self.cfg.kind
        p = float(self.cfg.p)
        if kind == "none" or p <= 0.0:
            return intended

        if kind == "individual":
            slip = (self.rng.random() < p)
            return self._slip_action(intended) if slip else intended

        if kind == "global":
            if self._global_slip is None:
                self._global_slip = bool(self.rng.random() < p)
            if self._global_slip:
                return self._slip_action(intended)
            return intended

        if kind == "local":
            key = (int(t), int(s[0]), int(s[1]), int(intended))
            if key not in self._local_choices:
                slip = (self.rng.random() < p)
                self._local_choices[key] = self._slip_action(intended) if slip else intended
            return int(self._local_choices[key])

        raise RuntimeError("Unreachable")

    def _vec_slip(self, intended: np.ndarray) -> np.ndarray:
        """Uniform random action != intended, fully vectorised."""
        n = len(intended)
        off = self.rng.integers(0, self.n_actions - 1, size=n).astype(np.int32)
        return off + (off >= intended).astype(np.int32)

    def apply_batch(
        self,
        intended: np.ndarray,
        *,
        t: int,
        pos_r: np.ndarray,
        pos_c: np.ndarray,
        agent_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Apply noise to a batch of agents at the same timestep.

        Parameters
        ----------
        intended   (N,) int32  intended actions
        t          current timestep (scalar)
        pos_r      (N,) int32  row positions
        pos_c      (N,) int32  col positions
        agent_ids  (N,) int     original agent indices (unused for most
                                noise types, kept for API compatibility)

        Returns
        -------
        (N,) int32 executed actions
        """
        kind = self.cfg.kind
        p = float(self.cfg.p)
        n = len(intended)

        if kind == "none" or p <= 0.0 or n == 0:
            return intended.copy()

        if kind == "individual":
            slip = self.rng.random(n) < p
            result = intended.copy()
            mask = slip
            if np.any(mask):
                result[mask] = self._vec_slip(intended[mask])
            return result

        if kind == "global":
            if self._global_slip is None:
                self._global_slip = bool(self.rng.random() < p)
            if self._global_slip:
                return self._vec_slip(intended)
            return intended.copy()

        if kind == "local":
            result = intended.copy()
            for k in range(n):
                key = (int(t), int(pos_r[k]), int(pos_c[k]), int(intended[k]))
                if key not in self._local_choices:
                    if self.rng.random() < p:
                        off = int(self.rng.integers(0, self.n_actions - 1))
                        self._local_choices[key] = off + (1 if off >= intended[k] else 0)
                    else:
                        self._local_choices[key] = int(intended[k])
                result[k] = self._local_choices[key]
            return result

        raise RuntimeError("Unreachable")
