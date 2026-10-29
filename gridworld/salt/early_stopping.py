"""Shared early-stopping helpers for training loops."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ReachTargetStopper:
    """Stop only after a reach-rate target is sustained for a full window.

    A positive target selects threshold-only early stopping. In that mode a
    flat training curve below the target must not terminate the run.
    """

    target: float = 0.0
    window_batches: int = 10
    consecutive_batches: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.target <= 1.0:
            raise ValueError("reach target must be in [0, 1]")
        if self.window_batches < 1:
            raise ValueError("reach-target window must be at least 1 batch")

    @property
    def enabled(self) -> bool:
        return self.target > 0.0

    @property
    def allows_plateau_stop(self) -> bool:
        """Plateau stopping is a fallback only when no target is configured."""
        return not self.enabled

    def update(self, reach_rate: float) -> bool:
        if not self.enabled:
            self.consecutive_batches = 0
            return False
        if math.isfinite(reach_rate) and reach_rate >= self.target:
            self.consecutive_batches += 1
        else:
            self.consecutive_batches = 0
        return self.consecutive_batches >= self.window_batches

    def reason(self) -> str:
        return (
            f"reach rate >= {self.target:.0%} for "
            f"{self.consecutive_batches} consecutive batches"
        )
