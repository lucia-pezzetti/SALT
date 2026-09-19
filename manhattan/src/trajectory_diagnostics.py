from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class RepeatedCycleSegment:
    """A consecutive repetition of a closed directed path."""

    start_step: int
    period: int
    repetitions: int
    cycle_nodes: Tuple[int, ...]

    @property
    def end_step(self) -> int:
        """Exclusive transition index at the end of the repeated segment."""
        return self.start_step + self.period * self.repetitions


def find_repeated_cycle_segments(
    path: Sequence[int],
    *,
    min_repetitions: int = 3,
    max_period: int = 24,
    max_segments: int = 4,
) -> list[RepeatedCycleSegment]:
    """Find dominant exact repeated cycles in a node trajectory.

    Detection operates on directed transitions, so cycles longer than an
    immediate A-B-A reversal are found as well. Overlapping descriptions of the
    same run are reduced to the one covering the most transitions, with the
    shortest period preferred when coverage is equal.
    """
    if min_repetitions < 2:
        raise ValueError("min_repetitions must be at least 2")
    if max_period < 2:
        raise ValueError("max_period must be at least 2")
    if max_segments < 1:
        raise ValueError("max_segments must be positive")

    nodes = tuple(int(node) for node in path)
    edges = tuple(zip(nodes, nodes[1:]))
    candidates = []

    for start in range(len(edges)):
        max_available_period = min(max_period, (len(edges) - start) // min_repetitions)
        for period in range(2, max_available_period + 1):
            block = edges[start:start + period]
            if block[-1][1] != block[0][0]:
                continue
            if edges[start + period:start + 2 * period] != block:
                continue

            repetitions = 2
            while (
                start + (repetitions + 1) * period <= len(edges)
                and edges[
                    start + repetitions * period:
                    start + (repetitions + 1) * period
                ] == block
            ):
                repetitions += 1
            if repetitions < min_repetitions:
                continue

            cycle_nodes = (block[0][0],) + tuple(edge[1] for edge in block)
            candidates.append(
                RepeatedCycleSegment(
                    start_step=start,
                    period=period,
                    repetitions=repetitions,
                    cycle_nodes=cycle_nodes,
                )
            )

    candidates.sort(
        key=lambda segment: (
            -(segment.period * segment.repetitions),
            segment.period,
            segment.start_step,
        )
    )

    selected = []
    for candidate in candidates:
        overlaps = any(
            candidate.start_step < existing.end_step
            and existing.start_step < candidate.end_step
            for existing in selected
        )
        if overlaps:
            continue
        selected.append(candidate)
        if len(selected) == max_segments:
            break

    return sorted(selected, key=lambda segment: segment.start_step)
