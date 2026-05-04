from __future__ import annotations
from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

Pos = Tuple[int, int]

def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def assign_goals(agents: List[Pos], targets: List[Pos]) -> List[Pos]:
    """
    Minimum-cost 1-to-1 assignment (Hungarian) from agent positions to target cells.
    If |agents| != |targets|, we pad with dummy nodes (high cost) and fall back to nearest target.
    """
    n = len(agents)
    m = len(targets)
    size = max(n, m)
    big = 10_000

    C = np.full((size, size), big, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            C[i, j] = manhattan(agents[i], targets[j])

    row_ind, col_ind = linear_sum_assignment(C)
    row_to_col = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    assigned: List[Pos] = []
    for i in range(n):
        j = row_to_col.get(i, -1)
        if 0 <= j < m:
            assigned.append(targets[j])
        else:
            dists = [manhattan(agents[i], t) for t in targets]
            assigned.append(targets[int(np.argmin(dists))])
    return assigned

def terminal_ot_cost(agents_terminal: List[Pos], targets: List[Pos]) -> float:
    assigned = assign_goals(agents_terminal, targets)
    return float(sum(manhattan(s, z) for s, z in zip(agents_terminal, assigned)))


def target_coverage_rate(agents_terminal: List[Pos], targets: List[Pos]) -> float:
    """Fraction of *unique* target positions with ≥1 agent on them."""
    target_set = set(map(tuple, targets))
    agent_set = set(map(tuple, agents_terminal))
    if not target_set:
        return 1.0
    return len(target_set & agent_set) / len(target_set)
