from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

Pos = Tuple[int, int]

def plot_snapshot(h: int, w: int, agents: List[Pos], goals: List[Pos], targets: List[Pos], title: str, outpath: str):
    fig = plt.figure(figsize=(max(6, w/1.6), max(4, h/1.6)))
    ax = plt.gca()
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    ax.scatter([p[1] for p in targets], [p[0] for p in targets], marker="*", s=220, label="targets")
    ax.scatter([p[1] for p in agents], [p[0] for p in agents], s=60, label="agents")

    for s, g in zip(agents, goals):
        ax.plot([s[1], g[1]], [s[0], g[0]], linewidth=1.0, alpha=0.6)

    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)

def plot_terminal_hist(h: int, w: int, agents_terminal: List[Pos], targets: List[Pos], outpath: str, title: str):
    last_col = w - 1
    counts = np.zeros(h, dtype=int)
    for r, c in agents_terminal:
        if c == last_col:
            counts[r] += 1

    target_counts = np.zeros(h, dtype=int)
    for r, c in targets:
        if c == last_col:
            target_counts[r] += 1

    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()
    xs = np.arange(h)
    ax.bar(xs - 0.2, target_counts, width=0.4, label="targets")
    ax.bar(xs + 0.2, counts, width=0.4, label="agents (terminal)")
    ax.set_xticks(xs)
    ax.set_xlabel("row")
    ax.set_ylabel("count at last column")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_eval_trajectories(
    h: int,
    w: int,
    targets: List[Pos],
    trajectories_by_algo: Dict[str, List[List[Pos]]],
    outpath: str,
    title: str,
    alpha: float = 0.5,
    line_width: float = 2.8,
):
    """
    Overlay agent trajectories for multiple algorithms on the same grid.
    """
    fig = plt.figure(figsize=(max(6, w / 1.6), max(4, h / 1.6)))
    ax = plt.gca()
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    if targets:
        ax.scatter(
            [p[1] for p in targets],
            [p[0] for p in targets],
            marker="*",
            s=220,
            c="black",
            label="targets",
            zorder=6,
        )

    cmap = plt.get_cmap("tab10")
    for i, (algo, trajs) in enumerate(trajectories_by_algo.items()):
        color = cmap(i % 10)
        label_used = False
        for traj in trajs:
            if not traj:
                continue
            xs = [p[1] for p in traj]
            ys = [p[0] for p in traj]
            ax.plot(
                xs,
                ys,
                color=color,
                alpha=alpha,
                linewidth=line_width,
                label=algo if not label_used else None,
            )
            ax.scatter(xs[0], ys[0], color=color, s=12, alpha=min(1.0, alpha * 2.2))
            ax.scatter(xs[-1], ys[-1], color=color, s=20, alpha=min(1.0, alpha * 3.0))
            label_used = True

    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)
