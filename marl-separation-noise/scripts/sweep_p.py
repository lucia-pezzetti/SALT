"""
Sweep over noise probabilities p and compare the three noise models.

For each p the script:
  1. trains (or loads) one shared Q-table,
  2. runs multi-agent rollouts for individual / local / global noise,
  3. repeats over several seeds for statistical robustness,
  4. saves a comparison plot  +  a JSON with all results.

Example
-------
python scripts/sweep_p.py --grid_h 10 --grid_w 15 --agents 5 --horizon 30 \
    --p_values 0.0 0.05 0.1 0.15 0.2 0.3 --n_seeds 10
"""
from __future__ import annotations
import argparse, os, json
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from sepnoise.env import GridConfig, MultiAgentGrid, SingleAgentGoalGrid
from sepnoise.noise import NoiseConfig
from sepnoise.qlearning import GoalConditionedTabularQ, QConfig, train_goal_q
from sepnoise.matching import assign_goals, terminal_ot_cost

NOISE_KINDS = ["individual", "local", "global"]
NOISE_COLORS = {"individual": "#1f77b4", "local": "#ff7f0e", "global": "#2ca02c"}


# ── helpers ──────────────────────────────────────────────────────────────────

def choose_action(Q: GoalConditionedTabularQ, s, z) -> int:
    q = Q.Q[s[0], s[1], z[0], z[1], :]
    return int(np.argmin(q))


def train_q(grid: GridConfig, p: float, seed: int, episodes: int,
            wb_run=None, wb_prefix: str = "train") -> GoalConditionedTabularQ:
    noise = NoiseConfig(kind="individual", p=p, rng_seed=seed)
    env = SingleAgentGoalGrid(grid, noise)
    model, _ = train_goal_q(env, episodes=episodes, qcfg=QConfig(), seed=seed,
                            log_every=5000, wb_run=wb_run, wb_prefix=wb_prefix)
    return model


def rollout(grid: GridConfig, noise_kind: str, p: float, n_agents: int,
            Q: GoalConditionedTabularQ, seed: int) -> Dict[str, Any]:
    """Run one multi-agent rollout and return the four KPIs."""
    noise = NoiseConfig(kind=noise_kind, p=p, rng_seed=seed)
    env = MultiAgentGrid(grid, noise, n_agents=n_agents)
    env.reset()
    env.goals = assign_goals(env.pos, env.targets)

    total_cost = 0.0
    arrival_time = [None] * n_agents

    for t in range(grid.horizon):
        actions = [
            4 if env.reached[i] else choose_action(Q, s, z)
            for i, (s, z) in enumerate(zip(env.pos, env.goals))
        ]
        step = env.step(actions)
        total_cost += float(step["step_cost"])

        for i, nr in enumerate(step["newly_reached"]):
            if nr and arrival_time[i] is None:
                arrival_time[i] = t + 1

    n_reached = sum(1 for at in arrival_time if at is not None)
    reached_times = [at for at in arrival_time if at is not None]

    return {
        "reach_rate": n_reached / n_agents,
        "mean_time_to_reach": float(np.mean(reached_times)) if reached_times else float("nan"),
        "total_cost": float(total_cost),
        "terminal_ot_cost": float(terminal_ot_cost(env.pos, env.targets)),
    }


# ── main sweep ───────────────────────────────────────────────────────────────

def run_sweep(args, wb_run=None) -> Dict[str, Any]:
    p_values = sorted(args.p_values)
    seeds = list(range(args.n_seeds))

    # results[noise_kind][p] = list of KPI dicts (one per seed)
    results: Dict[str, Dict[float, List[Dict]]] = {k: {} for k in NOISE_KINDS}

    for p in p_values:
        print(f"\n{'='*60}")
        print(f"  p = {p}")
        print(f"{'='*60}")

        # Train one Q-table per p (use seed=0 for training)
        grid = GridConfig(
            h=args.grid_h, w=args.grid_w, horizon=args.horizon,
            rng_seed=0, collision_penalty=args.collision_penalty,
            goal_bonus=args.goal_bonus,
        )
        Q = train_q(grid, p, seed=0, episodes=args.train_episodes,
                    wb_run=wb_run, wb_prefix=f"train/p{p}")

        for kind in NOISE_KINDS:
            kpi_list = []
            for s in seeds:
                # Use a different grid seed per rollout so agents start differently
                grid_run = GridConfig(
                    h=args.grid_h, w=args.grid_w, horizon=args.horizon,
                    rng_seed=s, collision_penalty=args.collision_penalty,
                    goal_bonus=args.goal_bonus,
                )
                kpi = rollout(grid_run, kind, p, args.agents, Q, seed=s)
                kpi_list.append(kpi)
            results[kind][p] = kpi_list
            # quick per-kind summary
            rr = np.mean([k["reach_rate"] for k in kpi_list])
            mt = np.nanmean([k["mean_time_to_reach"] for k in kpi_list])
            tc = np.mean([k["total_cost"] for k in kpi_list])
            ot = np.mean([k["terminal_ot_cost"] for k in kpi_list])
            print(f"  {kind:<12}  reach={rr:.2%}  mean_t={mt:.1f}  cost={tc:.1f}  OT={ot:.1f}")

            # ---- wandb per-(p, kind) logging ----
            if wb_run is not None:
                wb_run.log({
                    f"sweep/{kind}/reach_rate": float(rr),
                    f"sweep/{kind}/mean_time_to_reach": float(mt),
                    f"sweep/{kind}/total_cost": float(tc),
                    f"sweep/{kind}/terminal_ot_cost": float(ot),
                    "sweep/p": p,
                })

    return {"p_values": p_values, "seeds": seeds, "results": results}


# ── plotting ─────────────────────────────────────────────────────────────────

KPI_LABELS = {
    "reach_rate":        ("Reach rate",           True),   # (label, higher_is_better)
    "mean_time_to_reach":("Mean time to reach",   False),
    "total_cost":        ("Total cost",            False),
    "terminal_ot_cost":  ("Terminal OT cost",      False),
}


def make_plot(sweep: Dict[str, Any], outpath: str):
    p_values = sweep["p_values"]
    results = sweep["results"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for ax, (kpi_key, (label, _)) in zip(axes, KPI_LABELS.items()):
        for kind in NOISE_KINDS:
            means, stds = [], []
            for p in p_values:
                vals = [d[kpi_key] for d in results[kind][p]]
                # filter NaN for mean_time_to_reach when no agent reached
                valid = [v for v in vals if not (v != v)]
                if valid:
                    means.append(np.mean(valid))
                    stds.append(np.std(valid))
                else:
                    means.append(float("nan"))
                    stds.append(0.0)

            means_arr = np.array(means)
            stds_arr = np.array(stds)
            ax.plot(p_values, means_arr, "o-", label=kind, color=NOISE_COLORS[kind])
            ax.fill_between(
                p_values,
                means_arr - stds_arr,
                means_arr + stds_arr,
                alpha=0.18, color=NOISE_COLORS[kind],
            )

        ax.set_ylabel(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Noise probability  p")
    axes[3].set_xlabel("Noise probability  p")
    fig.suptitle("Multi-agent performance vs noise probability", fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"\nPlot saved to: {outpath}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Sweep noise probability p and compare noise models")
    ap.add_argument("--grid_h", type=int, default=10)
    ap.add_argument("--grid_w", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--p_values", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
    ap.add_argument("--n_seeds", type=int, default=10, help="Number of rollout seeds per (p, noise) pair")
    ap.add_argument("--train_episodes", type=int, default=50000, help="Q-learning episodes per p")
    ap.add_argument("--collision_penalty", type=float, default=0.0)
    ap.add_argument("--goal_bonus", type=float, default=10.0)
    ap.add_argument("--outdir", type=str, default="runs/sweep")
    # wandb
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="marl-separation-noise",
                    help="W&B project name")
    ap.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team/user)")
    args = ap.parse_args()

    # ---- Optional wandb init ----
    wb_run = None
    if args.wandb:
        import wandb
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"sweep_p_{datetime.now().strftime('%m%d_%H%M')}",
            tags=["sweep"],
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"sweep_{stamp}")
    os.makedirs(outdir, exist_ok=True)

    sweep = run_sweep(args, wb_run=wb_run)

    # Save raw results (convert float keys to strings for JSON)
    json_results = {}
    for kind in NOISE_KINDS:
        json_results[kind] = {str(p): v for p, v in sweep["results"][kind].items()}
    with open(os.path.join(outdir, "sweep_results.json"), "w") as f:
        json.dump({
            "args": vars(args),
            "p_values": sweep["p_values"],
            "n_seeds": len(sweep["seeds"]),
            "results": json_results,
        }, f, indent=2)

    plot_path = os.path.join(outdir, "sweep_comparison.png")
    make_plot(sweep, plot_path)

    # Print final summary table
    print(f"\n{'='*75}")
    header = f"{'p':>6}  {'noise':<12} {'reach_rate':>10} {'mean_t':>8} {'cost':>9} {'OT':>8}"
    print(header)
    print("-" * 75)
    for p in sweep["p_values"]:
        for kind in NOISE_KINDS:
            kpis = sweep["results"][kind][p]
            rr = np.mean([k["reach_rate"] for k in kpis])
            mt = np.nanmean([k["mean_time_to_reach"] for k in kpis])
            tc = np.mean([k["total_cost"] for k in kpis])
            ot = np.mean([k["terminal_ot_cost"] for k in kpis])
            mt_s = f"{mt:.2f}" if not np.isnan(mt) else "N/A"
            print(f"{p:>6.2f}  {kind:<12} {rr:>10.2%} {mt_s:>8} {tc:>9.2f} {ot:>8.2f}")
        print()
    print(f"Results saved to: {outdir}")

    # ---- wandb final artifacts ----
    if wb_run is not None:
        import wandb

        # Log the comparison plot
        wb_run.log({"sweep_comparison": wandb.Image(plot_path)})

        # Log a summary wandb.Table with averaged KPIs
        table = wandb.Table(columns=["p", "noise", "reach_rate", "mean_time_to_reach",
                                      "total_cost", "terminal_ot_cost"])
        for p in sweep["p_values"]:
            for kind in NOISE_KINDS:
                kpis = sweep["results"][kind][p]
                table.add_data(
                    p, kind,
                    float(np.mean([k["reach_rate"] for k in kpis])),
                    float(np.nanmean([k["mean_time_to_reach"] for k in kpis])),
                    float(np.mean([k["total_cost"] for k in kpis])),
                    float(np.mean([k["terminal_ot_cost"] for k in kpis])),
                )
        wb_run.log({"sweep_summary": table})

        # Save result JSON as artifact
        artifact = wandb.Artifact(f"sweep_results_{stamp}", type="results")
        artifact.add_file(os.path.join(outdir, "sweep_results.json"))
        artifact.add_file(plot_path)
        wb_run.log_artifact(artifact)

        wb_run.finish()


if __name__ == "__main__":
    main()
