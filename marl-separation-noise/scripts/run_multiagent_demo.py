from __future__ import annotations
import argparse, os, json
from datetime import datetime
import numpy as np

from sepnoise.env import GridConfig, MultiAgentGrid, SingleAgentGoalGrid
from sepnoise.noise import NoiseConfig
from sepnoise.qlearning import GoalConditionedTabularQ, QConfig, train_goal_q
from sepnoise.matching import assign_goals, terminal_ot_cost
from sepnoise.viz import plot_snapshot, plot_terminal_hist

def choose_action(Q: GoalConditionedTabularQ, s, z) -> int:
    q = Q.Q[s[0], s[1], z[0], z[1], :]
    return int(np.argmin(q))

def ensure_q(grid: GridConfig, p: float, q_path: str | None, seed: int,
             save_dir: str | None = None, wb_run=None):
    if q_path and os.path.exists(q_path):
        Q = GoalConditionedTabularQ.load(q_path)
        if Q.h != grid.h or Q.w != grid.w:
            print(f"[warn] Q-table shape ({Q.h}x{Q.w}) != grid ({grid.h}x{grid.w}), retraining...")
        else:
            return Q, {"loaded_from": q_path}

    # Noise kind is irrelevant for single-agent training (same marginal slip prob)
    noise = NoiseConfig(kind="individual", p=p, rng_seed=seed)
    env = SingleAgentGoalGrid(grid, noise)
    model, metrics = train_goal_q(env, episodes=250000, qcfg=QConfig(), seed=seed,
                                  log_every=800, wb_run=wb_run)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "Q.npy")
        model.save(out)
        print(f"[info] Saved trained Q-table to: {out}")

    return model, {"trained_quick": True, "train_metrics": metrics}

def run_once(kind: str, args, grid: GridConfig, base_outdir: str,
             Q: GoalConditionedTabularQ, qinfo: dict, wb_run=None):
    noise = NoiseConfig(kind=kind, p=args.p, rng_seed=args.seed)

    env = MultiAgentGrid(grid, noise, n_agents=args.agents)
    env.reset()
    env.goals = assign_goals(env.pos, env.targets)

    snapshots_t = set(args.snapshots)
    outdir = os.path.join(base_outdir, f"{kind}")
    os.makedirs(outdir, exist_ok=True)

    total_cost = 0.0
    arrival_time = [None] * args.agents   # timestep when each agent first reaches its goal
    roll = []
    for t in range(grid.horizon):
        if args.rematch_every > 0 and t > 0 and (t % args.rematch_every == 0):
            env.goals = assign_goals(env.pos, env.targets)

        actions = [
            4 if env.reached[i] else choose_action(Q, s, z)
            for i, (s, z) in enumerate(zip(env.pos, env.goals))
        ]
        step = env.step(actions)
        total_cost += float(step["step_cost"])

        # Record arrival times for agents that just reached their goal
        for i, nr in enumerate(step["newly_reached"]):
            if nr and arrival_time[i] is None:
                arrival_time[i] = t + 1   # step t produced the move, arrival at t+1

        roll.append({
            "t": t,
            "pos": step["pos"],
            "goals": list(env.goals),
            "exec_actions": step["exec_actions"],
            "step_cost": step["step_cost"],
            "collision_cost": step["collision_cost"],
            "reached": step["reached"],
        })

        # ---- wandb per-step logging ----
        if wb_run is not None:
            n_reached_so_far = sum(1 for at in arrival_time if at is not None)
            wb_run.log({
                f"rollout/{kind}/step_cost": step["step_cost"],
                f"rollout/{kind}/cumulative_cost": total_cost,
                f"rollout/{kind}/collision_cost": step["collision_cost"],
                f"rollout/{kind}/agents_reached": n_reached_so_far,
            }, step=t)

        if t in snapshots_t:
            snap_path = os.path.join(outdir, f"rollout_t{t:02d}.png")
            plot_snapshot(
                grid.h, grid.w,
                agents=step["pos"],
                goals=list(env.goals),
                targets=env.targets,
                title=f"{kind} noise (p={args.p})  t={t}",
                outpath=snap_path,
            )
            if wb_run is not None:
                import wandb
                wb_run.log({f"rollout/{kind}/snapshot_t{t:02d}": wandb.Image(snap_path)})

    # ---- KPI computation ----
    n_reached = sum(1 for at in arrival_time if at is not None)
    reach_rate = n_reached / args.agents
    reached_times = [at for at in arrival_time if at is not None]
    mean_time_to_reach = float(np.mean(reached_times)) if reached_times else float("nan")
    term_cost = terminal_ot_cost(env.pos, env.targets)

    hist_path = os.path.join(outdir, "terminal_hist.png")
    plot_terminal_hist(
        grid.h, grid.w, agents_terminal=env.pos, targets=env.targets,
        outpath=hist_path,
        title=f"Terminal occupancy vs targets — {kind} noise (p={args.p})"
    )

    metrics = {
        "noise_kind": kind,
        "p": args.p,
        "agents": args.agents,
        "horizon": args.horizon,
        "rematch_every": args.rematch_every,
        "reach_rate": reach_rate,
        "mean_time_to_reach": mean_time_to_reach,
        "total_cost": float(total_cost),
        "terminal_ot_cost": float(term_cost),
        "arrival_times": arrival_time,
        "q_info": qinfo,
    }

    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(outdir, "rollout.json"), "w", encoding="utf-8") as f:
        json.dump(roll, f)

    # ---- wandb summary metrics for this noise kind ----
    if wb_run is not None:
        import wandb
        wb_run.log({
            f"eval/{kind}/reach_rate": reach_rate,
            f"eval/{kind}/mean_time_to_reach": mean_time_to_reach,
            f"eval/{kind}/total_cost": float(total_cost),
            f"eval/{kind}/terminal_ot_cost": float(term_cost),
            f"eval/{kind}/terminal_hist": wandb.Image(hist_path),
        })

    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_h", type=int, default=7)
    ap.add_argument("--grid_w", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--agents", type=int, default=28)
    ap.add_argument("--p", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rematch_every", type=int, default=0)
    ap.add_argument("--q_path", type=str, default=None)
    ap.add_argument("--collision_penalty", type=float, default=0.0)
    ap.add_argument("--goal_bonus", type=float, default=10.0)
    ap.add_argument("--snapshots", type=int, nargs="*", default=[0, 5, 10, 15, 24])
    ap.add_argument("--outdir", type=str, default="runs/demo")
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
            name=f"demo_p{args.p}_agents{args.agents}",
            tags=["demo"],
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_outdir = os.path.join(args.outdir, f"p{args.p}_rematch{args.rematch_every}_{stamp}")
    os.makedirs(base_outdir, exist_ok=True)

    grid = GridConfig(h=args.grid_h, w=args.grid_w, horizon=args.horizon, rng_seed=args.seed,
                      collision_penalty=args.collision_penalty, goal_bonus=args.goal_bonus)
    Q, qinfo = ensure_q(grid, args.p, args.q_path, args.seed, save_dir=base_outdir,
                         wb_run=wb_run)

    all_metrics = []
    for kind in ["individual", "local", "global"]:
        all_metrics.append(run_once(kind, args, grid, base_outdir, Q, qinfo, wb_run=wb_run))

    with open(os.path.join(base_outdir, "all_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # ---- Print summary table ----
    header = f"{'noise':<12} {'reach_rate':>10} {'mean_t_reach':>13} {'total_cost':>11} {'terminal_OT':>12}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for m in all_metrics:
        mtr = m["mean_time_to_reach"]
        mtr_str = f"{mtr:.2f}" if not (mtr != mtr) else "N/A"  # NaN check
        print(f"{m['noise_kind']:<12} {m['reach_rate']:>10.2%} {mtr_str:>13} {m['total_cost']:>11.2f} {m['terminal_ot_cost']:>12.2f}")
    print(sep)

    # ---- wandb summary table ----
    if wb_run is not None:
        import wandb
        table = wandb.Table(columns=["noise", "reach_rate", "mean_time_to_reach", "total_cost", "terminal_ot_cost"])
        for m in all_metrics:
            table.add_data(m["noise_kind"], m["reach_rate"], m["mean_time_to_reach"],
                           m["total_cost"], m["terminal_ot_cost"])
        wb_run.log({"summary_table": table})
        wb_run.finish()

    print(f"\nSaved demo outputs to: {base_outdir}")

if __name__ == "__main__":
    main()
