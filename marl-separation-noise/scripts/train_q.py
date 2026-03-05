from __future__ import annotations
import argparse, os, json
from datetime import datetime

from sepnoise.env import GridConfig
from sepnoise.noise import NoiseConfig
from sepnoise.qlearning import QConfig, train_goal_q_multiagent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_h", type=int, default=7)
    ap.add_argument("--grid_w", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--agents", type=int, default=7,
                    help="Number of agents M per training episode")
    ap.add_argument("--noise", type=str, default="individual", choices=["none","individual","local","global"])
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--goal_bonus", type=float, default=10.0)
    ap.add_argument("--outdir", type=str, default="runs/train")
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
            name=f"train_q_{args.noise}_p{args.p}",
            tags=["train_q", args.noise],
        )

    grid = GridConfig(h=args.grid_h, w=args.grid_w, horizon=args.horizon, rng_seed=args.seed,
                      goal_bonus=args.goal_bonus)
    noise = NoiseConfig(kind=args.noise, p=args.p, rng_seed=args.seed)

    model, metrics = train_goal_q_multiagent(
        grid, noise, n_agents=args.agents,
        episodes=args.episodes, qcfg=QConfig(), seed=args.seed,
        wb_run=wb_run,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"{args.noise}_p{args.p}_{stamp}")
    os.makedirs(outdir, exist_ok=True)

    q_path = os.path.join(outdir, "Q.npy")
    model.save(q_path)

    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "train_metrics": metrics, "q_path": q_path}, f, indent=2)

    print(f"Saved Q table to: {q_path}")

    # ---- wandb final summary ----
    if wb_run is not None:
        import wandb
        wb_run.summary["mean_cost_last_500"] = metrics["mean_cost_last_500"]
        wb_run.log({"final/mean_cost_last_500": metrics["mean_cost_last_500"]})
        wandb.save(q_path)
        wb_run.finish()

if __name__ == "__main__":
    main()
