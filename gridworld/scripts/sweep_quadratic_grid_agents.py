#!/usr/bin/env python3
"""
Run repeated comparisons while varying square-grid size and/or number of agents.

This launcher mirrors `scripts/sweep_grid_agents.py` for training budget,
early-stop settings, and reporting, but enforces evaluation on:
  - square grids (`grid_w == grid_h`)
  - starts at bottom-left for all agents
  - targets sampled uniformly across the full grid

When `--constant_target_density` is set, square side length is computed from
the number of agents so target density over grid area stays approximately fixed.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def _parse_int_values(values: List[str]) -> List[int]:
    parsed: List[int] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.append(int(item))
    if not parsed:
        raise ValueError("At least one integer value must be provided.")
    return parsed


def _build_pairs(mode: str, sides: List[int], agents: List[int]) -> List[Tuple[int, int]]:
    if mode == "cartesian":
        return [(s, n) for s in sides for n in agents]
    if mode == "zip":
        if len(sides) != len(agents):
            raise ValueError(
                "For --mode zip, --grid_w_values and --agents_values must have the same length."
            )
        return list(zip(sides, agents))
    raise ValueError(f"Unknown mode: {mode}")


def _grid_side_from_agents_for_constant_target_density(
    agents: int,
    ref_agents: int,
    ref_grid_side: int,
) -> int:
    """
    Compute square-grid side that approximately preserves target density.

    With N targets (N == number of agents), preserving density on a square grid
    means N / side^2 ~= constant.
    """
    if agents <= 0 or ref_agents <= 0 or ref_grid_side <= 0:
        raise ValueError("agents, ref_agents, and ref_grid_side must be positive.")

    side = math.sqrt((agents * (ref_grid_side ** 2)) / ref_agents)
    return max(1, int(round(side)))


def _batches_for_target_interactions(
    target_interactions: int,
    batch_episodes: int,
    agents: int,
    horizon: int,
) -> Tuple[int, int]:
    per_batch = int(batch_episodes * agents * horizon)
    if per_batch <= 0:
        raise ValueError("per-batch interactions must be positive.")
    batches = max(1, int(round(target_interactions / per_batch)))
    effective = int(batches * per_batch)
    return batches, effective


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep run_comparison.py over square-grid size and number of agents."
    )
    ap.add_argument(
        "--grid_w_values",
        type=str,
        nargs="+",
        default=["10"],
        help="Square-grid side lengths to sweep (e.g. 20 30 40).",
    )
    ap.add_argument(
        "--agents_values",
        type=str,
        nargs="+",
        default=["2", "4", "6", "8"],
        help="Agent counts to sweep (e.g. 8 12 16, or 8,16,32).",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["cartesian", "zip"],
        default="cartesian",
        help="How to combine grid_w_values and agents_values.",
    )
    ap.add_argument("--p", type=str, default="0.1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval_seeds", type=int, default=50)
    ap.add_argument("--goal_bonus", type=float, default=0.0)
    ap.add_argument("--collision_penalty", type=float, default=0.0)
    ap.add_argument(
        "--target_interactions",
        type=int,
        default=2_000_000_000,
        help="Target total environment interactions per run (approximate).",
    )
    ap.add_argument("--sep_ppo_batch_eps", type=int, default=64)
    ap.add_argument("--mappo_batch_eps", type=int, default=64)
    ap.add_argument("--mappo", dest="run_mappo", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Train/evaluate MAPPO in each run (default on; --no-mappo skips it, "
                         "e.g. to sweep only Sep-PPO vs MFQ vs MFQ-local).")
    ap.add_argument(
        "--run_mfq",
        action="store_true",
        help="Also train/evaluate the Mean-Field Q-learning (MFQ) baseline in each run.",
    )
    ap.add_argument("--mfq_batch_eps", type=int, default=64)
    ap.add_argument(
        "--sep_ppo_early_stop_patience",
        type=int,
        default=40,
        help="Forwarded early-stop patience for Sep-PPO (0 disables).",
    )
    ap.add_argument(
        "--sep_ppo_early_stop_min_rel_update",
        type=float,
        default=0.0,
        help="Forwarded min relative update for Sep-PPO early stopping (0 disables).",
    )
    ap.add_argument(
        "--sep_ppo_early_stop_plateau_window",
        type=int,
        default=40,
        help="Forwarded plateau window (batches) for Sep-PPO hybrid early stop.",
    )
    ap.add_argument(
        "--sep_ppo_early_stop_max_delta_reach",
        type=float,
        default=0.005,
        help="Forwarded max reach-rate moving-average delta for Sep-PPO plateau.",
    )
    ap.add_argument(
        "--sep_ppo_early_stop_max_delta_reward",
        type=float,
        default=0.10,
        help="Forwarded max mean-reward moving-average delta for Sep-PPO plateau.",
    )
    ap.add_argument(
        "--mappo_early_stop_patience",
        type=int,
        default=40,
        help="Forwarded early-stop patience for MAPPO (0 disables).",
    )
    ap.add_argument(
        "--mappo_early_stop_min_rel_update",
        type=float,
        default=0.0,
        help="Forwarded min relative update for MAPPO early stopping (0 disables).",
    )
    ap.add_argument(
        "--mappo_early_stop_plateau_window",
        type=int,
        default=40,
        help="Forwarded plateau window (batches) for MAPPO hybrid early stop.",
    )
    ap.add_argument(
        "--mappo_early_stop_max_delta_reach",
        type=float,
        default=0.05,
        help="Forwarded max reach-rate moving-average delta for MAPPO plateau.",
    )
    ap.add_argument(
        "--mappo_early_stop_max_delta_reward",
        type=float,
        default=0.25,
        help="Forwarded max mean-reward moving-average delta for MAPPO plateau.",
    )
    ap.add_argument(
        "--early_stop_reach_target",
        type=float,
        default=0.99,
        help="Stop a method once training reach rate is sustained >= this target "
             "(applies to MAPPO, Sep-PPO, MFQ). 0 disables. Safe: only fires at target.",
    )
    ap.add_argument(
        "--early_stop_min_reach",
        type=float,
        default=0.5,
        help="MFQ / MFQ-local plateau stop only fires once the reach-rate moving "
             "average is >= this floor, so it can't false-stop during early low-reach "
             "exploration. 0 disables the guard.",
    )
    ap.add_argument(
        "--mfq_early_stop_patience",
        type=int,
        default=40,
        help="MFQ plateau early-stop patience in batches (0 disables the plateau stop).",
    )
    ap.add_argument(
        "--mfq_early_stop_plateau_window",
        type=int,
        default=40,
        help="MFQ plateau/target window in batches.",
    )
    ap.add_argument(
        "--mfq_early_stop_max_delta_reach",
        type=float,
        default=0.05,
        help="MFQ max reach-rate moving-average delta for the plateau stop.",
    )
    ap.add_argument(
        "--mfq_early_stop_max_delta_reward",
        type=float,
        default=0.1,
        help="MFQ max mean-reward moving-average delta for the plateau stop "
             "(<=0 uses a reach-rate-only plateau, robust to reward scale).",
    )
    ap.add_argument(
        "--mfq_explore_decay_cap_batches",
        type=int,
        default=5000,
        help="Decay the MFQ tau/eps exploration schedule to its floor over this many "
             "batches (default 5000), so scalability runs share the same schedule "
             "regardless of the (interaction-matched) budget. 0 = full budget.",
    )
    ap.add_argument(
        "--run_mfq_local",
        action="store_true",
        help="Also train/evaluate the faithful MFQ-local baseline (N-independent "
             "density-map obs + local mean field) in each run.",
    )
    ap.add_argument("--mfq_local_batch_eps", type=int, default=64)
    ap.add_argument("--mfq_local_obs_mode", type=str, default="fov",
                    choices=["density_map", "fov"],
                    help="MFQ-local target encoding: 'density_map' (K*K histogram) or "
                         "'fov' ((2R+1)^2 presence window). Both N-independent.")
    ap.add_argument("--mfq_local_density_bins", type=int, default=5,
                    help="MFQ-local density_map resolution K (obs_dim = 5 + K*K).")
    ap.add_argument("--mfq_local_fov_radius", type=int, default=3,
                    help="MFQ-local fov window half-width R (obs_dim = 5 + (2R+1)^2).")
    ap.add_argument("--mfq_local_mf_radius", type=int, default=3,
                    help="MFQ-local mean-field neighbourhood radius (Chebyshev, grid cells).")
    # Defaults mirror --mfq_* so MFQ-local stops on plateau like the other
    # baselines instead of grinding the full interaction budget (a plateau below
    # the 0.99 target would otherwise never early-stop).
    ap.add_argument("--mfq_local_early_stop_patience", type=int, default=40,
                    help="MFQ-local plateau early-stop patience in batches (0 disables).")
    ap.add_argument("--mfq_local_early_stop_plateau_window", type=int, default=40,
                    help="MFQ-local plateau/target window in batches.")
    ap.add_argument("--mfq_local_early_stop_max_delta_reach", type=float, default=0.05,
                    help="MFQ-local max reach-rate moving-average delta for the plateau stop.")
    ap.add_argument("--mfq_local_early_stop_max_delta_reward", type=float, default=0.25,
                    help="MFQ-local max mean-reward moving-average delta for the plateau stop "
                         "(<=0 uses a reach-rate-only plateau, robust to reward scale).")
    ap.add_argument("--mfq_local_explore_decay_cap_batches", type=int, default=5000,
                    help="Decay the MFQ-local tau/eps exploration schedule to its floor over "
                         "this many batches (default 5000). 0 = full budget.")
    ap.add_argument(
        "--reach_thresholds",
        type=str,
        default="0.7,0.8,0.9,0.95,0.99",
        help="Comma-separated reach-rate thresholds for stable milestone reporting.",
    )
    ap.add_argument(
        "--train_print_every_batches",
        type=int,
        default=500,
        help="Forwarded console print cadence in training batches (larger = less frequent).",
    )
    ap.add_argument(
        "--stability_window",
        type=int,
        default=10,
        help="Consecutive training batches required above threshold for stability.",
    )
    ap.add_argument("--outdir", type=str, default="runs/scalability")
    ap.add_argument(
        "--print_eval_traces",
        action="store_true",
        help=(
            "Forward --print_eval_traces to run_comparison.py to print starts, targets, "
            "trajectories, and per-step action traces during evaluation."
        ),
    )
    ap.add_argument(
        "--print_eval_max_seeds",
        type=int,
        default=1,
        help="Forward --print_eval_max_seeds to run_comparison.py when trace printing is enabled.",
    )
    ap.add_argument(
        "--constant_target_density",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compute square-grid side from each agents value to preserve "
            "target density over full grid area."
        ),
    )
    ap.add_argument(
        "--density_ref_agents",
        type=int,
        default=2,
        help=(
            "Reference agent count for --constant_target_density. "
            "Defaults to the first value in --agents_values."
        ),
    )
    ap.add_argument(
        "--density_ref_grid_w",
        type=int,
        default=10,
        help=(
            "Reference square-grid side for --constant_target_density. "
            "Defaults to the first value in --grid_w_values."
        ),
    )
    ap.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    args = ap.parse_args()
    args.grid_w_values = _parse_int_values(args.grid_w_values)
    args.agents_values = _parse_int_values(args.agents_values)

    run_comparison = Path("scripts/run_comparison.py")
    if not run_comparison.exists():
        run_comparison = Path(__file__).parent / "run_comparison.py"
    if args.constant_target_density:
        ref_agents = (
            args.density_ref_agents
            if args.density_ref_agents is not None
            else args.agents_values[0]
        )
        ref_grid_side = (
            args.density_ref_grid_w
            if args.density_ref_grid_w is not None
            else args.grid_w_values[0]
        )
        pairs = [
            (
                _grid_side_from_agents_for_constant_target_density(
                    agents=n,
                    ref_agents=ref_agents,
                    ref_grid_side=ref_grid_side,
                ),
                n,
            )
            for n in args.agents_values
        ]
        print(
            "[density] preserving target density over square area with "
            f"reference agents={ref_agents}, grid_side={ref_grid_side}"
        )
    else:
        pairs = _build_pairs(args.mode, args.grid_w_values, args.agents_values)

    total = len(pairs)
    for i, (grid_side, agents) in enumerate(pairs, start=1):
        grid_w = grid_side
        grid_h = grid_side
        horizon = 2 * grid_w
        sep_ppo_batches, sep_ppo_interactions = _batches_for_target_interactions(
            args.target_interactions, args.sep_ppo_batch_eps, agents, horizon
        )
        mappo_batches, mappo_interactions = _batches_for_target_interactions(
            args.target_interactions, args.mappo_batch_eps, agents, horizon
        )
        mfq_batches, mfq_interactions = _batches_for_target_interactions(
            args.target_interactions, args.mfq_batch_eps, agents, horizon
        )
        mfq_local_batches, mfq_local_interactions = _batches_for_target_interactions(
            args.target_interactions, args.mfq_local_batch_eps, agents, horizon
        )
        cmd = [
            sys.executable,
            str(run_comparison),
            "--sep_method",
            "ppo",
            "--grid_h",
            str(grid_h),
            "--grid_w",
            str(grid_w),
            "--horizon",
            str(horizon),
            "--agents",
            str(agents),
            "--p",
            args.p,
            "--noise_mode",
            "individual",
            "--seed",
            str(args.seed),
            "--eval_seeds",
            str(args.eval_seeds),
            "--goal_bonus",
            str(args.goal_bonus),
            "--collision_penalty",
            str(args.collision_penalty),
            "--eval_start_mode",
            "bottom_left",
            # Drives both training and evaluation target sampling: the
            # scalability experiment uses the whole grid.
            "--eval_target_region",
            "full_grid",
            "--outdir",
            args.outdir,
            "--sep_ppo_batches",
            str(sep_ppo_batches),
            "--sep_ppo_batch_eps",
            str(args.sep_ppo_batch_eps),
            "--sep_ppo_early_stop_patience",
            str(args.sep_ppo_early_stop_patience),
            "--sep_ppo_early_stop_min_rel_update",
            str(args.sep_ppo_early_stop_min_rel_update),
            "--sep_ppo_early_stop_plateau_window",
            str(args.sep_ppo_early_stop_plateau_window),
            "--sep_ppo_early_stop_max_delta_reach",
            str(args.sep_ppo_early_stop_max_delta_reach),
            "--sep_ppo_early_stop_max_delta_reward",
            str(args.sep_ppo_early_stop_max_delta_reward),
            "--mappo_batches",
            str(mappo_batches),
            "--mappo_batch_eps",
            str(args.mappo_batch_eps),
            "--mappo_early_stop_patience",
            str(args.mappo_early_stop_patience),
            "--mappo_early_stop_min_rel_update",
            str(args.mappo_early_stop_min_rel_update),
            "--mappo_early_stop_plateau_window",
            str(args.mappo_early_stop_plateau_window),
            "--mappo_early_stop_max_delta_reach",
            str(args.mappo_early_stop_max_delta_reach),
            "--mappo_early_stop_max_delta_reward",
            str(args.mappo_early_stop_max_delta_reward),
            "--early_stop_reach_target",
            str(args.early_stop_reach_target),
            "--early_stop_min_reach",
            str(args.early_stop_min_reach),
            "--reach_thresholds",
            args.reach_thresholds,
            "--train_print_every_batches",
            str(args.train_print_every_batches),
            "--stability_window",
            str(args.stability_window),
        ]
        if args.run_mfq:
            cmd.extend(
                [
                    "--run_mfq",
                    "--mfq_batches",
                    str(mfq_batches),
                    "--mfq_batch_eps",
                    str(args.mfq_batch_eps),
                    "--mfq_early_stop_patience",
                    str(args.mfq_early_stop_patience),
                    "--mfq_early_stop_plateau_window",
                    str(args.mfq_early_stop_plateau_window),
                    "--mfq_early_stop_max_delta_reach",
                    str(args.mfq_early_stop_max_delta_reach),
                    "--mfq_early_stop_max_delta_reward",
                    str(args.mfq_early_stop_max_delta_reward),
                    "--mfq_explore_decay_cap_batches",
                    str(args.mfq_explore_decay_cap_batches),
                ]
            )
        if args.run_mfq_local:
            cmd.extend(
                [
                    "--run_mfq_local",
                    "--mfq_local_batches",
                    str(mfq_local_batches),
                    "--mfq_local_batch_eps",
                    str(args.mfq_local_batch_eps),
                    "--mfq_local_obs_mode",
                    str(args.mfq_local_obs_mode),
                    "--mfq_local_density_bins",
                    str(args.mfq_local_density_bins),
                    "--mfq_local_fov_radius",
                    str(args.mfq_local_fov_radius),
                    "--mfq_local_mf_radius",
                    str(args.mfq_local_mf_radius),
                    "--mfq_local_early_stop_patience",
                    str(args.mfq_local_early_stop_patience),
                    "--mfq_local_early_stop_plateau_window",
                    str(args.mfq_local_early_stop_plateau_window),
                    "--mfq_local_early_stop_max_delta_reach",
                    str(args.mfq_local_early_stop_max_delta_reach),
                    "--mfq_local_early_stop_max_delta_reward",
                    str(args.mfq_local_early_stop_max_delta_reward),
                    "--mfq_local_explore_decay_cap_batches",
                    str(args.mfq_local_explore_decay_cap_batches),
                ]
            )
        if not args.run_mappo:
            cmd.append("--no-mappo")
        if args.wandb:
            cmd.append("--wandb")
        if args.print_eval_traces:
            cmd.extend(
                [
                    "--print_eval_traces",
                    "--print_eval_max_seeds",
                    str(args.print_eval_max_seeds),
                ]
            )

        print(
            f"\n[{i}/{total}] grid={grid_h}x{grid_w}, agents={agents}, horizon={horizon} | "
            f"target={args.target_interactions/1e6:.1f}M interactions"
        )
        print(
            f"  Sep-PPO: batches={sep_ppo_batches}, batch_eps={args.sep_ppo_batch_eps}, "
            f"effective={sep_ppo_interactions/1e6:.1f}M"
        )
        if args.run_mappo:
            print(
                f"  MAPPO  : batches={mappo_batches}, batch_eps={args.mappo_batch_eps}, "
                f"effective={mappo_interactions/1e6:.1f}M"
            )
        if args.run_mfq:
            print(
                f"  MFQ    : batches={mfq_batches}, batch_eps={args.mfq_batch_eps}, "
                f"effective={mfq_interactions/1e6:.1f}M"
            )
        if args.run_mfq_local:
            print(
                f"  MFQ-loc: batches={mfq_local_batches}, batch_eps={args.mfq_local_batch_eps}, "
                f"effective={mfq_local_interactions/1e6:.1f}M"
            )
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
