#!/bin/bash
# Evaluate one of the four routing ablations on each Clariden GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

rank="${SLURM_PROCID:-${ABLATION_VARIANT:-0}}"
seed="${ABLATION_SEED:-4}"
base_label="${BASE_RUN_LABEL:-south_manhattan_20agents_5M_terminalsp_alltimeinit_bfloat16_uturn_ablation}"

case "$rank" in
    0)
        variant="minedge2s_bonus0s"
        min_edge_seconds=2
        pickup_bonus_seconds=0
        ;;
    1)
        variant="minedge2s_bonus10s"
        min_edge_seconds=2
        pickup_bonus_seconds=10
        ;;
    2)
        variant="minedge0s_bonus10s"
        min_edge_seconds=0
        pickup_bonus_seconds=10
        ;;
    3)
        variant="minedge0s_bonus50s"
        min_edge_seconds=0
        pickup_bonus_seconds=50
        ;;
    *)
        echo "Expected rank/ABLATION_VARIANT in 0..3, got '$rank'." >&2
        exit 2
        ;;
esac

train_label="${base_label}_${variant}"
q_table="q_tables/qlearning_${train_label}_seed${seed}.pkl"
if [ ! -f "$q_table" ]; then
    echo "Missing Q-table: $q_table" >&2
    exit 1
fi

shopt -s nullglob
training_logs=(logs/qlearning_${train_label}_offset*_epochs*_seed${seed}.txt)
latest_episode=""
if [ "${#training_logs[@]}" -gt 0 ]; then
    latest_episode="$(awk '
        /\[checkpoint\] (periodic|final): saving Q-table at global episode/ {
            episode = $0
            sub(/^.*global episode /, "", episode)
            sub(/ .*/, "", episode)
            gsub(/,/, "", episode)
            pending = episode
            next
        }
        /^\[checkpoint\] saved / && pending != "" {
            print pending
            pending = ""
        }
    ' "${training_logs[@]}" | sort -n | tail -n 1)"
fi
if [ -z "$latest_episode" ]; then
    echo "Could not determine the latest checkpoint episode for $q_table from training logs." >&2
    exit 1
fi
if [ -n "${CHECKPOINT_EPISODE:-}" ] && [ "$latest_episode" -ne "$CHECKPOINT_EPISODE" ]; then
    echo "Expected checkpoint $CHECKPOINT_EPISODE for $q_table, but the latest successful save is $latest_episode." >&2
    exit 1
fi

export TRAIN_RUN_LABEL="$train_label"
export EVAL_RUN_LABEL="${EVAL_BASE_RUN_LABEL:-trajectory_compare}_${variant}_checkpoint${latest_episode}"
export FIXED_Q_TABLE_SEED="$seed"
export FIXED_EVAL_SEED="${EVAL_SEED:-$seed}"
export TOTAL_EPOCHS="$latest_episode"
export MIN_EDGE_TRAVEL_TIME_SECONDS="$min_edge_seconds"
export PICKUP_BONUS_SECONDS="$pickup_bonus_seconds"
export FLEET_SIZE="${FLEET_SIZE:-5}"
export EVAL_ITERATIONS="${EVAL_ITERATIONS:-10}"
export PRINT_EVAL_TRAJECTORIES=1
export PRINT_Q_CYCLE_DIAGNOSTICS="${PRINT_Q_CYCLE_DIAGNOSTICS:-1}"

echo "[trajectory rank $rank] variant=$variant train_seed=$seed eval_seed=$FIXED_EVAL_SEED checkpoint=$latest_episode"
exec bash "$SCRIPT_DIR/run_eval_fleet_rank.sh"
