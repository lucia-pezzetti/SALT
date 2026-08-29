#!/bin/bash
# Map four ranks on one Clariden node to four routing/reward configurations.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

rank="${SLURM_PROCID:-${ABLATION_VARIANT:-0}}"
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

export FIXED_SEED="${ABLATION_SEED:-4}"
export RUN_LABEL="${base_label}_${variant}"
export MIN_EDGE_TRAVEL_TIME_SECONDS="$min_edge_seconds"
export PICKUP_BONUS_SECONDS="$pickup_bonus_seconds"

echo "[ablation rank $rank] variant=$variant seed=$FIXED_SEED"
exec bash "$SCRIPT_DIR/run_rank.sh"
