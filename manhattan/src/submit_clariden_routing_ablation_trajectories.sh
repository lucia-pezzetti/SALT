#!/bin/bash
# Submit one four-GPU trajectory-comparison job for each requested training seed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ACCOUNT="${ACCOUNT:-aa004}"
PARTITION="${PARTITION:-normal}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SEEDS="${SEEDS:-4}"
EVAL_SEED="${EVAL_SEED:-}"
FLEET_SIZE="${FLEET_SIZE:-5}"
EVAL_ITERATIONS="${EVAL_ITERATIONS:-10}"
BASE_RUN_LABEL="${BASE_RUN_LABEL:-south_manhattan_20agents_5M_terminalsp_alltimeinit_bfloat16_uturn_ablation}"
EVAL_BASE_RUN_LABEL="${EVAL_BASE_RUN_LABEL:-trajectory_compare}"
Q_TABLE_DTYPE="${Q_TABLE_DTYPE:-bfloat16}"
CHECKPOINT_EPISODE="${CHECKPOINT_EPISODE:-}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'USAGE'
Usage: bash submit_clariden_routing_ablation_trajectories.sh [options]

For every requested seed, submits one node running these four evaluations:
  minedge2s_bonus0s, minedge2s_bonus10s,
  minedge0s_bonus10s, minedge0s_bonus50s.

Options:
  --seeds "4 5 6 7"          Trained Q-table seeds (default: "4")
  --eval-seed N              Use the same evaluation seed for every training seed
  --fleet-size N             Agents per trajectory instance (default: 5)
  --eval-iterations N        Instances printed per training variant (default: 10)
  --base-run-label LABEL     Prefix used by the four trained Q-tables
  --eval-run-label LABEL     Prefix used by trajectory logs
  --q-table-dtype DTYPE      float32, float16, or bfloat16 (default: bfloat16)
  --checkpoint-episode N     Require every Q-table to be saved at episode N
  --account ACCOUNT          CSCS account (default: aa004)
  --partition PARTITION      Slurm partition (default: normal)
  --time HH:MM:SS            Time limit for each seed job (default: 02:00:00)
  --dry-run                  Validate files and print sbatch commands
  -h, --help                 Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --seeds) SEEDS="$2"; shift 2 ;;
        --eval-seed) EVAL_SEED="$2"; shift 2 ;;
        --fleet-size) FLEET_SIZE="$2"; shift 2 ;;
        --eval-iterations) EVAL_ITERATIONS="$2"; shift 2 ;;
        --base-run-label) BASE_RUN_LABEL="$2"; shift 2 ;;
        --eval-run-label) EVAL_BASE_RUN_LABEL="$2"; shift 2 ;;
        --q-table-dtype) Q_TABLE_DTYPE="$2"; shift 2 ;;
        --checkpoint-episode) CHECKPOINT_EPISODE="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value_name in FLEET_SIZE EVAL_ITERATIONS; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer, got '$value'." >&2
        exit 2
    fi
done
if [ -n "$EVAL_SEED" ] && ! [[ "$EVAL_SEED" =~ ^[0-9]+$ ]]; then
    echo "EVAL_SEED must be a non-negative integer." >&2
    exit 2
fi
if [ -n "$CHECKPOINT_EPISODE" ] && ! [[ "$CHECKPOINT_EPISODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "CHECKPOINT_EPISODE must be a positive integer." >&2
    exit 2
fi
case "$Q_TABLE_DTYPE" in
    float32|float16|bfloat16) ;;
    *) echo "Invalid Q_TABLE_DTYPE '$Q_TABLE_DTYPE'." >&2; exit 2 ;;
esac

variants=(
    minedge2s_bonus0s
    minedge2s_bonus10s
    minedge0s_bonus10s
    minedge0s_bonus50s
)

mkdir -p logs cache q_tables
shopt -s nullglob

latest_checkpoint_episode() {
    local train_label="$1"
    local seed="$2"
    local training_logs=(logs/qlearning_${train_label}_offset*_epochs*_seed${seed}.txt)
    if [ "${#training_logs[@]}" -eq 0 ]; then
        return 1
    fi
    local latest
    latest="$(awk '
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
    [ -n "$latest" ] || return 1
    printf '%s\n' "$latest"
}

validated_seeds=()
for seed in $SEEDS; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
        echo "Each seed must be a non-negative integer, got '$seed'." >&2
        exit 2
    fi

    missing=()
    for variant in "${variants[@]}"; do
        train_label="${BASE_RUN_LABEL}_${variant}"
        path="q_tables/qlearning_${train_label}_seed${seed}.pkl"
        [ -f "$path" ] || missing+=("$path")
        if [ -f "$path" ]; then
            if ! latest_episode="$(latest_checkpoint_episode "$train_label" "$seed")"; then
                echo "Cannot determine the latest checkpoint for $path from logs." >&2
                exit 1
            fi
            if [ -n "$CHECKPOINT_EPISODE" ] && [ "$latest_episode" -ne "$CHECKPOINT_EPISODE" ]; then
                echo "Expected checkpoint $CHECKPOINT_EPISODE for $path, but the latest successful save is $latest_episode." >&2
                exit 1
            fi
            echo "seed=$seed variant=$variant latest_checkpoint=$latest_episode"
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "Cannot evaluate seed $seed; missing Q-tables:" >&2
        printf '  %s\n' "${missing[@]}" >&2
        exit 1
    fi
    validated_seeds+=("$seed")
done

submitted_jobs=()
for seed in "${validated_seeds[@]}"; do
    export_vars="ALL,ABLATION_SEED=${seed},FLEET_SIZE=${FLEET_SIZE},EVAL_ITERATIONS=${EVAL_ITERATIONS},BASE_RUN_LABEL=${BASE_RUN_LABEL},EVAL_BASE_RUN_LABEL=${EVAL_BASE_RUN_LABEL},Q_TABLE_DTYPE=${Q_TABLE_DTYPE}"
    [ -z "$EVAL_SEED" ] || export_vars+=",EVAL_SEED=${EVAL_SEED}"
    [ -z "$CHECKPOINT_EPISODE" ] || export_vars+=",CHECKPOINT_EPISODE=${CHECKPOINT_EPISODE}"
    cmd=(
        sbatch --parsable
        -A "$ACCOUNT"
        -p "$PARTITION"
        --time="$TIME_LIMIT"
        --job-name="mab-traj-s${seed}"
        --export="$export_vars"
        submit_clariden_routing_ablation_trajectories.sbatch
    )

    if [ "$DRY_RUN" = "1" ]; then
        printf '[dry-run]'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        job_id="DRYRUN-seed${seed}"
    else
        job_id="$("${cmd[@]}")"
        job_id="${job_id%%;*}"
    fi
    echo "seed=$seed variants=4 -> job $job_id"
    submitted_jobs+=("$job_id")
done

echo
echo "Submitted jobs:"
printf '  %s\n' "${submitted_jobs[@]}"
echo "Monitor with: squeue -u \"\$USER\" -o \"%.12i %.18j %.10T %.10M %.10L %.8D %R\""
