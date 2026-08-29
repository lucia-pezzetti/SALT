#!/bin/bash
# Submit a dependency chain whose four ranks train four ablation variants.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ACCOUNT="${ACCOUNT:-aa004}"
PARTITION="${PARTITION:-normal}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
NUM_AGENTS="${NUM_AGENTS:-20}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5000000}"
CHUNK_EPOCHS="${CHUNK_EPOCHS:-500000}"
START_OFFSET="${START_OFFSET:-0}"
CHECKPOINT_FREQUENCY="${CHECKPOINT_FREQUENCY:-100000}"
EVAL_FREQUENCY="${EVAL_FREQUENCY:-999999999}"
BASE_RUN_LABEL="${BASE_RUN_LABEL:-south_manhattan_20agents_5M_terminalsp_alltimeinit_bfloat16_uturn_ablation}"
Q_TABLE_DTYPE="${Q_TABLE_DTYPE:-bfloat16}"
ABLATION_SEED="${ABLATION_SEED:-4}"
FINAL_EVAL="${FINAL_EVAL:-1}"
INIT_ALL_TIME_SLICES="${INIT_ALL_TIME_SLICES:-1}"
ALLOW_EXISTING_QTABLES="${ALLOW_EXISTING_QTABLES:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'USAGE'
Usage: bash submit_clariden_routing_ablation_chunks.sh [options]

Runs four configurations concurrently on the four GPUs of one Clariden node:
  rank 0: minimum edge time 2s, pickup bonus 0s
  rank 1: minimum edge time 2s, pickup bonus 10s
  rank 2: original edge times, pickup bonus 10s
  rank 3: original edge times, pickup bonus 50s

All configurations use the same seed and the existing U-turn masks.

Options:
  --account ACCOUNT              CSCS account (default: aa004)
  --partition PARTITION          Slurm partition (default: normal)
  --time HH:MM:SS                Time limit per chunk (default: 12:00:00)
  --seed N                       Shared seed for all four variants (default: 4)
  --num-agents N                 Agents per episode (default: 20)
  --total-epochs N               Total intended episodes (default: 5000000)
  --chunk-epochs N               Episodes per chunk (default: 500000)
  --start-offset N               Resume at this global episode (default: 0)
  --checkpoint-frequency N       Save every N episodes (default: 100000)
  --eval-frequency N             Periodic evaluation cadence (default: 999999999)
  --base-run-label LABEL         Prefix shared by the four output labels
  --q-table-dtype DTYPE          float32, float16, or bfloat16 (default: bfloat16)
  --time0-init-only              Initialize shortest paths only at time slice 0
  --no-final-eval                Skip the final EPOCHS=0 evaluation job
  --allow-existing-qtables       Permit offset-0 reuse of matching Q-table paths
  --dry-run                      Print commands without submitting
  -h, --help                     Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --account) ACCOUNT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --seed) ABLATION_SEED="$2"; shift 2 ;;
        --num-agents) NUM_AGENTS="$2"; shift 2 ;;
        --total-epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
        --chunk-epochs) CHUNK_EPOCHS="$2"; shift 2 ;;
        --start-offset) START_OFFSET="$2"; shift 2 ;;
        --checkpoint-frequency) CHECKPOINT_FREQUENCY="$2"; shift 2 ;;
        --eval-frequency) EVAL_FREQUENCY="$2"; shift 2 ;;
        --base-run-label) BASE_RUN_LABEL="$2"; shift 2 ;;
        --q-table-dtype) Q_TABLE_DTYPE="$2"; shift 2 ;;
        --time0-init-only) INIT_ALL_TIME_SLICES=0; shift ;;
        --no-final-eval) FINAL_EVAL=0; shift ;;
        --allow-existing-qtables) ALLOW_EXISTING_QTABLES=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value_name in TOTAL_EPOCHS CHUNK_EPOCHS START_OFFSET CHECKPOINT_FREQUENCY NUM_AGENTS ABLATION_SEED; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$value_name must be a non-negative integer, got '$value'." >&2
        exit 2
    fi
done
if [ "$TOTAL_EPOCHS" -le 0 ] || [ "$CHUNK_EPOCHS" -le 0 ]; then
    echo "TOTAL_EPOCHS and CHUNK_EPOCHS must be positive." >&2
    exit 2
fi
if [ "$START_OFFSET" -gt "$TOTAL_EPOCHS" ]; then
    echo "START_OFFSET cannot exceed TOTAL_EPOCHS." >&2
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
if [ "$START_OFFSET" -eq 0 ] && [ "$ALLOW_EXISTING_QTABLES" != "1" ]; then
    existing=()
    for variant in "${variants[@]}"; do
        path="q_tables/qlearning_${BASE_RUN_LABEL}_${variant}_seed${ABLATION_SEED}.pkl"
        [ ! -e "$path" ] || existing+=("$path")
    done
    if [ "${#existing[@]}" -gt 0 ]; then
        echo "Refusing offset-0 submission because matching Q-tables exist:" >&2
        printf '  %s\n' "${existing[@]}" >&2
        echo "Choose a new --base-run-label or remove the files." >&2
        exit 1
    fi
fi

submit_one() {
    local offset="$1"
    local epochs="$2"
    local skip_final_eval="$3"
    local init_sp="$4"
    local dependency="${5:-}"
    local job_name="$6"

    local export_vars
    export_vars="ALL,NUM_AGENTS=${NUM_AGENTS},EPOCHS=${epochs},TOTAL_EPOCHS_FOR_SCHEDULE=${TOTAL_EPOCHS},EPISODE_OFFSET=${offset},CHECKPOINT_FREQUENCY=${CHECKPOINT_FREQUENCY},EVAL_FREQUENCY=${EVAL_FREQUENCY},SKIP_FINAL_EVAL=${skip_final_eval},BASE_RUN_LABEL=${BASE_RUN_LABEL},Q_TABLE_DTYPE=${Q_TABLE_DTYPE},INIT_FROM_SHORTEST_PATHS=${init_sp},INIT_ALL_TIME_SLICES=${INIT_ALL_TIME_SLICES},ABLATION_SEED=${ABLATION_SEED}"

    local cmd=(
        sbatch --parsable
        -A "$ACCOUNT"
        -p "$PARTITION"
        --time="$TIME_LIMIT"
        --job-name="$job_name"
        --export="$export_vars"
    )
    [ -z "$dependency" ] || cmd+=(--dependency="afterok:${dependency}")
    cmd+=(submit_clariden_routing_ablation.sbatch)

    if [ "$DRY_RUN" = "1" ]; then
        printf '[dry-run]' >&2
        printf ' %q' "${cmd[@]}" >&2
        printf '\n' >&2
        echo "DRYRUN-$offset"
    else
        local submitted
        if ! submitted="$("${cmd[@]}")"; then
            echo "Failed to submit ablation chunk at offset ${offset}." >&2
            exit 1
        fi
        echo "${submitted%%;*}"
    fi
}

echo "Submitting four-variant Manhattan ablation chain:"
echo "  account=$ACCOUNT partition=$PARTITION time=$TIME_LIMIT"
echo "  seed=$ABLATION_SEED agents=$NUM_AGENTS dtype=$Q_TABLE_DTYPE"
echo "  base_run_label=$BASE_RUN_LABEL"
echo "  total_epochs=$TOTAL_EPOCHS chunk_epochs=$CHUNK_EPOCHS start_offset=$START_OFFSET"
printf '  variant=%s\n' "${variants[@]}"

prev_job=""
offset="$START_OFFSET"
submitted_jobs=()
while [ "$offset" -lt "$TOTAL_EPOCHS" ]; do
    remaining=$((TOTAL_EPOCHS - offset))
    epochs="$CHUNK_EPOCHS"
    [ "$remaining" -ge "$epochs" ] || epochs="$remaining"

    init_sp=0
    [ "$offset" -ne 0 ] || init_sp=1
    job_id="$(submit_one "$offset" "$epochs" 1 "$init_sp" "$prev_job" "mab_${offset}")"
    echo "  chunk offset=$offset epochs=$epochs -> job $job_id"
    submitted_jobs+=("$job_id")
    prev_job="$job_id"
    offset=$((offset + epochs))
done

if [ "$FINAL_EVAL" = "1" ]; then
    job_id="$(submit_one "$TOTAL_EPOCHS" 0 0 0 "$prev_job" "mab_final")"
    echo "  final evaluation offset=$TOTAL_EPOCHS -> job $job_id"
    submitted_jobs+=("$job_id")
fi

echo
echo "Submitted jobs:"
printf '  %s\n' "${submitted_jobs[@]}"
echo "Monitor with: squeue -u \"\$USER\""
