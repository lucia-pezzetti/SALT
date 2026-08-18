#!/bin/bash
# Rank-aware launcher for the Manhattan Q-learning run.
#
# Under SLURM it runs ONE seed per task (seed = $SLURM_PROCID), so a job with
# --ntasks-per-node=4 --gpus-per-task=1 fills all 4 GH200 GPUs of a Clariden node
# with 4 independent seeds (the code is single-GPU: jax.vmap on one device).
# Outside SLURM it falls back to a sequential loop over ${SEEDS[@]} for local dev.
#
# Env knobs (all overridable):
#   EPOCHS                    training epochs for this chunk (default 5000000)
#   NUM_AGENTS                agents per episode (default 5)
#   PICKUP_BONUS              reward for reaching target (default 50)
#   EVAL_FREQUENCY            periodic evaluation cadence in episodes (default 10000)
#   CHECKPOINT_FREQUENCY      Q-table checkpoint cadence in episodes; 0=end only
#   EPISODE_OFFSET            completed episodes before this chunk (default 0)
#   TOTAL_EPOCHS_FOR_SCHEDULE total intended episodes across chunks (default EPOCHS)
#   SKIP_FINAL_EVAL=1         skip expensive final evaluation for intermediate chunks
#   INIT_FROM_SHORTEST_PATHS=0 skip shortest-path initialization
#   INIT_ALL_TIME_SLICES=0    initialize shortest-path values only at time slice 0
#   SEED_OFFSET               add this to Slurm/local ranks to get seed ranges 4-7, 8-11, ...
#   RUN_LABEL                 output filename label
#   Q_TABLE_DTYPE             Q-table storage dtype: float32, float16, or bfloat16 (default bfloat16)
#   SEEDS                     space-separated seeds for the local (non-SLURM) fallback
#   USE_CONTAINER=1           skip conda activation (python is provided by the container image)
#   CONDA_ENV                 conda env name for the non-container path (default ride-sharing)
#   ZONE_SHP                  path to taxi_zones.shp
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # -> manhattan/src

# ---- JAX / W&B configuration ----
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export JAX_ENABLE_X64=False
export JAX_ENABLE_COMPILATION_CACHE=True
export JAX_COMPILATION_CACHE_SIZE="${JAX_COMPILATION_CACHE_SIZE:-1000}"
export ENABLE_PJRT_COMPATIBILITY="${ENABLE_PJRT_COMPATIBILITY:-1}"
# Don't grab all GPU memory up front; with --gpus-per-task=1 each rank sees one GPU anyway.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Offline by default: Clariden compute nodes have no outbound internet. `wandb sync` afterwards.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-ride-sharing-optimized}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ride-sharing-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/ride-sharing-cache}"

EPOCHS="${EPOCHS:-5000000}"
NUM_AGENTS="${NUM_AGENTS:-5}"
PICKUP_BONUS="${PICKUP_BONUS:-50}"
EVAL_FREQUENCY="${EVAL_FREQUENCY:-10000}"
CHECKPOINT_FREQUENCY="${CHECKPOINT_FREQUENCY:-0}"
EPISODE_OFFSET="${EPISODE_OFFSET:-0}"
TOTAL_EPOCHS_FOR_SCHEDULE="${TOTAL_EPOCHS_FOR_SCHEDULE:-$EPOCHS}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
INIT_FROM_SHORTEST_PATHS="${INIT_FROM_SHORTEST_PATHS:-1}"
INIT_ALL_TIME_SLICES="${INIT_ALL_TIME_SLICES:-1}"
SEED_OFFSET="${SEED_OFFSET:-0}"
RUN_LABEL="${RUN_LABEL:-south_manhattan_${NUM_AGENTS}agents_5M}"
Q_TABLE_PREFIX="${Q_TABLE_PREFIX:-qlearning_${RUN_LABEL}}"
Q_TABLE_DTYPE="${Q_TABLE_DTYPE:-bfloat16}"
ZONE_SHP="${ZONE_SHP:-../data/processed/taxi_zones.shp}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
mkdir -p ./cache ./logs ./q_tables "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

run_one() {
    local SEED="$1"
    local jc="/tmp/jax_cache_parallel/seed_${SEED}"; mkdir -p "$jc"
    local wd="/tmp/ride-sharing-wandb/seed_${SEED}"; mkdir -p "$wd"
    local q_table_path="./q_tables/${Q_TABLE_PREFIX}_seed${SEED}.pkl"
    local log_file="logs/${Q_TABLE_PREFIX}_offset${EPISODE_OFFSET}_epochs${EPOCHS}_seed${SEED}.txt"
    echo "[rank ${SLURM_PROCID:-local}] seed=${SEED} epochs=${EPOCHS} agents=${NUM_AGENTS} offset=${EPISODE_OFFSET} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "  total schedule epochs=${TOTAL_EPOCHS_FOR_SCHEDULE} eval_frequency=${EVAL_FREQUENCY} checkpoint_frequency=${CHECKPOINT_FREQUENCY}"
    echo "  pickup_bonus=${PICKUP_BONUS} horizon_value=negative_shortest_path_remaining_time"
    echo "  init_from_shortest_paths=${INIT_FROM_SHORTEST_PATHS} init_all_time_slices=${INIT_ALL_TIME_SLICES}"
    echo "  q_table=${q_table_path} dtype=${Q_TABLE_DTYPE}"
    echo "  log=${log_file}"
    python -c "import jax; print('  jax devices:', jax.devices())" || true
    local cmd=(
        python main.py
        --env_type manhattan
        --zone_shp "$ZONE_SHP"
        --manhattan_area south_manhattan
        --discrete
        --dt 1.0
        --epochs "$EPOCHS"
        --noise
        --noise_level 0.1
        --no_congestion True
        --q_table_dtype "$Q_TABLE_DTYPE"
        --sample_starts_from_three_fixed
        --three_fixed_selection_method betweenness
        --no_round_trip
        --epsilon_start 1.0
        --eval_frequency "$EVAL_FREQUENCY"
        --random_offsets
        --num_agents "$NUM_AGENTS"
        --pickup_bonus "$PICKUP_BONUS"
        --cycle_length 90
        --cache_dir ./cache
        --num_workers 4
        --seed "${SEED}"
        --q_table_path "$q_table_path"
        --checkpoint_frequency "$CHECKPOINT_FREQUENCY"
        --episode_offset "$EPISODE_OFFSET"
        --total_epochs_for_schedule "$TOTAL_EPOCHS_FOR_SCHEDULE"
    )
    if [ "$INIT_FROM_SHORTEST_PATHS" != "0" ]; then
        cmd+=(--init_from_shortest_paths)
        if [ "$INIT_ALL_TIME_SLICES" != "0" ]; then
            cmd+=(--init_all_time_slices)
        fi
    fi
    if [ "$SKIP_FINAL_EVAL" != "0" ]; then
        cmd+=(--skip_final_eval)
    fi
    cmd+=(
        --eval_reassignment_baselines
        --reassignment_periods 5,10
    )
    JAX_COMPILATION_CACHE_DIR="$jc" WANDB_DIR="$wd" \
    "${cmd[@]}" > "$log_file" 2>&1
}

# ---- environment activation (skipped inside a container) ----
if [ "${USE_CONTAINER:-0}" != "1" ]; then
    CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
    if [ ! -x "$CONDA_EXE" ]; then
        echo "conda not found; set CONDA_EXE or USE_CONTAINER=1." >&2; exit 1
    fi
    eval "$("$CONDA_EXE" shell.bash hook)"
    conda activate "${CONDA_ENV:-ride-sharing}"
fi

if [ -n "${SLURM_PROCID:-}" ]; then
    # Cluster: one seed per rank. rank i -> seed i (scales across nodes too).
    run_one "$((SEED_OFFSET + SLURM_PROCID))"
else
    # Local fallback: sequential seeds.
    read -r -a _seeds <<< "${SEEDS:-0}"
    for s in "${_seeds[@]}"; do run_one "$((SEED_OFFSET + s))"; done
fi
