#!/bin/bash
# Rank-aware launcher for the Manhattan Q-learning run.
#
# Under SLURM it runs ONE seed per task (seed = $SLURM_PROCID), so a job with
# --ntasks-per-node=4 --gpus-per-task=1 fills all 4 GH200 GPUs of a Clariden node
# with 4 independent seeds (the code is single-GPU: jax.vmap on one device).
# Outside SLURM it falls back to a sequential loop over ${SEEDS[@]} for local dev.
#
# Env knobs (all overridable):
#   EPOCHS          training epochs (default 5000000; the warmup step sets it small)
#   SEEDS           space-separated seeds for the local (non-SLURM) fallback
#   USE_CONTAINER=1 skip conda activation (python is provided by the container image)
#   CONDA_ENV       conda env name for the non-container path (default ride-sharing)
#   ZONE_SHP        path to taxi_zones.shp
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
ZONE_SHP="${ZONE_SHP:-../data/processed/taxi_zones.shp}"
mkdir -p ./cache ./logs ./q_tables "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

run_one() {
    local SEED="$1"
    local jc="/tmp/jax_cache_parallel/seed_${SEED}"; mkdir -p "$jc"
    local wd="/tmp/ride-sharing-wandb/seed_${SEED}"; mkdir -p "$wd"
    echo "[rank ${SLURM_PROCID:-local}] seed=${SEED} epochs=${EPOCHS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    python -c "import jax; print('  jax devices:', jax.devices())" || true
    JAX_COMPILATION_CACHE_DIR="$jc" WANDB_DIR="$wd" \
    python main.py \
        --env_type manhattan \
        --zone_shp "$ZONE_SHP" \
        --manhattan_area south_manhattan \
        --discrete \
        --dt 1.0 \
        --epochs "$EPOCHS" \
        --noise \
        --noise_level 0.1 \
        --no_congestion True \
        --q_table_dtype bfloat16 \
        --sample_starts_from_three_fixed \
        --three_fixed_selection_method betweenness \
        --no_round_trip \
        --epsilon_start 1.0 \
        --eval_frequency 10000 \
        --random_offsets \
        --num_agents 5 \
        --cycle_length 90 \
        --cache_dir ./cache \
        --num_workers 4 \
        --seed "${SEED}" \
        --q_table_path ./q_tables/qlearning_south_manhattan_5M_seed${SEED}.pkl \
        --init_from_shortest_paths \
        --eval_reassignment_baselines \
        --reassignment_periods 5,10 \
        > "logs/qlearning_south_manhattan_5M_seed${SEED}.txt" 2>&1
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
    run_one "${SLURM_PROCID}"
else
    # Local fallback: sequential seeds.
    read -r -a _seeds <<< "${SEEDS:-0}"
    for s in "${_seeds[@]}"; do run_one "$s"; done
fi
