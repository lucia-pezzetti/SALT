#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set JAX environment variables
# Base cache directory - each process will get its own subdirectory
export JAX_COMPILATION_CACHE_BASE=/tmp/jax_cache_parallel

# NVIDIA GPU platform selection for the discrete Q-learning run.
export JAX_ENABLE_X64=False
export JAX_ENABLE_COMPILATION_CACHE=True
export JAX_COMPILATION_CACHE_SIZE=1000
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export ENABLE_PJRT_COMPATIBILITY="${ENABLE_PJRT_COMPATIBILITY:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ride-sharing-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/ride-sharing-cache}"

# Weights & Biases configuration
export WANDB_PROJECT="${WANDB_PROJECT:-ride-sharing-optimized}"
export WANDB_MODE="${WANDB_MODE:-online}"  # Use "offline" or "disabled" without internet.
export WANDB_DIR="${WANDB_DIR:-/tmp/ride-sharing-wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/ride-sharing-wandb-cache}"

# Create base cache directory
mkdir -p $JAX_COMPILATION_CACHE_BASE
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR"

echo "=== JAX Configuration ==="
echo "JAX_COMPILATION_CACHE_BASE: $JAX_COMPILATION_CACHE_BASE"
echo "JAX_PLATFORMS: ${JAX_PLATFORMS:-auto}"
echo "JAX_COMPILATION_CACHE_SIZE: $JAX_COMPILATION_CACHE_SIZE"
echo "ENABLE_PJRT_COMPATIBILITY: $ENABLE_PJRT_COMPATIBILITY"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_MODE: $WANDB_MODE"
echo "WANDB_DIR: $WANDB_DIR"
echo "MPLCONFIGDIR: $MPLCONFIGDIR"

# Activate Python environment
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
CONDA_ENV="${CONDA_ENV:-ride-sharing}"
if [ ! -x "$CONDA_EXE" ]; then
    echo "Could not find conda."
    echo "   Set CONDA_EXE=/path/to/conda or update run_main.sh."
    exit 1
fi

eval "$("$CONDA_EXE" shell.bash hook)"
conda activate "$CONDA_ENV"

echo "=== Python Environment ==="
echo "Conda env: $CONDA_ENV"
echo "Python: $(which python)"
python - <<'PY'
import os
import jax
backend = jax.default_backend()
devices = jax.devices()
device_platforms = sorted({getattr(device, "platform", "unknown") for device in devices})
requested_platforms = [
    p.strip().lower()
    for p in os.environ.get("JAX_PLATFORMS", "").split(",")
    if p.strip()
]
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {backend}")
print(f"JAX devices: {devices}")
print(f"JAX device platforms: {device_platforms}")
if any(p in {"cuda", "gpu"} for p in requested_platforms):
    if not any(p in {"cuda", "gpu"} for p in device_platforms) and backend.lower() not in {"cuda", "gpu"}:
        raise SystemExit(
            "Expected an NVIDIA CUDA JAX backend, but JAX did not expose a CUDA/GPU device. "
            "Install the CUDA-enabled JAX package and check that the NVIDIA driver is visible. "
            "For CPU fallback, run: JAX_PLATFORMS=cpu bash src/run_main.sh"
        )
if os.environ.get("CONDA_DEFAULT_ENV") == "ride-sharing-metal" and "metal" not in backend.lower():
    raise SystemExit(
        "Expected the JAX Metal GPU backend. "
        "Use CONDA_ENV=ride-sharing JAX_PLATFORMS=cpu for CPU fallback."
    )
PY

echo "=== Starting Training ==="
echo "JAX optimizations applied:"
echo "  - Pre-compiled JAX functions for maximum speed"
echo "  - Accelerated computations on the active JAX backend"
echo "  - Reduced compilation overhead with caching"
echo "  - Enhanced batch operations with vmap"
echo "  - Automatic JAX memory management"
echo ""
echo "Performance notes:"
echo "  - First run: Slow due to JAX compilation (5-10 minutes)"
echo "  - Subsequent runs: Much faster due to caching (30-60 seconds)"
echo "  - Backend: Printed above so CPU/GPU placement is explicit"
echo "  - Metrics: Logged to Weights & Biases for real-time monitoring"
echo ""
echo "Press Ctrl+C to stop if needed."

# Create cache directory for graph and distance data
mkdir -p ./cache
mkdir -p ./logs
mkdir -p ./q_tables

# Define seeds to run (each seed will result in different starts/pickups)
SEEDS=(0)

echo "=== Running seeds sequentially ==="
echo "Seeds: ${SEEDS[@]}"
echo ""

# Run each seed/offset combination sequentially
FAILED=0
for SEED in "${SEEDS[@]}"; do
    LOG_FILE="logs/qlearning_south_manhattan_5M_epochs_10agents_seed${SEED}.txt"
    # LOG_FILE="logs/test_2zones.txt"
    echo "Running seed ${SEED} (log: ${LOG_FILE})"
    
    # Each process gets its own JAX compilation cache directory to avoid contention
    JAX_CACHE_DIR="${JAX_COMPILATION_CACHE_BASE}/seed_${SEED}"
    mkdir -p "${JAX_CACHE_DIR}"
    
    # Run foreground to keep executions sequential
    if JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_DIR}" \
    python main.py \
        --env_type manhattan \
        --zone_shp "${ZONE_SHP:-../data/processed/taxi_zones.shp}" \
        --manhattan_area south_manhattan \
        --discrete \
        --dt 1.0 \
        --epochs 5000000 \
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
        --num_agents 10 \
        --cycle_length 90 \
        --cache_dir ./cache \
        --num_workers 4 \
        --seed ${SEED} \
        --q_table_path ./q_tables/qlearning_south_manhattan_test_5M_epochs_seed0.pkl \
        --init_from_shortest_paths \
        > ${LOG_FILE} 2>&1; then
        echo "Completed seed ${SEED}"
    else
        echo "Failed seed ${SEED} (check ${LOG_FILE})"
        FAILED=$((FAILED + 1))
    fi
        # --init_q_table_path ./q_tables/q_table_manhattan_env_1zones_100ksteps_initialization_20agents_seed1_90cycle_5dt.pkl \
        # --q_table_path ./q_tables/qlearning_1zones_250kepochs_10agents_seed${SEED}_offset${OFFSET}_90cycle_5dt_eps1.pkl \
        # --init_from_shortest_paths \
        # --init_all_time_slices \
        # --all_nodes_starts_pickups \
        # --sample_starts_from_three_fixed \
        # --three_fixed_selection_method {degree, closeness, betweenness}
        # --start_zones "Upper East Side North" \
        # --pickup_zones "Yorkville West" \
        # TO ADD NOISE:
        # --noise \
        # --noise_level 0.1 \
        # TO ADD BFLOAT16:
        # --q_table_dtype bfloat16 \
    # Small delay between runs
    sleep 0.2
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All training runs completed successfully."
    echo "Outputs: Check logs/qlearning_south_manhattan_5M_epochs_10agents_seed*.txt"
    echo "Metrics: Logged to Weights & Biases for analysis"
else
    TOTAL_RUNS=${#SEEDS[@]}
    echo "${FAILED} out of ${TOTAL_RUNS} runs failed. Check log files for details."
    exit 1
fi
