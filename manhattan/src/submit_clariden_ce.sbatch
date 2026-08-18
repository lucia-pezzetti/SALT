#!/bin/bash
# Clariden submission using the CSCS container engine (recommended by CSCS).
# Same 4-GPU-per-node layout as submit_clariden.sbatch, but each rank runs inside
# the container defined by clariden.toml (no conda; USE_CONTAINER=1 set in the EDF).
#
# Prereqs (once):
#   1) Build + import the image  -> see Dockerfile.clariden
#   2) Edit clariden.toml `image` and `mounts` paths
# Submit from manhattan/src:   sbatch submit_clariden_ce.sbatch
#
#SBATCH --job-name=ride-qlearn-ce
#SBATCH --account=aa004              # <-- verify with sacctmgr
#SBATCH --partition=normal           # normal=12h; `low` for 24h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
# Container environment is supplied on each srun step below.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs cache q_tables

CE_ENV=(--environment=./clariden.toml)

# Optional cache warmup. Keep disabled for the Manhattan path: main.py currently
# disables persistent Manhattan graph/distance cache files, so this would just run
# an extra short training chunk and write to the real Q-table.
if [ "${RUN_CACHE_WARMUP:-0}" = "1" ]; then
    echo "=== optional cache warmup (single rank, in container) ==="
    EPOCHS="${WARMUP_EPOCHS:-2000}" srun "${CE_ENV[@]}" --ntasks=1 --gpus-per-task=1 --cpus-per-task=72 bash run_rank.sh
fi

# 2) Real run: 4 ranks, one GPU each, seeds 0..3.
echo "=== 4-way parallel run in container (seeds 0..3) ==="
srun "${CE_ENV[@]}" --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=72 bash run_rank.sh
