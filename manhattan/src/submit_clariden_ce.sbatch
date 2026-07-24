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
#SBATCH --environment=./clariden.toml   # container engine EDF; applies to all srun steps
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs cache q_tables

# 1) Warm the shared graph/distance cache with a single rank (avoids the 4-way build race).
if ! ls cache/simple_distances_*.pkl >/dev/null 2>&1; then
    echo "=== warming graph/distance cache (single rank, in container) ==="
    EPOCHS=2000 srun --ntasks=1 --gpus-per-task=1 --cpus-per-task=72 bash run_rank.sh
fi

# 2) Real run: 4 ranks, one GPU each, seeds 0..3.
echo "=== 4-way parallel run in container (seeds 0..3) ==="
srun --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=72 bash run_rank.sh
