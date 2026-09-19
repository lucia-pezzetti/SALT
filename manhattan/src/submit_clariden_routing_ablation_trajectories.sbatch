#!/bin/bash
#SBATCH --job-name=mab-traj
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-ablation-traj-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs cache q_tables

echo "=== Four trajectory evaluations, one ablation per GPU ==="
srun \
    --network=disable_rdzv_get \
    --environment=./clariden.toml \
    --ntasks-per-node=4 \
    --gpus-per-task=1 \
    --cpus-per-task=72 \
    bash run_routing_ablation_trajectory_rank.sh
