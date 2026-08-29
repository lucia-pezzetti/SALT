#!/bin/bash
#SBATCH --job-name=manhattan-ablation
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-ablation-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs cache q_tables

echo "=== Four Manhattan ablations on one node, one GPU per variant ==="
srun \
    --network=disable_rdzv_get \
    --environment=./clariden.toml \
    --ntasks-per-node=4 \
    --gpus-per-task=1 \
    --cpus-per-task=72 \
    bash run_routing_ablation_rank.sh
