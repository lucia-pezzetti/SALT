#!/bin/bash
# Clariden (CSCS) submission: fill ALL 4 GH200 GPUs of one node with 4 independent
# seeds in parallel. The code is single-GPU, and Clariden nodes are exclusive, so
# this is the way to avoid idle GPUs (a job array of 1-GPU jobs would grab a whole
# 4-GPU node each -> 12 idle GPUs).
#
# Submit from manhattan/src:   sbatch submit_clariden.sbatch
# Uses the conda / uenv environment path (run_rank.sh activates conda).
#
#SBATCH --job-name=ride-qlearn
#SBATCH --account=aa004              # <-- verify: `sacctmgr show assoc user=$USER format=account`
#SBATCH --partition=normal           # normal=12h; use `low` for 24h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 4 ranks -> 4 GPUs
#SBATCH --gpus-per-task=1            # 1 GH200 per rank; SLURM sets CUDA_VISIBLE_DEVICES=SLURM_LOCALID
#SBATCH --cpus-per-task=72           # one full Grace CPU (72 cores) per rank
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs cache q_tables

# Optional cache warmup. Keep disabled for the Manhattan path: main.py currently
# disables persistent Manhattan graph/distance cache files, so this would just run
# an extra short training chunk and write to the real Q-table.
if [ "${RUN_CACHE_WARMUP:-0}" = "1" ]; then
    echo "=== optional cache warmup (single rank) ==="
    EPOCHS="${WARMUP_EPOCHS:-2000}" srun --ntasks=1 --gpus-per-task=1 --cpus-per-task=72 bash run_rank.sh
fi

# Real run: 4 ranks, one GPU each, seeds 0..3 in parallel.
echo "=== 4-way parallel run (seeds 0..3) ==="
srun --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=72 bash run_rank.sh
