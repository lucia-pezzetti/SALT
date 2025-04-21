#!/usr/bin/env bash
#SBATCH --job-name=jax_taxi_cpu
#SBATCH --output=logs/jax_taxi_cpu_%j.out
#SBATCH --error=logs/jax_taxi_cpu_%j.err
#SBATCH --time=12:00:00

# Use a CPU‐only partition
#SBATCH --partition=hw_nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

# Ensure logs directory exists
mkdir -p logs

# Activate your virtualenv
source ~/ride-sharing-venv/bin/activate

# Diagnostics
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Using Python:       $(which python)"
echo "Virtualenv at:      $VIRTUAL_ENV"
echo "CUDA_VISIBLE_DEVICES= $CUDA_VISIBLE_DEVICES"  # should be empty
echo "JAX sees devices:" 
python - <<'EOF'
import jax
print(jax.devices())
EOF

# Launch your JAX training on CPU
python -u jax_main.py