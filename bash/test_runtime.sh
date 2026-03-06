#!/bin/bash -l
#SBATCH --job-name=test-conda
#SBATCH --partition=interruptible_gpu
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/users/%u/slurm_logs/test_%j.out

echo "=== Running test.sh ==="

# --- Load shared runtime (CUDA, Conda, GPU checks etc) ---
source bash/base_runtime.sh || exit 1

echo "=== Test success! ==="