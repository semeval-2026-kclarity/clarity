#!/bin/bash -l
#SBATCH --job-name=taskB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G
#SBATCH --time=01:30:00

# --- Load shared runtime (CUDA, Conda, GPU checks etc) ---
source bash/base_runtime.sh || exit 1

# -------------------------------
# TRAINING RUNS BEGIN
# Uncomment the runs you wish to execute and update resource/time limits accordingly.
# -------------------------------

# -------------------------------
# DEBERTA V3 BASE (EVASION)
# -------------------------------

# echo "=== Training: DEBERTA V3 Base + Stratified Splits + QA Marked ==="
# python scripts/train_single_task.py --config configs/experiments/task_b/fine_tuning/deberta_v3/base/qa_marked/stratified.yaml

# -------------------------------
# DEBERTA V3 LARGE (EVASION)
# -------------------------------

# echo "=== Training: DEBERTA V3 Large + Stratified Splits + QA Marked ==="
# python scripts/train_single_task.py --config configs/experiments/task_b/fine_tuning/deberta_v3/large/qa_marked/stratified.yaml

echo "=== ALL TRAINING COMPLETED SUCCESSFULLY ✅ ==="