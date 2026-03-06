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
# ROBERTA BASE (EVASION)
# -------------------------------

# echo "=== Training: ROBERTA Base + Stratified Splits + Pair ==="
# python scripts/train_single_task.py --config configs/experiments/task_b/fine_tuning/roberta/base/pair/stratified.yaml

# echo "=== Training: ROBERTA Base + Stratified Splits + QA Marked ==="
# python scripts/train_single_task.py --config configs/experiments/task_b/fine_tuning/roberta/base/qa_marked/stratified.yaml

# -------------------------------
# ROBERTA LARGE (EVASION)
# -------------------------------

# echo "=== Training: ROBERTA Large + Stratified Splits + QA Marked ==="
# python scripts/train_single_task.py --config configs/experiments/task_b/fine_tuning/roberta/large/qa_marked/stratified.yaml

# echo "=== Training: ROBERTA Large + President Disjoint Splits + QA Marked ==="
# python scripts/train_single_task.py --config configs/experiments/task_b/fine_tuning/roberta/large/qa_marked/president_disjoint.yaml

# echo "=== Training: ROBERTA Large + Stratified Splits + QA Marked + Balanced Weighting ==="
# python scripts/train_single_task.py --config configs/experiments/task_b/fine_tuning/roberta/large/qa_marked/balanced_weighted_stratified.yaml

# echo "=== Training: ROBERTA Large + Stratified Splits + QA Marked + Sqrt Weighting ==="
# python scripts/train_single_task.py --config configs/experiments/task_b/fine_tuning/roberta/large/qa_marked/sqrt_weighted_stratified.yaml

echo "=== ALL TRAINING COMPLETED SUCCESSFULLY ✅ ==="