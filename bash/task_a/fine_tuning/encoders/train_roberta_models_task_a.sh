#!/bin/bash -l
#SBATCH --job-name=taskA
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
# ROBERTA BASE (DIRECT CLARITY)
# -------------------------------

# echo "=== Training: ROBERTA Base + Stratified Splits + Pair ==="
# python scripts/train_single_task.py --config configs/experiments/task_a/fine_tuning/roberta/base/pair/stratified.yaml

# echo "=== Training: ROBERTA Base + Stratified Splits + QA Marked ==="
# python scripts/train_single_task.py --config configs/experiments/task_a/fine_tuning/roberta/base/qa_marked/stratified.yaml

# echo "=== Training: ROBERTA Base + Stratified Splits + QA Marked + Naive Masked ==="
# python scripts/train_single_task.py --config configs/experiments/task_a/fine_tuning/roberta/base/qa_marked/stratified_naive_masked.yaml

# echo "=== Training: ROBERTA Base + Stratified Splits + QA Marked + Aware Masked ==="
# python scripts/train_single_task.py --config configs/experiments/task_a/fine_tuning/roberta/base/qa_marked/stratified_aware_masked.yaml


# -------------------------------
# ROBERTA LARGE (DIRECT CLARITY)
# -------------------------------

# echo "=== Training: ROBERTA Large + Stratified Splits + Pair ==="
# python scripts/train_single_task.py --config configs/experiments/task_a/fine_tuning/roberta/large/pair/stratified.yaml

# echo "=== Training: ROBERTA Large + Stratified Splits + QA Marked ==="
# python scripts/train_single_task.py --config configs/experiments/task_a/fine_tuning/roberta/large/qa_marked/stratified.yaml

# echo "=== Training: ROBERTA Large + Stratified Splits + QA Marked + Balanced Weights ==="
# python scripts/train_single_task.py --config configs/experiments/task_a/fine_tuning/roberta/large/qa_marked/balanced_weighted_stratified.yaml

# echo "=== Training: ROBERTA Large + Stratified Splits + QA Marked + Sqrt Weights ==="
# python scripts/train_single_task.py --config configs/experiments/task_a/fine_tuning/roberta/large/qa_marked/sqrt_weighted_stratified.yaml

echo "=== ALL TRAINING COMPLETED SUCCESSFULLY ✅ ==="