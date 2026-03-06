#!/usr/bin/env bash

echo "=== Base runtime starting on $(hostname) ==="

# module load cuda || true
# module purge

# -------------------------------------------------
# Resolve .env relative to this script (collab safe)
# -------------------------------------------------
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_FILE="$BASE_DIR/../.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading env from: $ENV_FILE"
    source "$ENV_FILE"
else
    echo "ERROR - .env not found at:"
    echo "  $ENV_FILE"
    echo ""
    echo "Each collaborator must create their own .env (see .env.example)"
    exit 1
fi

# Initialise conda from user-provided base dir
if [ -z "$CONDA_BASE_DIR" ]; then
  echo "ERROR - CONDA_BASE_DIR not set"
  echo "Set it in your .env file"
  exit 1
fi

source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"

# -------------------------------------------------
# Shared safety check so jobs fail fast if .env is wrong
# -------------------------------------------------
if [ -z "$MODEL_CACHE_DIR" ] || \
   [ -z "$SCRATCH_DIR" ] || \
   [ -z "$MODEL_OUTPUTS_DIR" ] || \
   [ -z "$MASKING_CACHE_DIR" ] || \
   [ -z "$CONDA_ENV_DIR" ] || \
   [ -z "$WANDB_ENTITY" ] || \
   [ -z "$WANDB_PROJECT" ] || \
   [ -z "$HF_API_KEY" ] || \
   [ -z "$OPENAI_API_KEY" ] || \
   [ -z "$WANDB_API_KEY" ]; then

  echo "ERROR - Required environment variables not set."
  echo ""
  echo "Missing variables:"

  [ -z "$MODEL_CACHE_DIR" ]    && echo "  - MODEL_CACHE_DIR"
  [ -z "$SCRATCH_DIR" ]    && echo "  - SCRATCH_DIR"
  [ -z "$MODEL_OUTPUTS_DIR" ]    && echo "  - MODEL_OUTPUTS_DIR"
  [ -z "$MASKING_CACHE_DIR" ]    && echo "  - MASKING_CACHE_DIR"
  [ -z "$CONDA_ENV_DIR" ] && echo "  - CONDA_ENV_DIR"
  [ -z "$WANDB_ENTITY" ]   && echo "  - WANDB_ENTITY"
  [ -z "$WANDB_API_KEY" ]  && echo "  - WANDB_API_KEY"
  [ -z "$HF_API_KEY" ]  && echo "  - HF_API_KEY"
  [ -z "$OPENAI_API_KEY" ]  && echo "  - OPENAI_API_KEY"
  [ -z "$WANDB_PROJECT" ]  && echo "  - WANDB_PROJECT"

  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
      return 1
  else
      exit 1
  fi
fi

# -------------------------------------------------
# Export vars so Python + W&B can actually see them
# -------------------------------------------------

export SCRATCH_DIR
export MODEL_CACHE_DIR
export MODEL_OUTPUTS_DIR
export MASKING_CACHE_DIR
export CONDA_ENV_DIR
export WANDB_ENTITY
export WANDB_API_KEY
export HF_API_KEY
export WANDB_PROJECT
export OPENAI_API_KEY

conda activate "$CONDA_ENV_DIR"

# Stablise jobs
# export PYTORCH_ENABLE_MPS_FALLBACK=1
# export TORCH_CUDA_FUSER_DISABLE_FALLBACK=1
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export TORCH_DISABLE_SDPA=1

echo "=== SLURM Job ID: $SLURM_JOB_ID ==="

echo ""
echo "=== Python Environment ==="
which python
python --version
echo "Activated conda env: $CONDA_ENV_DIR"
echo "CUDA module loaded: $(module list 2>&1 | grep cuda)"

echo "=== Checking CUDA & GPU ==="
python - << 'EOF'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
EOF

echo ""
echo "=== GPU Check with nvidia-smi ==="
nvidia-smi

echo ""
echo "=== GPU Debug Info ==="
nvidia-debugdump -l

echo "=== modules (if any) ==="
module list 2>&1

echo "=== LD_LIBRARY_PATH CUDA bits ==="
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | egrep -i "cuda|cudnn|nccl" || true

python - <<'PY'
import os, torch
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
print("LD_LIBRARY_PATH head:", (os.environ.get("LD_LIBRARY_PATH","")[:400]))
print("torch.version.cuda:", torch.version.cuda)
PY

echo ""
echo "=== Training Runs Begin ==="
echo ""

