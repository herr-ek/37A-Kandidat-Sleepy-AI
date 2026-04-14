#!/bin/bash
# ==============================================================================
# Hyperparameter sweep — deep learning models (FullyConnected)
#
# Submit from the project root:
#   sbatch jobs/sweep_fc.sh
#
# Each array task trains one FCNN configuration.  The GPU node is requested
# automatically; adjust the #SBATCH directives below for your allocation.
# ==============================================================================

#SBATCH --job-name=sleepy-fcnn
#SBATCH --output=jobs/logs/%A/fcnn_%A_%a.out
#SBATCH --error=jobs/logs/%A/fcnn_%A_%a.err
#SBATCH --array=0-23          # 24 configurations (indices 0–23)
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=short

set -euo pipefail

# Make sure we run from the project root regardless of where sbatch was called.
cd "$SLURM_SUBMIT_DIR"

# Create log directory if it does not exist yet.
mkdir -p jobs/logs/$SLURM_ARRAY_JOB_ID

# Activate the project virtual environment.
source .venv/bin/activate

# Add NVIDIA CUDA libraries from venv to library path
# This allows PyTorch to find the CUDA runtime libraries
VENV_NVIDIA_LIBS="${PWD}/.venv/lib/python3.12/site-packages/nvidia"
if [ -d "$VENV_NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}/cu13/lib:${VENV_NVIDIA_LIBS}/cudnn/lib:${VENV_NVIDIA_LIBS}/nccl/lib:${VENV_NVIDIA_LIBS}/cusparselt/lib:${LD_LIBRARY_PATH:-}"
fi

# Diagnostic: Check if CUDA is available to PyTorch
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" || echo "Failed to check CUDA availability"

# ==============================================================================
# Configuration table
#
# Each entry: "MODEL --flag value ..."
#
# FullyConnected (rows 0–23):
#   window_size ∈ {30, 60, 90}  ×  hidden_sizes ∈ {128x64, 256x128}  ×  dropout ∈ {0.2, 0.3} ×  max_pos_weight ∈ {10.0, 15.0}

# ==============================================================================
CONFIGS=(
    # FullyConnected
    "FullyConnected --window-size 30 --hidden-sizes 128,64  --dropout 0.2"
    "FullyConnected --window-size 30 --hidden-sizes 128,64  --dropout 0.2 --max-pos-weight 15.0"
    "FullyConnected --window-size 30 --hidden-sizes 128,64  --dropout 0.3"
    "FullyConnected --window-size 30 --hidden-sizes 128,64  --dropout 0.3 --max-pos-weight 15.0"
    "FullyConnected --window-size 30 --hidden-sizes 256,128 --dropout 0.2"
    "FullyConnected --window-size 30 --hidden-sizes 256,128 --dropout 0.2 --max-pos-weight 15.0"
    "FullyConnected --window-size 30 --hidden-sizes 256,128 --dropout 0.3"
    "FullyConnected --window-size 30 --hidden-sizes 256,128 --dropout 0.3 --max-pos-weight 15.0"
    "FullyConnected --window-size 60 --hidden-sizes 128,64  --dropout 0.2"
    "FullyConnected --window-size 60 --hidden-sizes 128,64  --dropout 0.2 --max-pos-weight 15.0"
    "FullyConnected --window-size 60 --hidden-sizes 128,64  --dropout 0.3"
    "FullyConnected --window-size 60 --hidden-sizes 128,64  --dropout 0.3 --max-pos-weight 15.0"
    "FullyConnected --window-size 60 --hidden-sizes 256,128 --dropout 0.2"
    "FullyConnected --window-size 60 --hidden-sizes 256,128 --dropout 0.2 --max-pos-weight 15.0"
    "FullyConnected --window-size 60 --hidden-sizes 256,128 --dropout 0.3"
    "FullyConnected --window-size 60 --hidden-sizes 256,128 --dropout 0.3 --max-pos-weight 15.0"
    "FullyConnected --window-size 90 --hidden-sizes 128,64  --dropout 0.2"
    "FullyConnected --window-size 90 --hidden-sizes 128,64  --dropout 0.2 --max-pos-weight 15.0"
    "FullyConnected --window-size 90 --hidden-sizes 128,64  --dropout 0.3"
    "FullyConnected --window-size 90 --hidden-sizes 128,64  --dropout 0.3 --max-pos-weight 15.0"
    "FullyConnected --window-size 90 --hidden-sizes 256,128 --dropout 0.2"
    "FullyConnected --window-size 90 --hidden-sizes 256,128 --dropout 0.2 --max-pos-weight 15.0"
    "FullyConnected --window-size 90 --hidden-sizes 256,128 --dropout 0.3"
    "FullyConnected --window-size 90 --hidden-sizes 256,128 --dropout 0.3 --max-pos-weight 15.0"
)


# Split the config string for this task into an array of arguments.
IFS=' ' read -ra ARGS <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${ARGS[0]}"
EXTRA=("${ARGS[@]:1}")

echo "============================================================"
echo "Task ID  : $SLURM_ARRAY_TASK_ID"
echo "Model    : $MODEL"
echo "Extra    : ${EXTRA[*]}"
echo "Node     : $(hostname)"
echo "GPU      : ${CUDA_VISIBLE_DEVICES:-<none set>}"
echo "Started  : $(date)"
echo "============================================================"

python jobs/train_single_deep.py \
    --model "$MODEL" \
    "${EXTRA[@]}" \
    --epochs 20 \
    --batch-size 1024

echo "Finished : $(date)"
