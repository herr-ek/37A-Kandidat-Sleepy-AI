#!/bin/bash
# ==============================================================================
# Hyperparameter sweep — SVM model
#
# Submit from the project root:
#   sbatch jobs/sweep_svm.sh
#
# Each array task trains one model configuration and saves the result to
# data/models/.  Adjust the #SBATCH directives below for your allocation.
# ==============================================================================

#SBATCH --job-name=sleepy-classical
#SBATCH --output=jobs/logs/%A/svm_%A_%a.out
#SBATCH --error=jobs/logs/%A/svm_%A_%a.err
#SBATCH --array=0-7          # 8 configurations (indices 0–7)
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

# ==============================================================================
# Configuration table
#
# Each entry is a space-separated string:
#   "MODEL [--flag value ...]"
#
# Rows 0-7:  SVM  C ∈ {0.01, 0.1, 1.0, 10.0}  ×  normalized ∈ {True, False}
# ==============================================================================
CONFIGS=(
    "SVM --C 0.01"
    "SVM --C 0.1"
    "SVM --C 1.0"
    "SVM --C 10.0"
    "SVM --C 0.01 --no-normalized"
    "SVM --C 0.1 --no-normalized"
    "SVM --C 1.0 --no-normalized"
    "SVM --C 10.0 --no-normalized"
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
echo "Started  : $(date)"
echo "============================================================"

python jobs/train_single_classical.py \
    --model "$MODEL" \
    "${EXTRA[@]}"

echo "Finished : $(date)"
