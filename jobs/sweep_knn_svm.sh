#!/bin/bash
# ==============================================================================
# Hyperparameter sweep — classical ML models (KNN, SVM)
#
# Submit from the project root:
#   sbatch jobs/sweep_classical.sh
#
# Each array task trains one model configuration and saves the result to
# data/models/.  Adjust the #SBATCH directives below for your allocation.
# ==============================================================================

#SBATCH --job-name=sleepy-classical
#SBATCH --output=jobs/logs/%A/classical_%A_%a.out
#SBATCH --error=jobs/logs/%A/classical_%A_%a.err
#SBATCH --array=0-16          # 17 configurations (indices 0–16)
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
# Rows 0–8:   KNN  k ∈ {3, 5, 7, 11, 15, 21, 51, 101, 201}  ×  normalized ∈ {True, False}
# Rows 9–16:  SVM  C ∈ {0.01, 0.1, 1.0, 10.0}  ×  normalized ∈ {True, False}
# ==============================================================================
CONFIGS=(
    "KNN --n-neighbors 3"
    "KNN --n-neighbors 5"
    "KNN --n-neighbors 7"
    "KNN --n-neighbors 11"
    "KNN --n-neighbors 15"
    "KNN --n-neighbors 21"
    "KNN --n-neighbors 51"
    "KNN --n-neighbors 101"
    "KNN --n-neighbors 201"
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
