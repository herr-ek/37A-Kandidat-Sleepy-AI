#!/bin/bash
# ==============================================================================
# Fit check — KNN model
#
# Loads a saved KNN checkpoint and evaluates it on the predefined train /
# validation / test splits, then prints an overfitting/underfitting report.
#
# Submit from the project root:
#   sbatch jobs/train_err_knn.sh
#
# Override the model by passing MODEL_JSON as an environment variable:
#   MODEL_JSON=data/models/knn_apnea_101.json sbatch jobs/train_err_knn.sh
# ==============================================================================

#SBATCH --job-name=sleepy-knn-fitcheck
#SBATCH --output=jobs/logs/%j/knn_fitcheck_%j.out
#SBATCH --error=jobs/logs/%j/knn_fitcheck_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=long

set -euo pipefail

# Run from project root regardless of where sbatch was called.
cd "$SLURM_SUBMIT_DIR"

# Create log directory.
mkdir -p "jobs/logs/$SLURM_JOB_ID"

# Activate virtual environment.
source .venv/bin/activate

# Default model — override with MODEL_JSON env var.
MODEL_JSON="${MODEL_JSON:-data/models/batch/knn_k501_norm_20260422_095944.json}"

echo "=== KNN Fit Check ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Model  : $MODEL_JSON"
echo "Node   : $(hostname)"
echo "Started: $(date)"
echo "=============================="

python jobs/train_err.py --model-json "$MODEL_JSON"

echo "=============================="
echo "Finished: $(date)"
