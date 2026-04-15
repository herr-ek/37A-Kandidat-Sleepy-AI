#!/bin/bash
# ==============================================================================
# Hyperparameter sweep — RandomForest
#
# Submit from the project root:
#   sbatch jobs/sweep_forest.sh
#
# Each array task trains one model configuration and saves the result to
# data/models/.  Adjust the #SBATCH directives below for your allocation.
# ==============================================================================

#SBATCH --job-name=sleepy-forest
#SBATCH --output=jobs/logs/%A/forest_%A_%a.out
#SBATCH --error=jobs/logs/%A/forest_%A_%a.err
#SBATCH --array=0-17          # 18 configurations (indices 0–17)
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=long

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
# Rows 0-17:   RandomForest  n_estimators ∈ {50, 100, 200}  ×  max_depth ∈ {10, 15, 20}  ×  normalized ∈ {True, False}
# ==============================================================================
CONFIGS=(
    "RandomForest --n-estimators 50  --max-depth 10"
    "RandomForest --n-estimators 50  --max-depth 15"
    "RandomForest --n-estimators 50  --max-depth 20"
    "RandomForest --n-estimators 100 --max-depth 10"
    "RandomForest --n-estimators 100 --max-depth 15"
    "RandomForest --n-estimators 100 --max-depth 20"
    "RandomForest --n-estimators 200 --max-depth 10"
    "RandomForest --n-estimators 200 --max-depth 15"
    "RandomForest --n-estimators 200 --max-depth 20"
    "RandomForest --n-estimators 50  --max-depth 10 --no-normalized"
    "RandomForest --n-estimators 50  --max-depth 15 --no-normalized"
    "RandomForest --n-estimators 50  --max-depth 20 --no-normalized"
    "RandomForest --n-estimators 100 --max-depth 10 --no-normalized"
    "RandomForest --n-estimators 100 --max-depth 15 --no-normalized"
    "RandomForest --n-estimators 100 --max-depth 20 --no-normalized"
    "RandomForest --n-estimators 200 --max-depth 10 --no-normalized"
    "RandomForest --n-estimators 200 --max-depth 15 --no-normalized"
    "RandomForest --n-estimators 200 --max-depth 20 --no-normalized"
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
