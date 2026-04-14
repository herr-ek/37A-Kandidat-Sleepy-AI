#!/bin/bash
# ==============================================================================
# Smoke-test sweep — verifies all model implementations work end-to-end.
#
# Runs one small config per model type (classical + deep) with --no-save so
# nothing is written to disk.  Requests a GPU so deep models exercise the
# CUDA path.
#
# Submit from the project root:
#   sbatch jobs/test_sweep.sh
# ==============================================================================

#SBATCH --job-name=sleepy-test
#SBATCH --output=jobs/logs/%j/test_%j.out
#SBATCH --error=jobs/logs/%j/test_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --gres=gpu:L4:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p jobs/logs/$SLURM_JOB_ID

source .venv/bin/activate

# Add NVIDIA CUDA libraries from venv to library path
VENV_NVIDIA_LIBS="${PWD}/.venv/lib/python3.12/site-packages/nvidia"
if [ -d "$VENV_NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}/cu13/lib:${VENV_NVIDIA_LIBS}/cudnn/lib:${VENV_NVIDIA_LIBS}/nccl/lib:${VENV_NVIDIA_LIBS}/cusparselt/lib:${LD_LIBRARY_PATH:-}"
fi

# Diagnostic: Check if CUDA is available to PyTorch
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" || echo "Failed to check CUDA availability"

PASS=0
FAIL=0

run_test() {
    local label="$1"
    shift
    printf "  %-50s" "$label"
    if python "$@" --no-save 2>&1 | tail -1 | grep -q "successfully"; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        python "$@" --no-save 2>&1 | tail -5
        FAIL=$((FAIL + 1))
    fi
}

echo "============================================================"
echo "Smoke-test sweep"
echo "Node     : $(hostname)"
echo "GPU      : ${CUDA_VISIBLE_DEVICES:-<none set>}"
echo "Started  : $(date)"
echo "============================================================"

echo ""
echo "--- Classical models ---"
run_test "KNN (k=5)"                jobs/train_single_classical.py --model KNN          --n-neighbors 5
run_test "RandomForest (n=50, d=5)" jobs/train_single_classical.py --model RandomForest --n-estimators 50 --max-depth 5
run_test "SVM (C=1.0)"              jobs/train_single_classical.py --model SVM          --C 1.0

echo ""
echo "--- Deep learning models ---"
run_test "CNN1D (w=60)"             jobs/train_single_deep.py --model CNN1D          --window-size 60 --epochs 2
run_test "FullyConnected (w=60)"    jobs/train_single_deep.py --model FullyConnected  --window-size 60 --epochs 2
run_test "RNN (w=60)"               jobs/train_single_deep.py --model RNN             --window-size 60 --epochs 2

echo ""
echo "============================================================"
echo "Results  : $PASS passed, $FAIL failed"
echo "Finished : $(date)"
echo "============================================================"

[ "$FAIL" -eq 0 ]
