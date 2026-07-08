#!/bin/bash
# Submit LoRA training as a GPU batch job on the CBS Research Grid (SGE).
# Run this from the neural-theorem-prover project root:
#   bash scripts/submit_train_gpu.sh
#
# Submits two sequential jobs:
#   1. download_model  — fetches DeepSeek-Prover-V1.5-RL from HuggingFace
#   2. lora_train_v2   — LoRA fine-tuning (GPU, 80GB, ~3-4h on A100)
#
# If the model is already at models/pretrained/deepseek-prover-v1.5-rl/,
# the download job is skipped and only training is submitted.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="models/pretrained/deepseek-prover-v1.5-rl"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "=== CBS Grid LoRA Training Submission ==="

# Check if model weights already present
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "Model weights found at $MODEL_DIR — skipping download."
    HOLD_ARG=""
else
    echo "Model not found. Submitting download job first..."
    DL_JOB=$(grid_run \
        --grid_submit=batch \
        --grid_mem=8G \
        --grid_ncpus=4 \
        -wd "$(pwd)" \
        -N "deepseek_download" \
        -o "$LOG_DIR/deepseek_download.o" \
        -e "$LOG_DIR/deepseek_download.e" \
        scripts/download_model.py 2>&1 | grep -oP 'job \K[0-9]+' || true)
    echo "Download job submitted: $DL_JOB"
    HOLD_ARG="--grid_hold=$DL_JOB"
fi

# Submit training job (held until download completes if needed)
TRAIN_JOB=$(grid_run \
    --grid_submit=batch \
    --grid_gpu \
    --grid_mem=80G \
    --grid_ncpus=8 \
    --grid_long \
    ${HOLD_ARG:-} \
    -wd "$(pwd)" \
    -N "deepseek_lora_v2" \
    -o "$LOG_DIR/deepseek_lora_v2.o" \
    -e "$LOG_DIR/deepseek_lora_v2.e" \
    scripts/run_lora_train_v2.sh 2>&1 | grep -oP 'job \K[0-9]+' || true)
echo "Training job submitted: $TRAIN_JOB"

echo ""
echo "Monitor with:"
echo "  qstat -j $TRAIN_JOB"
echo "  tail -f $LOG_DIR/deepseek_lora_v2.o"
echo ""
echo "When training completes, run evaluation with:"
echo "  bash scripts/submit_eval_gpu.sh"
