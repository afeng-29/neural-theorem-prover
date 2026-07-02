#!/bin/bash
# LoRA fine-tuning for DeepSeek-Prover on a high-memory GPU (A100/H100).
# Runs without SLURM — just execute directly: bash scripts/run_lora_train_v2.sh
#
# Requirements:
#   - Python 3.10+ with venv at ./venv (or activate your own env first)
#   - CUDA 12.x, PyTorch with BF16 support
#   - ~40GB GPU VRAM for full BF16 (no quantization)
#   - DeepSeek-Prover-V1.5-RL weights at: models/pretrained/deepseek-prover-v1.5-rl
#     (or pass MODEL_PATH env var)
#
# For smaller GPUs (< 40GB): export USE_4BIT=1 before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${MODEL_PATH:-models/pretrained/deepseek-prover-v1.5-rl}"
TRAIN_DATA="data/deepseek_lora_train_v2.jsonl"
OUTPUT_DIR="models/finetuned/deepseek_lora_v2"

# Activate venv if present and not already active
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Optional: 4-bit for smaller GPUs
EXTRA_ARGS=""
if [ "${USE_4BIT:-0}" = "1" ]; then
    echo ">>> Using 4-bit NF4 quantization"
    EXTRA_ARGS="--load-in-4bit"
fi

echo "=== DeepSeek-Prover LoRA fine-tuning (v2) ==="
echo "Model:      $MODEL_PATH"
echo "Data:       $TRAIN_DATA ($(wc -l < $TRAIN_DATA) examples)"
echo "Output:     $OUTPUT_DIR"
echo ""

python3 training/finetune_deepseek_v2.py \
    --model-path   "$MODEL_PATH" \
    --train-data   "$TRAIN_DATA" \
    --output-dir   "$OUTPUT_DIR" \
    --epochs       3 \
    --batch-size   4 \
    --grad-accum   4 \
    --lr           2e-4 \
    --lora-r       64 \
    --lora-alpha   128 \
    --max-length   2048 \
    --save-steps   200 \
    $EXTRA_ARGS

echo ""
echo "=== Training done. Adapter at $OUTPUT_DIR/lora_adapter ==="
echo ""
echo "Next: run miniF2F evaluation with:"
echo "  python3 scripts/run_minif2f_eval.py \\"
echo "    --model-type deepseek \\"
echo "    --model-path $MODEL_PATH \\"
echo "    --lora-adapter $OUTPUT_DIR/lora_adapter \\"
echo "    --lean-project lean_project/ \\"
echo "    --split test \\"
echo "    --top-k 32 \\"
echo "    --max-new-tokens 1024 \\"
echo "    --timeout 300 \\"
echo "    --output results/minif2f_deepseek_lora_v2_test.json"
