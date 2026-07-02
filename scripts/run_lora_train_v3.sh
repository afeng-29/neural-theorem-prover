#!/bin/bash
# LoRA v3: trained on multi-step competition-math proofs (miniF2F valid + filtered Lean-Workbook).
#
# Key changes vs v2:
#   - Training data: 170 multi-step proofs instead of 10,561 one-liners
#   - Lower LR (1e-4 vs 2e-4) and fewer epochs (2 vs 3) — small dataset, avoid overfit
#   - Smaller LoRA rank (r=32, alpha=64) — less capacity needed for 170 examples
#   - CUDA_VISIBLE_DEVICES=0 — avoids multi-GPU tensor corruption on SGE nodes
#
# Submit: grid_run --grid_submit=batch --grid_gpu --grid_mem=60G \
#             /user/af3698/neural-theorem-prover/scripts/run_lora_train_v3.sh

set -euo pipefail

cd /user/af3698/neural-theorem-prover
source venv/bin/activate

# Use SGE-allocated GPU; fall back to 0 if not set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$HOME/.elan/bin:$PATH"

MODEL_PATH="models/pretrained/deepseek-prover-v1.5-rl"
TRAIN_DATA="data/deepseek_lora_multistep.jsonl"
OUTPUT_DIR="models/finetuned/deepseek_lora_v3"

echo "=== DeepSeek-Prover LoRA v3 fine-tuning ==="
echo "Data: $TRAIN_DATA ($(wc -l < $TRAIN_DATA) examples of multi-step competition math)"
echo "Output: $OUTPUT_DIR"

python3 training/finetune_deepseek_v2.py \
    --model-path   "$MODEL_PATH" \
    --train-data   "$TRAIN_DATA" \
    --output-dir   "$OUTPUT_DIR" \
    --epochs       2 \
    --batch-size   4 \
    --grad-accum   4 \
    --lr           1e-4 \
    --lora-r       32 \
    --lora-alpha   64 \
    --max-length   2048 \
    --save-steps   50

echo ""
echo "=== Training complete. Adapter at $OUTPUT_DIR/lora_adapter ==="
