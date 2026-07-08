#!/bin/bash
# Submit miniF2F-test evaluation as a GPU batch job on the CBS Research Grid (SGE).
# Run after training completes:
#   bash scripts/submit_eval_gpu.sh
#
# Expects:
#   - LoRA adapter at: models/finetuned/deepseek_lora_v2/lora_adapter/
#   - Lean 4 + Mathlib built in: lean_project/
#   - Base model at: models/pretrained/deepseek-prover-v1.5-rl/ (or $MODEL_PATH)

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-models/pretrained/deepseek-prover-v1.5-rl}"
ADAPTER="models/finetuned/deepseek_lora_v2/lora_adapter"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" results

if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: LoRA adapter not found at $ADAPTER"
    echo "Run training first: bash scripts/submit_train_gpu.sh"
    exit 1
fi

# Activate venv if present
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

cat > /tmp/run_eval_$$.sh << INNEREOF
#!/bin/bash
set -euo pipefail
cd $(pwd)
[ -f venv/bin/activate ] && source venv/bin/activate
python3 scripts/run_minif2f_eval.py \\
    --model-type deepseek \\
    --model-path "$MODEL_PATH" \\
    --lora-adapter "$ADAPTER" \\
    --lean-project lean_project/ \\
    --split test \\
    --top-k 32 \\
    --max-new-tokens 1024 \\
    --timeout 300 \\
    --output results/minif2f_deepseek_lora_v2_test.json \\
    --resume
INNEREOF
chmod +x /tmp/run_eval_$$.sh

EVAL_JOB=$(grid_run \
    --grid_submit=batch \
    --grid_gpu \
    --grid_mem=40G \
    --grid_ncpus=4 \
    --grid_long \
    -wd "$(pwd)" \
    -N "minif2f_eval_lora_v2" \
    -o "$LOG_DIR/minif2f_eval_lora_v2.o" \
    -e "$LOG_DIR/minif2f_eval_lora_v2.e" \
    /tmp/run_eval_$$.sh 2>&1 | grep -oP 'job \K[0-9]+' || true)

echo "Eval job submitted: $EVAL_JOB"
echo ""
echo "Monitor with:"
echo "  qstat -j $EVAL_JOB"
echo "  tail -f $LOG_DIR/minif2f_eval_lora_v2.o"
echo ""
echo "Results will be written to: results/minif2f_deepseek_lora_v2_test.json"
