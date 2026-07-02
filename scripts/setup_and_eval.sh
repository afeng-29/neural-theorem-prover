#!/bin/bash
# Full Lean setup + miniF2F-test evaluation.
# Submitted to a GPU node after training completes.
# Steps: install Lean toolchain → lake cache get → lake build → run eval
set -euo pipefail

cd /user/af3698/neural-theorem-prover
source venv/bin/activate

# Use SGE-allocated GPU if set; fall back to 0. Do NOT hardcode 0 — on multi-GPU
# nodes SGE may allocate GPU 1+ and hardcoding 0 hits another job's memory.
# device_map={"": "cuda:0"} in Python ensures single-GPU placement regardless.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$HOME/.elan/bin:$PATH"
LEAN_PROJECT="$(pwd)/lean_project"
MODEL_PATH="${MODEL_PATH:-models/pretrained/deepseek-prover-v1.5-rl}"
ADAPTER="models/finetuned/deepseek_lora_v2/lora_adapter"

echo "=== [1/4] Lean toolchain ==="
TOOLCHAIN=$(cat "$LEAN_PROJECT/lean-toolchain")
echo "Toolchain: $TOOLCHAIN"
elan toolchain install "$TOOLCHAIN" 2>/dev/null || true
elan default "$TOOLCHAIN"
lean --version

echo ""
echo "=== [2/4] Download prebuilt Mathlib oleans ==="
cd "$LEAN_PROJECT"
lake exe cache get 2>&1 || echo "Cache miss — will compile from source (slow)"

echo ""
echo "=== [3/4] lake build ==="
# RLIMIT_AS = 4GB on this cluster. By step ~1515, mmap'd oleans fill most of it.
# Shrink thread stacks from 8MB → 256KB so pthread_create still fits within the limit.
ulimit -s 256
export LEAN_NUM_THREADS=1
lake build TheoremProver 2>&1
echo "Lean project built OK"

echo ""
echo "=== [4/4] miniF2F-test evaluation ==="
cd /user/af3698/neural-theorem-prover
mkdir -p results

# Synchronize CUDA kernels so assertion errors show the real callsite.
export CUDA_LAUNCH_BLOCKING=1

python3 scripts/run_minif2f_eval.py \
    --model-type deepseek \
    --model-path "$MODEL_PATH" \
    --lora-adapter "$ADAPTER" \
    --lean-project "$LEAN_PROJECT" \
    --split test \
    --top-k 32 \
    --max-new-tokens 1024 \
    --timeout 300 \
    --output results/minif2f_deepseek_lora_v2_test.json \
    --resume

echo ""
echo "=== Evaluation complete. Results: results/minif2f_deepseek_lora_v2_test.json ==="
python3 -c "
import json, sys
try:
    data = json.load(open('results/minif2f_deepseek_lora_v2_test.json'))
    # save_results writes {summary: {...}, results: {pid: {...}, ...}}
    results = data.get('results', data)
    solved = sum(1 for v in results.values() if v.get('proved'))
    total = len(results)
    print(f'Solved: {solved}/{total} = {100*solved/total:.1f}%')
    print(f'Baseline to beat: 24.6% (60/244)')
except Exception as e:
    print(f'Could not parse results: {e}', file=sys.stderr)
"
