#!/bin/bash
# Evaluate base DeepSeek-Prover-V1.5-RL (no LoRA) with top_k=64.
# Goal: improve over the 24.6% (60/244) we got at top_k=32.
set -euo pipefail

cd /user/af3698/neural-theorem-prover
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PATH="$HOME/.elan/bin:$PATH"
LEAN_PROJECT="$(pwd)/lean_project"
MODEL_PATH="models/pretrained/deepseek-prover-v1.5-rl"

echo "=== [1/3] Lean toolchain ==="
TOOLCHAIN=$(cat "$LEAN_PROJECT/lean-toolchain")
elan toolchain install "$TOOLCHAIN" 2>/dev/null || true
elan default "$TOOLCHAIN"
lean --version

echo "=== [2/3] lake build ==="
cd "$LEAN_PROJECT"
lake exe cache get 2>&1 || echo "Cache miss — building from source"
ulimit -s 256
export LEAN_NUM_THREADS=1
lake build TheoremProver 2>&1
echo "Lean build OK"

echo "=== [3/3] Eval: base model, top_k=64 ==="
cd /user/af3698/neural-theorem-prover
mkdir -p results
export CUDA_LAUNCH_BLOCKING=1

python3 scripts/run_minif2f_eval.py \
    --model-type deepseek \
    --model-path "$MODEL_PATH" \
    --lean-project "$LEAN_PROJECT" \
    --split test --top-k 64 --max-new-tokens 1024 \
    --timeout 300 \
    --output results/minif2f_deepseek_base_topk64_test.json \
    --resume

python3 -c "
import json, sys
data = json.load(open('results/minif2f_deepseek_base_topk64_test.json'))
r = data.get('results', data)
proved = sum(1 for v in r.values() if isinstance(v, dict) and v.get('proved'))
total = len([v for v in r.values() if isinstance(v, dict)])
print(f'Solved: {proved}/{total} = {100*proved/total:.1f}%')
print(f'Previous best (top_k=32): 24.6% (60/244)')
print(f'Baseline to beat: 24.6%')
"
