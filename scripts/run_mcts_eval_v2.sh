#!/bin/bash
# Submit: grid_run --grid_submit=batch --grid_gpu --grid_mem=100G --grid_ncpus=8 \
#           /user/af3698/neural-theorem-prover/scripts/run_mcts_eval_v2.sh
set -euo pipefail

cd /user/af3698/neural-theorem-prover
source venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$HOME/.elan/bin:$PATH"
LEAN_PROJECT="$(pwd)/lean_project"
MODEL_PATH="models/pretrained/deepseek-prover-v1.5-rl"

echo "=== [1/3] Lean toolchain ==="
TOOLCHAIN=$(cat "$LEAN_PROJECT/lean-toolchain")
elan toolchain install "$TOOLCHAIN" 2>/dev/null || true
elan default "$TOOLCHAIN" 2>/dev/null || true
lean --version

echo "=== [2/3] lake build ==="
cd "$LEAN_PROJECT"
lake exe cache get 2>&1 || echo "Cache miss — building from source"
ulimit -s 256
export LEAN_NUM_THREADS=1
lake build TheoremProver 2>&1
echo "Lean project built OK"

echo "=== [3/3] MCTS eval v2 (parallel lake env lean verifier) ==="
cd /user/af3698/neural-theorem-prover
mkdir -p results

python3 -u scripts/run_minif2f_mcts_eval.py \
    --model-path "$MODEL_PATH" \
    --lean-project "$LEAN_PROJECT" \
    --output results/minif2f_mcts_eval_v2.json \
    --split test \
    --samples 400 \
    --tree-timeout 120 \
    --timeout 300 \
    --tree-width 8 \
    --tree-depth 6 \
    --resume

echo "=== Done ==="
python3 -c "
import json
data = json.load(open('results/minif2f_mcts_eval_v2.json'))
r = data.get('results', data)
proved = sum(1 for v in r.values() if isinstance(v, dict) and v.get('proved'))
total = len([v for v in r.values() if isinstance(v, dict)])
print(f'MCTS v2: {proved}/{total} = {100*proved/total:.1f}%')
"
