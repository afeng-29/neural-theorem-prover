#!/bin/bash
# Evaluate DeepSeek-Prover-V1.5-RL on miniF2F using tree search + high-sample-count
# whole-proof generation. Targets the published 60.2% result.
#
# Key parameters vs prior runs:
#   --samples 400   : up from 32/64 (paper uses ~3200, 400 gets ~50-55%)
#   --max-new-tokens 2048 : up from 1024 (allows long proofs)
#   --tree-timeout 120 : BFS tree search phase (efficient for structured proofs)
#   --timeout 300   : total budget per theorem
#
# Submit:
#   grid_run --grid_submit=batch --grid_gpu --grid_mem=60G \
#       /user/af3698/neural-theorem-prover/scripts/eval_mcts.sh
set -euo pipefail

cd /user/af3698/neural-theorem-prover
source venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$HOME/.elan/bin:$PATH"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LEAN_PROJECT="$(pwd)/lean_project"
MODEL_PATH="models/pretrained/deepseek-prover-v1.5-rl"
OUTPUT="results/minif2f_mcts_eval.json"

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

echo "=== [3/3] MCTS + whole-proof eval ==="
cd /user/af3698/neural-theorem-prover
mkdir -p results

python3 scripts/run_minif2f_mcts_eval.py \
    --model-path "$MODEL_PATH" \
    --lean-project "$LEAN_PROJECT" \
    --split test \
    --output "$OUTPUT" \
    --samples 400 \
    --max-new-tokens 2048 \
    --timeout 300 \
    --tree-timeout 120 \
    --tree-width 8 \
    --tree-depth 6 \
    --resume

echo ""
echo "=== Results ==="
python3 -c "
import json, sys
data = json.load(open('$OUTPUT'))
r = data.get('results', data)
proved = sum(1 for v in r.values() if isinstance(v, dict) and v.get('proved'))
total = len([v for v in r.values() if isinstance(v, dict)])
tree = sum(1 for v in r.values() if isinstance(v, dict) and v.get('proved') and v.get('method') == 'tree_search')
whole = sum(1 for v in r.values() if isinstance(v, dict) and v.get('proved') and v.get('method') == 'whole_proof')
print(f'Solved: {proved}/{total} = {100*proved/max(total,1):.1f}%')
print(f'  via tree search: {tree}')
print(f'  via whole-proof: {whole}')
print(f'Published target:  60.2% (147/244)')
print(f'Previous best:     24.6%  (60/244, base model top_k=32)')
"
