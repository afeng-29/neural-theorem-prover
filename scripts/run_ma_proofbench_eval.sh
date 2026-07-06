#!/bin/bash
# MA-ProofBench evaluation — DeepSeek-Prover-V1.5-RL (whole-proof + MCTS tree search).
#
# Submit (DeepSeek + MCTS, ~12-20hrs):
#   grid_run --grid_submit=batch --grid_gpu --grid_mem=100G --grid_ncpus=8 --grid_long \
#       /user/af3698/neural-theorem-prover/scripts/run_ma_proofbench_eval.sh
#
# Submit (ByT5 only, no GPU needed, ~2hrs):
#   grid_run --grid_submit=batch --grid_mem=20G --grid_ncpus=8 \
#       /user/af3698/neural-theorem-prover/scripts/run_ma_proofbench_eval.sh byt5
set -euo pipefail

MODE="${1:-deepseek}"  # deepseek | byt5

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
lake build TheoremProver 2>&1 | tail -5
echo "Lean project built OK"
cd /user/af3698/neural-theorem-prover

echo "=== [3/3] MA-ProofBench eval (mode=$MODE) ==="
mkdir -p results

if [ "$MODE" = "deepseek" ]; then
    python3 -u scripts/run_ma_proofbench_eval.py \
        --model-type deepseek \
        --model-path "$MODEL_PATH" \
        --lean-project "$LEAN_PROJECT" \
        --output results/ma_proofbench_deepseek.json \
        --samples 200 \
        --max-new-tokens 512 \
        --timeout 300 \
        --tree-timeout 120 \
        --tree-width 8 \
        --tree-depth 6 \
        --resume
elif [ "$MODE" = "byt5" ]; then
    python3 -u scripts/run_ma_proofbench_eval.py \
        --model-type byt5-pretrained \
        --model-path models/pretrained/byt5-small \
        --lean-project "$LEAN_PROJECT" \
        --output results/ma_proofbench_byt5.json \
        --top-k 32 \
        --timeout 120 \
        --tree-timeout 0 \
        --resume
else
    echo "Unknown mode: $MODE (use 'deepseek' or 'byt5')"
    exit 1
fi

echo ""
echo "=== Final scores ==="
python3 -c "
import json, sys
for f in ['results/ma_proofbench_deepseek.json', 'results/ma_proofbench_byt5.json']:
    try:
        d = json.load(open(f))
        s = d.get('summary', {})
        print(f'{f}: {s.get(\"proved\",\"?\")}/{s.get(\"total\",\"?\")} = {s.get(\"pct\",\"?\")}%')
        for lvl in [\"level1\", \"level2\"]:
            lv = s.get(lvl, {})
            if lv:
                n, t = lv.get(\"proved\",0), lv.get(\"total\",0)
                print(f'  {lvl}: {n}/{t} = {100*n/max(t,1):.1f}%')
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f'{f}: error - {e}')
"
