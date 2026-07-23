#!/bin/bash
# Reverify MCTS v2 results (119 claimed proofs) via individual lake env lean calls.
# Submit: grid_run --grid_submit=batch --grid_gpu --grid_mem=20G --grid_ncpus=8 \
#           /user/af3698/neural-theorem-prover/scripts/run_reverify_mcts_v2.sh
set -euo pipefail

cd /user/af3698/neural-theorem-prover
source venv/bin/activate

export PATH="$HOME/.elan/bin:$PATH"
LEAN_PROJECT="$(pwd)/lean_project"

echo "=== [1/2] Lean setup ==="
TOOLCHAIN=$(cat "$LEAN_PROJECT/lean-toolchain")
elan default "$TOOLCHAIN" 2>/dev/null || true
ulimit -s 256
export LEAN_NUM_THREADS=1
cd "$LEAN_PROJECT" && lake build TheoremProver 2>&1 | tail -3
cd /user/af3698/neural-theorem-prover

echo "=== [2/2] Reverify MCTS v2 (119 claimed proofs) ==="
python3 -u scripts/reverify_topk.py results/minif2f_mcts_eval_v2.json
