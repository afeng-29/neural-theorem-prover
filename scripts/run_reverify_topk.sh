#!/bin/bash
# Reverify top_k=256 whole-proof results via individual lake-env-lean calls.
# Submit: grid_run --grid_submit=batch --grid_gpu --grid_mem=20G --grid_ncpus=8 \
#           /user/af3698/neural-theorem-prover/scripts/run_reverify_topk.sh
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
# Warm up the cache so lean env calls are fast
cd "$LEAN_PROJECT" && lake build TheoremProver 2>&1 | tail -3
cd /user/af3698/neural-theorem-prover

echo "=== [2/2] Reverify top_k=256 (114 claimed proofs) ==="
python3 -u scripts/reverify_topk.py results/minif2f_deepseek_base_topk256_test.json
