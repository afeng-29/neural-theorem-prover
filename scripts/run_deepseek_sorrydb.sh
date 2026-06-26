#!/bin/bash
#SBATCH --job-name=ds_sorrydb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/deepseek_sorrydb_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/deepseek_sorrydb_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# DeepSeek-Prover-V1.5-RL on SorryDB calculus/analysis goals.
# Uses whole-proof generation + batch subprocess verification (no REPL).
# 100 goals × ~51s = ~1.5 hours; well within 8-hour limit.

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
unset GITHUB_ACCESS_TOKEN
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

MODEL=models/pretrained/deepseek-prover-v1.5-rl

echo "=== Step 1: Refresh SorryDB calculus goals ==="
python3 scripts/fetch_sorrydb_calculus.py \
    --output data/sorrydb_calculus.jsonl \
    --show-types 2>/dev/null || echo "(fetch failed — using cached data/sorrydb_calculus.jsonl)"

echo ""
echo "=== Step 2: DeepSeek proof search on SorryDB (100 goals, timeout=300s) ==="
python3 scripts/run_deepseek_sorrydb.py \
    --goals-file   data/sorrydb_calculus.jsonl \
    --lean-project lean_project/ \
    --model-path   "$MODEL" \
    --max-goals    100 \
    --timeout      300 \
    --top-k        32 \
    --load-in-4bit \
    --output       results/sorrydb_deepseek.json

echo "=== Done. Results in results/sorrydb_deepseek.json ==="
