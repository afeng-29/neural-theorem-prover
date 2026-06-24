#!/bin/bash
#SBATCH --job-name=sorrydb_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/sorrydb_eval_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/sorrydb_eval_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# elan/lean/lake are installed in ~/.elan/bin (shared NFS home dir)
export PATH="$HOME/.elan/bin:$PATH"

# Unset token — LeanDojo uses local repo tracing (no GitHub API needed).
# A set token triggers an HTTPS call at import that fails on compute nodes.
unset GITHUB_ACCESS_TOKEN

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

# Step 1: Download and filter SorryDB (fast, CPU-only)
echo "=== Step 1: Fetch SorryDB calculus goals ==="
python scripts/fetch_sorrydb_calculus.py \
    --output data/sorrydb_calculus.jsonl \
    --show-types

echo ""
echo "=== Step 2: Run proof search on SorryDB goals ==="
python scripts/run_sorrydb_eval.py \
    --goals-file   data/sorrydb_calculus.jsonl \
    --lean-project lean_project/ \
    --pretrained   models/pretrained/leandojo-lean4-tacgen-byt5-small \
    --finetuned    models/finetuned/calculus/ \
    --max-goals    50 \
    --timeout      120 \
    --top-k        32 \
    --model        both

echo "=== Done. Results in results/sorrydb_comparison.json ==="
