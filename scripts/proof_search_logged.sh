#!/bin/bash
#SBATCH --job-name=ps_logged
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/ps_logged_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/ps_logged_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
unset GITHUB_ACCESS_TOKEN
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "=== Proof search with tactic logging — analysis model ==="
python scripts/compare_proof_search.py \
    --lean-project lean_project/ \
    --finetuned    models/finetuned/analysis/ \
    --timeout      120 \
    --top-k        32 \
    --model        finetuned \
    --log-tactics \
    --output       results/proof_search_logged.json

echo "=== Done ==="
