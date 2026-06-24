#!/bin/bash
#SBATCH --job-name=proof_comparison
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/proof_comparison_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/proof_comparison_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# elan/lean/lake are installed in ~/.elan/bin (shared NFS home dir)
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"

# LeanDojo traces a LOCAL repo and does not need GITHUB_ACCESS_TOKEN.
# If the token is set it triggers a HTTPS call to api.github.com at import
# time, which fails on cluster compute nodes (SSL cert mismatch). Unset it.
unset GITHUB_ACCESS_TOKEN

# Point Python's ssl module to the cluster CA bundle so any remaining HTTPS
# calls (e.g. HuggingFace tokenizer downloads) succeed.
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "=== Proof Search Comparison: Pretrained vs Fine-Tuned ==="
echo "n_theorems: 23 (calculus ProofGoals.lean Group A-D)"
echo "timeout: 120s per theorem"
echo "top_k: 32 tactic candidates per step"

python scripts/compare_proof_search.py \
    --lean-project lean_project/ \
    --pretrained   models/pretrained/leandojo-lean4-tacgen-byt5-small \
    --finetuned    models/finetuned/calculus/ \
    --timeout      120 \
    --top-k        32 \
    --model        both

echo "=== Done. Results in results/proof_search_comparison.json ==="
