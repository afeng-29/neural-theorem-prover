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

# GITHUB_ACCESS_TOKEN is required for LeanDojo to trace the local repo
# Set it via: export GITHUB_ACCESS_TOKEN=<your token>  before sbatch
if [ -z "$GITHUB_ACCESS_TOKEN" ]; then
    echo "ERROR: GITHUB_ACCESS_TOKEN not set. LeanDojo needs it to trace theorems."
    echo "Run: export GITHUB_ACCESS_TOKEN=<your token> && sbatch scripts/run_comparison.sh"
    exit 1
fi

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
