#!/bin/bash
#SBATCH --job-name=eval_analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/eval_analysis_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/eval_analysis_%j.log
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

# Step 1: Tactic accuracy — analysis model on calculus test set (117 examples)
# Gives the missing leg of the 3-way comparison:
#   pretrained 5.98%  /  calculus-FT 16.24%  /  analysis-FT ???
echo "=== Step 1: Tactic accuracy — analysis model on calculus test set ==="
python training/finetune.py --eval-only \
    --test-data      data/calculus/test.jsonl \
    --base-model     models/finetuned/analysis/ \
    --top-k          10 \
    --metrics-output results/analysis_on_calculus_metrics.json

echo ""

# Step 2: Proof search — analysis model on 24 calculus ProofGoals theorems
echo "=== Step 2: Proof search — analysis model on 24 calculus theorems ==="
python scripts/compare_proof_search.py \
    --lean-project lean_project/ \
    --pretrained   models/pretrained/leandojo-lean4-tacgen-byt5-small \
    --finetuned    models/finetuned/analysis/ \
    --timeout      120 \
    --top-k        32 \
    --model        finetuned \
    --output       results/proof_search_analysis.json

echo "=== Done ==="
