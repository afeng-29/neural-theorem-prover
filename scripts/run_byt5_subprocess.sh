#!/bin/bash
#SBATCH --job-name=byt5_subproc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/byt5_subprocess_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/byt5_subprocess_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# Clean ByT5 comparison via subprocess lake build — no REPL needed.
# Runs pretrained + analysis-FT ByT5 on 24 calculus theorems using
# _prove_byt5_subprocess (top-k single-step tactics tried via verify_proofs_parallel).
# Gives valid comparable numbers separate from the DeepSeek model switch.

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
unset GITHUB_ACCESS_TOKEN
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "=== ByT5 subprocess evaluation (no REPL) on 24 calculus theorems ==="
echo "Pretrained:  models/pretrained/leandojo-lean4-tacgen-byt5-small"
echo "Fine-tuned:  models/finetuned/analysis/"
echo "Mode:        --use-subprocess (1-step lake build, no LeanDojo REPL)"
echo ""

python3 scripts/compare_proof_search.py \
    --lean-project lean_project/ \
    --pretrained   models/pretrained/leandojo-lean4-tacgen-byt5-small \
    --finetuned    models/finetuned/analysis/ \
    --timeout      120 \
    --top-k        32 \
    --model        both \
    --model-type   byt5 \
    --use-subprocess \
    --output       results/byt5_subprocess_comparison.json

echo "=== Done. Results in results/byt5_subprocess_comparison.json ==="
