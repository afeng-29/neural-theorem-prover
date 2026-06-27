#!/bin/bash
#SBATCH --job-name=mf2f_byt5p
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/minif2f_byt5_pretrained_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/minif2f_byt5_pretrained_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# ByT5-small pretrained (no fine-tuning) on miniF2F-test (244 problems).
# 1-step tactic proofs only (no REPL). Expected: ~244 × 15s ≈ 1h.

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
unset GITHUB_ACCESS_TOKEN
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "=== miniF2F: ByT5-small pretrained ==="

python3 scripts/run_minif2f_eval.py \
    --model-type   byt5-pretrained \
    --model-path   models/pretrained/leandojo-lean4-tacgen-byt5-small \
    --lean-project lean_project/ \
    --split        test \
    --top-k        32 \
    --timeout      120 \
    --output       results/minif2f_byt5_pretrained_test.json \
    --resume

echo "=== Done. Results in results/minif2f_byt5_pretrained_test.json ==="
