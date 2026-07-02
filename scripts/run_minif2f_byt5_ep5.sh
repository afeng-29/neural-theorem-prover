#!/bin/bash
#SBATCH --job-name=mf2f_ep5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/minif2f_byt5_ep5_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/minif2f_byt5_ep5_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# Evaluate ByT5 fine-tuned on Mathlib for 5 epochs on miniF2F-test.

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
unset GITHUB_ACCESS_TOKEN
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "=== miniF2F: ByT5 Mathlib FT epoch 5 ==="

python3 scripts/run_minif2f_eval.py \
    --model-type   byt5-ft \
    --model-path   models/finetuned/mathlib_all_ep5/checkpoint-15676 \
    --lean-project lean_project/ \
    --split        test \
    --top-k        32 \
    --timeout      120 \
    --output       results/minif2f_byt5_ep5_test.json \
    --resume

echo "=== Done. Results in results/minif2f_byt5_ep5_test.json ==="
