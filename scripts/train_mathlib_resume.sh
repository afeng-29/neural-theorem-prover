#!/bin/bash
#SBATCH --job-name=byt5_mat_r
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/train_mathlib_resume_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/train_mathlib_resume_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# Resume full Mathlib fine-tuning from the epoch-1 checkpoint.
# The first run (job 51168198) completed epoch 1 before hitting the 24h wall.

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "=== Resuming full Mathlib fine-tuning from checkpoint ==="
echo "Checkpoint: models/finetuned/mathlib_all/checkpoint-15676"

python3 training/finetune.py \
    --train-data    data/mathlib_all/train.jsonl \
    --val-data      data/mathlib_all/val.jsonl \
    --test-data     data/mathlib_all/test.jsonl \
    --base-model    models/finetuned/mathlib_all/checkpoint-15676 \
    --output-dir    models/finetuned/mathlib_all/ \
    --epochs        5 \
    --batch-size    4 \
    --grad-accum    4 \
    --max-input-length 512 \
    --resume

echo ""
echo "=== Done. Model at models/finetuned/mathlib_all/ ==="
