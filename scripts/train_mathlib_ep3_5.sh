#!/bin/bash
#SBATCH --job-name=byt5_ep3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/train_mathlib_ep3_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/train_mathlib_ep3_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# Continue Mathlib fine-tuning from epoch-2 checkpoint (mathlib_all_ep2_5/checkpoint-15676).
# Trains 3 more epochs (epochs 3-5 of the full run).

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "=== Continuing Mathlib fine-tuning from ep2 checkpoint ==="
echo "Base: models/finetuned/mathlib_all_ep2_5/checkpoint-15676"

python3 training/finetune.py \
    --train-data       data/mathlib_all/train.jsonl \
    --val-data         data/mathlib_all/val.jsonl \
    --test-data        data/mathlib_all/test.jsonl \
    --base-model       models/finetuned/mathlib_all_ep2_5/checkpoint-15676 \
    --output-dir       models/finetuned/mathlib_all_ep3_5/ \
    --epochs           3 \
    --batch-size       4 \
    --grad-accum       4 \
    --max-input-length 512

echo "=== Done. Model at models/finetuned/mathlib_all_ep3_5/ ==="
