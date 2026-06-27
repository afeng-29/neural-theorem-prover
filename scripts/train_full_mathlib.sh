#!/bin/bash
#SBATCH --job-name=byt5_mathlib
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/train_mathlib_all_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/train_mathlib_all_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# Fine-tune ByT5-small on ALL of Mathlib (250K examples).
# Warm start from analysis FT checkpoint (models/finetuned/analysis/).
# Expected: ~18-22h for 5 epochs over 250K examples.

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "=== Step 1: Prepare full Mathlib data ==="

if [ ! -f data/mathlib_all/train.jsonl ]; then
    python3 data/prepare_calculus.py \
        --domain all \
        --output-dir data/mathlib_all
    echo "Data prepared."
else
    echo "Data already at data/mathlib_all/, skipping prep."
fi

TRAIN_SIZE=$(wc -l < data/mathlib_all/train.jsonl 2>/dev/null || echo 0)
echo "Training examples: $TRAIN_SIZE"

echo ""
echo "=== Step 2: Fine-tune ByT5 on full Mathlib ==="

python3 training/finetune.py \
    --train-data    data/mathlib_all/train.jsonl \
    --val-data      data/mathlib_all/val.jsonl \
    --test-data     data/mathlib_all/test.jsonl \
    --base-model    models/finetuned/analysis/ \
    --output-dir    models/finetuned/mathlib_all/ \
    --epochs        5 \
    --batch-size    4 \
    --grad-accum    4 \
    --max-input-length 512

echo ""
echo "=== Done. Model at models/finetuned/mathlib_all/ ==="
