#!/bin/bash
#SBATCH --job-name=ds_qlora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/deepseek_qlora_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/deepseek_qlora_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# QLoRA fine-tuning of DeepSeek-Prover-V1.5-RL on Mathlib calculus (6,734 examples).
# 4-bit NF4 base + LoRA r=16 adapters; fits V100 16GB.
# ~3 epochs × 6734 steps = ~20h estimate on V100.

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
unset GITHUB_ACCESS_TOKEN
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "=== QLoRA Training: DeepSeek-Prover on calculus data ==="
echo "Base model: models/pretrained/deepseek-prover-v1.5-rl"
echo "Data:       data/calculus/train.jsonl  (6734 examples)"
echo "Output:     models/finetuned/deepseek-qlora-calculus/"
echo ""

python3 scripts/train_deepseek_qlora.py \
    --model-path   models/pretrained/deepseek-prover-v1.5-rl \
    --train-file   data/calculus/train.jsonl \
    --val-file     data/calculus/val.jsonl \
    --output-dir   models/finetuned/deepseek-qlora-calculus \
    --epochs       3 \
    --lr           2e-4 \
    --max-length   512 \
    --grad-accum   8 \
    --warmup-steps 50 \
    --lora-r       16 \
    --lora-alpha   32

echo ""
echo "=== Training complete. Best adapter at models/finetuned/deepseek-qlora-calculus/best ==="
echo "Submit evaluation with: sbatch scripts/run_deepseek_qlora_eval.sh"
