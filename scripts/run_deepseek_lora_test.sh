#!/bin/bash
#SBATCH --job-name=ds_ltest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/deepseek_lora_test_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/deepseek_lora_test_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# Evaluate DeepSeek-Prover + LoRA adapter on miniF2F-test.
# Run after run_deepseek_lora_train.sh.

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
unset GITHUB_ACCESS_TOKEN
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "=== DeepSeek-Prover + LoRA: miniF2F test ==="

python3 scripts/run_minif2f_eval.py \
    --model-type     deepseek \
    --model-path     models/pretrained/deepseek-prover-v1.5-rl \
    --lora-adapter   models/finetuned/deepseek_minif2f/lora_adapter \
    --lean-project   lean_project/ \
    --split          test \
    --top-k          32 \
    --max-new-tokens 1024 \
    --timeout        300 \
    --load-in-4bit \
    --output         results/minif2f_deepseek_lora_test.json \
    --resume

echo "=== Done. Results in results/minif2f_deepseek_lora_test.json ==="
