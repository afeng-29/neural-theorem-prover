#!/bin/bash
#SBATCH --job-name=ds_valid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/deepseek_valid_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/deepseek_valid_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# DeepSeek-Prover on miniF2F-validation (244 problems).
# Results used as LoRA training data; no test-split leakage.

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
unset GITHUB_ACCESS_TOKEN
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "=== DeepSeek-Prover: miniF2F validation (LoRA training data gen) ==="

python3 scripts/run_minif2f_eval.py \
    --model-type     deepseek \
    --model-path     models/pretrained/deepseek-prover-v1.5-rl \
    --lean-project   lean_project/ \
    --split          validation \
    --top-k          32 \
    --max-new-tokens 1024 \
    --timeout        300 \
    --load-in-4bit \
    --output         results/minif2f_deepseek_valid.json \
    --resume

echo "=== Done. ==="

# If eval succeeded, build training data
if [ -f results/minif2f_deepseek_valid.json ]; then
    echo "=== Building LoRA training data ==="
    python3 scripts/make_deepseek_train_data.py \
        results/minif2f_deepseek_valid.json \
        data/deepseek_lora_train.jsonl
    echo "=== Training data written to data/deepseek_lora_train.jsonl ==="
fi
