#!/bin/bash
#SBATCH --job-name=ds_sorry2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/deepseek_sorrydb_resume_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/deepseek_sorrydb_resume_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# Resume SorryDB evaluation from goal 72 (first run crashed at 71 with OOM).
# Fixes applied: input truncation at 2048 tokens + per-goal OOM catch.

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
unset GITHUB_ACCESS_TOKEN
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "=== SorryDB Resume: goals 72-100 ==="

python3 scripts/run_deepseek_sorrydb.py \
    --goals-file   data/sorrydb_calculus.jsonl \
    --lean-project lean_project/ \
    --model-path   models/pretrained/deepseek-prover-v1.5-rl \
    --max-goals    100 \
    --skip-goals   71 \
    --timeout      300 \
    --top-k        32 \
    --load-in-4bit \
    --output       results/sorrydb_deepseek_resume.json

echo "=== Done. Results in results/sorrydb_deepseek_resume.json ==="
