#!/bin/bash
#SBATCH --job-name=mf2f_ds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/minif2f_deepseek_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/minif2f_deepseek_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# DeepSeek-Prover-V1.5-RL zero-shot on miniF2F-test (244 problems).
# Expected: ~244 × 90s = ~6h at top_k=8 / batch_size=2 on V100.

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
unset GITHUB_ACCESS_TOKEN
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "=== miniF2F: DeepSeek-Prover zero-shot ==="

python3 scripts/run_minif2f_eval.py \
    --model-type   deepseek \
    --model-path   models/pretrained/deepseek-prover-v1.5-rl \
    --lean-project lean_project/ \
    --split        test \
    --top-k        8 \
    --max-new-tokens 512 \
    --timeout      300 \
    --load-in-4bit \
    --output       results/minif2f_deepseek_test.json \
    --resume

echo "=== Done. Results in results/minif2f_deepseek_test.json ==="
