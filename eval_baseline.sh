#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/eval_baseline_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/eval_baseline_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Pretrained model baseline eval ==="
python training/finetune.py --eval-only \
    --test-data  data/calculus/test.jsonl \
    --base-model models/pretrained/leandojo-lean4-tacgen-byt5-small \
    --top-k 10

# Save a copy to results/ for easy comparison
cp models/pretrained/leandojo-lean4-tacgen-byt5-small/test_metrics.json \
   results/pretrained_test_metrics.json

echo "=== Fine-tuned model eval (sanity check) ==="
python training/finetune.py --eval-only \
    --test-data  data/calculus/test.jsonl \
    --base-model models/finetuned/calculus/ \
    --top-k 10

cp models/finetuned/calculus/test_metrics.json \
   results/finetuned_test_metrics.json

echo "=== Comparison ==="
python - <<'PYEOF'
import json

with open("results/pretrained_test_metrics.json") as f:
    pre = json.load(f)
with open("results/finetuned_test_metrics.json") as f:
    ft = json.load(f)

print(f"{'Metric':<25} {'Pretrained':>12} {'Fine-tuned':>12} {'Delta':>10}")
print("-" * 62)
print(f"{'Top-1 exact match':<25} {pre['top1_exact_match']:>12.2%} {ft['top1_exact_match']:>12.2%} {ft['top1_exact_match']-pre['top1_exact_match']:>+10.2%}")
print(f"{'Top-10 exact match':<25} {pre['top10_exact_match']:>12.2%} {ft['top10_exact_match']:>12.2%} {ft['top10_exact_match']-pre['top10_exact_match']:>+10.2%}")
print(f"{'n_samples':<25} {pre['n_samples']:>12} {ft['n_samples']:>12}")
PYEOF
