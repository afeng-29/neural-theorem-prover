#!/bin/bash
#SBATCH --job-name=deepseek_ps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/deepseek_proof_search_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/deepseek_proof_search_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
unset GITHUB_ACCESS_TOKEN
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

MODEL=models/pretrained/deepseek-prover-v1.5-rl

echo "=== Step 0: Sanity check — DeepSeek-Prover whole-proof generation ==="
python3 -c "
import logging
logging.basicConfig(level=logging.INFO)
from prover.tactic_model import DeepSeekProverModel

m = DeepSeekProverModel('$MODEL', load_in_4bit=True)
theorem = 'Continuous (fun _ : ℝ => c)'
hyps = ['c : ℝ']
print('Theorem:', theorem)
print('Hypotheses:', hyps)
scripts = m.generate_proofs(theorem, hyps, n=16, max_new_tokens=128)
print()
print(f'Generated {len(scripts)} proof scripts:')
for i, script in enumerate(scripts, 1):
    print(f'  script {i:2d}: {script}')

known_correct = ['exact continuous_const', 'fun_prop', 'continuity', 'simp', 'exact?']
found_in_scripts = []
for script in scripts:
    for tactic in script:
        if tactic in known_correct and tactic not in found_in_scripts:
            found_in_scripts.append(tactic)
print()
if found_in_scripts:
    print('FOUND known-correct tactics in scripts:', found_in_scripts)
else:
    print('NONE of', known_correct, 'found in any generated script')
"

echo ""
echo "=== Step 1: Proof search — DeepSeek-Prover on 24 calculus theorems ==="
python3 scripts/compare_proof_search.py \
    --lean-project lean_project/ \
    --finetuned    "$MODEL" \
    --timeout      300 \
    --top-k        32 \
    --model        finetuned \
    --model-type   deepseek \
    --load-in-4bit \
    --log-tactics \
    --output       results/proof_search_deepseek.json

echo ""
echo "=== Step 2: Proof search — DeepSeek-Prover via test_pipeline.py ==="
python3 test_pipeline.py \
    --model-path   "$MODEL" \
    --model-type   deepseek \
    --load-in-4bit \
    --timeout      300 \
    --top-k        32

echo "=== Done ==="
