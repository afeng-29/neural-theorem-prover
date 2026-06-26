#!/bin/bash
#SBATCH --job-name=ds_qlora_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/deepseek_qlora_eval_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/deepseek_qlora_eval_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# Evaluate QLoRA-fine-tuned DeepSeek vs base DeepSeek on 24 calculus theorems.
# Requires: models/finetuned/deepseek-qlora-calculus/best/ (from run_deepseek_qlora_train.sh)

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.elan/bin:$HOME/.local/node16/bin:$PATH"
unset GITHUB_ACCESS_TOKEN
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

ADAPTER=models/finetuned/deepseek-qlora-calculus/best
BASE=models/pretrained/deepseek-prover-v1.5-rl

if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: LoRA adapter not found at $ADAPTER"
    echo "Run run_deepseek_qlora_train.sh first."
    exit 1
fi

echo "=== Step 1: Base DeepSeek (no LoRA) on 24 calculus theorems ==="
python3 scripts/compare_proof_search.py \
    --lean-project lean_project/ \
    --pretrained   "$BASE" \
    --finetuned    "$ADAPTER" \
    --timeout      300 \
    --top-k        32 \
    --model        pretrained \
    --model-type   deepseek \
    --load-in-4bit \
    --output       results/deepseek_qlora_comparison.json

echo ""
echo "=== Step 2: QLoRA-fine-tuned DeepSeek (base + LoRA adapter) ==="
# Use --finetuned flag by running compare with both models
# We pass lora-adapter via a wrapper; easier to invoke run script directly
python3 - <<'PYEOF'
import sys, json
from pathlib import Path
sys.path.insert(0, ".")
from prover import ProofSearch

ADAPTER = "models/finetuned/deepseek-qlora-calculus/best"
BASE    = "models/pretrained/deepseek-prover-v1.5-rl"

# Same 24-theorem list as compare_proof_search.py
from scripts.compare_proof_search import CALCULUS_THEOREMS, run_search

print("Loading QLoRA model (base + adapter)...")
prover = ProofSearch(
    model_path=BASE,
    lean_project="lean_project/",
    top_k=32,
    model_type="deepseek",
    load_in_4bit=True,
    lora_adapter=ADAPTER,
)

results = []
n_ok = 0
for label, thm, hyps, difficulty in CALCULUS_THEOREMS:
    print(f"  [{label}] ...", end=" ", flush=True)
    r = prover.prove(thm, hyps, timeout=300.0)
    status = "OK" if r.verified else "FAIL"
    print(f"{status} ({r.elapsed_seconds:.0f}s)")
    if r.verified:
        n_ok += 1
    results.append({
        "label": label, "theorem": thm, "success": r.verified,
        "proof": r.proof, "elapsed_seconds": r.elapsed_seconds,
    })

n = len(results)
print(f"\nQLoRA: {n_ok}/{n} proved ({100*n_ok/n:.1f}%)")

# Merge into existing JSON
out = json.loads(Path("results/deepseek_qlora_comparison.json").read_text())
out["qlora_finetuned"] = {
    "model": BASE, "adapter": ADAPTER,
    "success": n_ok, "total": n, "success_rate": n_ok/n,
    "results": results,
}
Path("results/deepseek_qlora_comparison.json").write_text(json.dumps(out, indent=2))
print("Saved results/deepseek_qlora_comparison.json")
PYEOF

echo ""
echo "=== Done. Results in results/deepseek_qlora_comparison.json ==="
