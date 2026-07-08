#!/bin/bash
# Download DeepSeek-Prover-V1.5-RL weights from HuggingFace.
# Submitted as a batch job by submit_train_gpu.sh — do not run directly on login node.
set -euo pipefail

cd /user/af3698/neural-theorem-prover
source venv/bin/activate

DEST="models/pretrained/deepseek-prover-v1.5-rl"
mkdir -p "$DEST"

echo "=== Downloading DeepSeek-Prover-V1.5-RL ==="
echo "Target: $(pwd)/$DEST"
date

python3 - << 'PYEOF'
import sys
from huggingface_hub import snapshot_download
from pathlib import Path

dest = Path("models/pretrained/deepseek-prover-v1.5-rl")
dest.mkdir(parents=True, exist_ok=True)

print(f"Downloading deepseek-ai/DeepSeek-Prover-V1.5-RL to {dest.resolve()} ...")
sys.stdout.flush()

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-Prover-V1.5-RL",
    local_dir=str(dest),
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)
print("Download complete.")
PYEOF

echo "=== Download finished ===" && date
