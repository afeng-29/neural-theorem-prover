#!/apps/anaconda3/bin/python
"""Download DeepSeek-Prover-V1.5-RL from HuggingFace."""
import os, sys
from pathlib import Path

# Add project venv to path
proj = Path(__file__).parent.parent
venv_site = proj / "venv/lib/python3.10/site-packages"
if venv_site.exists():
    sys.path.insert(0, str(venv_site))

from huggingface_hub import snapshot_download

dest = proj / "models/pretrained/deepseek-prover-v1.5-rl"
dest.mkdir(parents=True, exist_ok=True)

print(f"Downloading to {dest} ...")
snapshot_download(
    repo_id="deepseek-ai/DeepSeek-Prover-V1.5-RL",
    local_dir=str(dest),
    local_dir_use_symlinks=False,
)
print("Download complete.")
