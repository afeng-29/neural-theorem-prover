#!/usr/bin/env bash
# setup.sh — one-command environment setup for the neural theorem prover
# Run from the repo root: bash setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEAN_PROJECT="$REPO_ROOT/lean_project"
MODELS_DIR="$REPO_ROOT/models/pretrained"

echo "=== Neural Theorem Prover Setup ==="
echo "Repo root: $REPO_ROOT"

# ── 1. Check prerequisites ────────────────────────────────────────────────────
echo ""
echo "[1/6] Checking prerequisites..."

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_PYTHON="3.10"
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "  Python $PYTHON_VERSION — OK"
else
    echo "ERROR: Python $PYTHON_VERSION found, but 3.10+ required."
    exit 1
fi

# ── 2. Python virtual environment ─────────────────────────────────────────────
echo ""
echo "[2/6] Setting up Python virtual environment..."

if [ ! -d "$REPO_ROOT/.venv" ]; then
    python3 -m venv "$REPO_ROOT/.venv"
    echo "  Created .venv"
else
    echo "  .venv already exists — skipping creation"
fi

source "$REPO_ROOT/.venv/bin/activate"
pip install --upgrade pip --quiet
pip install -r "$REPO_ROOT/requirements.txt" --quiet
echo "  Python dependencies installed"

# ── 3. elan (Lean version manager) ───────────────────────────────────────────
echo ""
echo "[3/6] Installing elan (Lean version manager)..."

if command -v elan &>/dev/null; then
    echo "  elan already installed: $(elan --version)"
else
    curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --no-modify-path
    # Add elan to PATH for the rest of this script
    export PATH="$HOME/.elan/bin:$PATH"
    echo "  elan installed"
fi

export PATH="$HOME/.elan/bin:$PATH"

# ── 4. Lean 4 via elan ────────────────────────────────────────────────────────
echo ""
echo "[4/6] Installing Lean 4 toolchain..."

# Install the toolchain pinned in lean_project/lean-toolchain
LEAN_TOOLCHAIN=$(cat "$LEAN_PROJECT/lean-toolchain" 2>/dev/null || echo "leanprover/lean4:v4.14.0")
echo "  Target toolchain: $LEAN_TOOLCHAIN"

elan toolchain install "$LEAN_TOOLCHAIN" 2>/dev/null || true
elan default "$LEAN_TOOLCHAIN"
echo "  Lean toolchain ready: $(lean --version)"

# ── 5. Build the Lean project (downloads + compiles mathlib) ──────────────────
echo ""
echo "[5/6] Building lean_project (downloads mathlib — this can take 20-40 min on first run)..."
echo "  Mathlib will be cached at ~/.elan and ~/.cache/mathlib after the first build."

cd "$LEAN_PROJECT"

# Use lake exe cache get to pull prebuilt mathlib OLean files instead of compiling
if command -v lake &>/dev/null; then
    lake exe cache get 2>/dev/null || echo "  (cache miss — compiling mathlib from source)"
    lake build --quiet
    echo "  Lean project built successfully"
else
    echo "WARNING: lake not found. Run 'cd lean_project && lake build' manually after elan is on PATH."
fi

cd "$REPO_ROOT"

# ── 6. Download pretrained ReProver checkpoint ────────────────────────────────
echo ""
echo "[6/6] Downloading ReProver pretrained checkpoint..."
echo "  Model: kaiyuy/leandojo-lean4-tacgen-byt5-small"
echo "  Destination: $MODELS_DIR"

mkdir -p "$MODELS_DIR"

python3 - <<'PYEOF'
import os, sys
models_dir = os.environ.get("MODELS_DIR", "models/pretrained")
os.makedirs(models_dir, exist_ok=True)
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_id = "kaiyuy/leandojo-lean4-tacgen-byt5-small"
    local_path = os.path.join(models_dir, "leandojo-lean4-tacgen-byt5-small")
    if os.path.exists(os.path.join(local_path, "config.json")):
        print(f"  Checkpoint already at {local_path} — skipping download")
    else:
        print(f"  Downloading {model_id}...")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        tok.save_pretrained(local_path)
        model.save_pretrained(local_path)
        print(f"  Saved to {local_path}")
except ImportError:
    print("  WARNING: transformers not importable. Run 'source .venv/bin/activate' first.")
    sys.exit(1)
PYEOF

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Before running inference, set your GitHub token (required by LeanDojo):"
echo "  export GITHUB_ACCESS_TOKEN=<your_token>"
echo ""
echo "Activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Run a test:"
echo "  python3 -c \"from prover import ProofSearch; print('import OK')\""
echo ""
echo "See README.md for full usage."
