# Neural Theorem Prover (Lean 4)

A transformer-based tactic prediction system for Lean 4. Given a theorem statement, it runs best-first proof search using the [ReProver](https://github.com/lean-dojo/ReProver) model (ByT5-small, ~300M params) and [LeanDojo](https://github.com/lean-dojo/LeanDojo) for live Lean 4 interaction.

## Quick start: prove a new theorem

```bash
# 1. One-time setup
bash setup.sh

# 2. Set your GitHub token (required by LeanDojo for mathlib access)
export GITHUB_ACCESS_TOKEN=<your_github_token>

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Build the Lean project (downloads + compiles mathlib — 20-40 min first time)
cd lean_project && lake exe cache get && lake build && cd ..

# 5. Prove a theorem
python -c "
from prover import ProofSearch
prover = ProofSearch(
    model_path='models/pretrained/leandojo-lean4-tacgen-byt5-small',
    lean_project='./lean_project',
)
result = prover.prove('∀ n : ℕ, n + 0 = n', hypotheses=[], timeout=60)
print('Verified:', result.verified)
print('Proof:', result.proof)
"
```

---

## Full setup

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| Git | any recent version |
| curl | for elan installer |
| Disk space | ~10 GB (mathlib cache) |
| RAM | 8 GB minimum (16 GB recommended) |

A GPU is optional for inference (the small ByT5 checkpoint runs on CPU in ~2-5s/tactic).
A GPU (≥16 GB VRAM) is required for efficient fine-tuning.

### Step-by-step

```bash
# Clone / navigate to the repo
cd theorem-prover

# Run the one-command setup script
# This: installs elan+Lean, creates a venv, installs Python deps,
#       downloads the ReProver checkpoint.
bash setup.sh

# Set GitHub token — needed for LeanDojo to download mathlib
export GITHUB_ACCESS_TOKEN=ghp_...   # https://github.com/settings/tokens

# Activate venv
source .venv/bin/activate

# Build Lean project (pulls prebuilt mathlib oleans to avoid full compilation)
cd lean_project
lake exe cache get   # downloads prebuilt .olean files
lake build           # finishes the build
cd ..

# Verify everything works
python test_pipeline.py --dry-run   # smoke test without Lean
python test_pipeline.py             # full test (requires Lean + LeanDojo)
```

---

## Running inference

### Python API

```python
from prover import ProofSearch

prover = ProofSearch(
    model_path="models/pretrained/leandojo-lean4-tacgen-byt5-small",
    lean_project="./lean_project",
    top_k=32,        # tactic candidates per step
)

result = prover.prove(
    theorem="∀ n : ℕ, n + 0 = n",
    hypotheses=[],         # list of "name : Type" strings
    timeout=60,            # seconds
    max_depth=20,          # max tactic depth
)

print(result.proof)        # "simp" or multi-line tactic string, or None
print(result.verified)     # bool — Lean accepted the proof
print(result.steps)        # list of (state, tactic) pairs
print(result.elapsed_seconds)
```

### CLI test suite

```bash
python test_pipeline.py [--model-path PATH] [--timeout 60] [--top-k 32]
# Results saved to results/baseline_results.json
```

### Interactive notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Extracting fine-tuning data

```bash
# Extract all proofs from a mathlib module
python data/extract.py \
    --module Mathlib.Data.Nat.Basic \
    --output data/nat_basic.jsonl

# Quick test with a cap on theorems
python data/extract.py \
    --module Mathlib.Data.Nat.Basic \
    --max-theorems 50 \
    --output data/nat_basic_small.jsonl
```

Output is JSONL; each line is one theorem with fields:
`theorem_name`, `statement`, `hypotheses`, `steps`, `full_proof`, `source_module`.

---

## Fine-tuning (when you have domain data)

```bash
# Edit training/finetune.py and uncomment the main() call at the bottom, then:
python training/finetune.py \
    --train-data data/nat_basic.jsonl \
    --base-model models/pretrained/leandojo-lean4-tacgen-byt5-small \
    --output-dir models/finetuned/ \
    --epochs 5 \
    --batch-size 16
```

The fine-tuned model is a drop-in replacement for the pretrained one — just point
`model_path` at `models/finetuned/` in `ProofSearch`.

---

## Project structure

```
theorem-prover/
├── README.md                   # this file
├── FINDINGS.md                 # system survey (ReProver, DeepSeek-Prover, etc.)
├── setup.sh                    # one-command environment setup
├── requirements.txt
├── test_pipeline.py            # Step 4: baseline test runner
│
├── lean_project/               # Lean 4 project with mathlib dependency
│   ├── lean-toolchain          # pins Lean version
│   ├── lakefile.lean           # declares mathlib dependency
│   └── TestProofs.lean         # verification proofs
│
├── prover/
│   ├── __init__.py             # exports ProofSearch, ProofResult
│   ├── search.py               # best-first proof search
│   ├── tactic_model.py         # ByT5 tactic model wrapper
│   └── lean_interface.py       # LeanDojo wrapper
│
├── data/
│   └── extract.py              # mathlib4 data extraction
│
├── training/
│   └── finetune.py             # fine-tuning scaffold (ready but not run)
│
├── models/
│   └── pretrained/             # downloaded checkpoint (setup.sh puts it here)
│       └── leandojo-lean4-tacgen-byt5-small/
│
├── results/
│   └── baseline_results.json   # test output from test_pipeline.py
│
└── notebooks/
    └── demo.ipynb              # interactive demo
```

---

## Architecture notes

**Tactic model** — `prover/tactic_model.py`  
Wraps `kaiyuy/leandojo-lean4-tacgen-byt5-small`, a ByT5-based seq2seq model.
Input: proof state string. Output: ranked tactic strings via beam search.
ByT5 operates on raw UTF-8 bytes so Lean 4's Unicode math symbols (⊢, ∀, ℕ, …) need no special tokenizer handling.

**Proof search** — `prover/search.py`  
Best-first search over the proof tree. Priority = cumulative negative log-probability of the tactic sequence. Maintains a min-heap; expands the most confident partial proof first.

**Lean interface** — `prover/lean_interface.py`  
Wraps LeanDojo's `Dojo` context manager to apply tactics interactively.
Each `open_proof()` call spawns a Lean subprocess; it is always used as a context manager to guarantee cleanup.

**Fallback** — If you can't get LeanDojo working (version mismatch, etc.), see `prover/tactic_model.py::CausalLMTacticModel` for a prompt-based fallback using any causal LM (e.g., `deepseek-ai/deepseek-math-7b-base`). You lose interactive search but can still do whole-proof generation + verification.

---

## Known issues / gotchas

- **`GITHUB_ACCESS_TOKEN` required** — LeanDojo fetches mathlib from GitHub. Without the token, it will fail with an authentication error.
- **Lean version pinning** — `lean-toolchain` pins `leanprover/lean4:v4.14.0`. If LeanDojo updates its expected version, update this file to match and re-run `lake build`.
- **Subprocess leak** — Always use `LeanInterface.open_proof()` as a context manager. Exiting normally (or via exception) ensures the Lean subprocess is cleaned up.
- **First mathlib build** — `lake build` the first time will compile all of mathlib (~40 min without prebuilt oleans). Run `lake exe cache get` first to pull prebuilt artifacts.
- **MPS (Apple Silicon)** — ByT5 runs on MPS but generation can be slower than CPU for small batch sizes. If you see degraded performance, pass `device="cpu"` to `ProofSearch`.
