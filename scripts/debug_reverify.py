#!/user/af3698/neural-theorem-prover/venv/bin/python
"""Debug why verify_one rejects valid proofs."""
import json, os, re, subprocess
from pathlib import Path

LEAN_PROJECT = Path("/user/af3698/neural-theorem-prover/lean_project").resolve()
ELAN_ENV = {
    **os.environ,
    "PATH": f"{Path.home() / '.elan' / 'bin'}:{os.environ.get('PATH', '')}",
}
PREAMBLE = """import Mathlib
import Aesop

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Finset

"""

# Test with a trivial proof
goals_path = LEAN_PROJECT / "ProofGoals.lean"
original = goals_path.read_text() if goals_path.exists() else None

src = PREAMBLE + "theorem test_norm_num : (1 : ℕ) + 1 = 2 := by\n  norm_num\n"
print(f"Writing:\n{src}")
goals_path.write_text(src)

result = subprocess.run(
    ["lake", "build", "TheoremProver"],
    cwd=LEAN_PROJECT, capture_output=True, text=True, timeout=120,
    env=ELAN_ENV,
)
print(f"\nreturncode: {result.returncode}")
print(f"stdout:\n{result.stdout[:500]}")
print(f"stderr:\n{result.stderr[:500]}")

has_error_regex = bool(re.search(r"ProofGoals\.lean:\d+:\d+:.*?error:", result.stdout + result.stderr))
print(f"\nregex match (old style): {has_error_regex}")

if original is not None:
    goals_path.write_text(original)
