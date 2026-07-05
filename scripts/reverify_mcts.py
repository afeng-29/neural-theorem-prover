#!/user/af3698/neural-theorem-prover/venv/bin/python
"""
Re-verify all 'proved' results from the MCTS run using a correct single-proof verifier.
Writes a corrected results file and prints the actual score.
"""
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)

LEAN_PROJECT = Path("/user/af3698/neural-theorem-prover/lean_project").resolve()
RESULTS_FILE = Path("/user/af3698/neural-theorem-prover/results/minif2f_mcts_eval.json")
OUTPUT_FILE  = Path("/user/af3698/neural-theorem-prover/results/minif2f_mcts_reverified.json")

PREAMBLE = """\
import Mathlib
import Aesop

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Finset

"""

ELAN_ENV = {
    **os.environ,
    "PATH": f"{Path.home() / '.elan' / 'bin'}:{os.environ.get('PATH', '')}",
}


def verify_one(formal_statement: str, proof: str) -> bool:
    """Run lake build on a single proof and return True if it compiles cleanly."""
    goals_path = LEAN_PROJECT / "ProofGoals.lean"
    original = goals_path.read_text() if goals_path.exists() else None

    base = re.sub(r":=\s*sorry\s*$", "", formal_statement.strip())
    lines = [PREAMBLE.rstrip(), "", f"{base} := by"]
    for ln in proof.splitlines():
        s = ln.strip()
        lines.append(f"  {s}" if s else "")
    src = "\n".join(lines)

    try:
        goals_path.write_text(src)
        result = subprocess.run(
            ["lake", "build", "TheoremProver"],
            cwd=LEAN_PROJECT, capture_output=True, text=True, timeout=120,
            env=ELAN_ENV,
        )
        out = result.stdout + result.stderr

        # Must compile cleanly: no errors, no sorry warnings
        if result.returncode != 0:
            return False
        if re.search(r"ProofGoals\.lean:\d+:\d+:.*?error:", out):
            return False
        if "uses 'sorry'" in out:
            return False
        return True

    except subprocess.TimeoutExpired:
        logger.warning("Timeout verifying proof")
        return False
    except Exception as e:
        logger.warning("Error: %s", e)
        return False
    finally:
        if original is not None:
            goals_path.write_text(original)
        elif goals_path.exists():
            goals_path.unlink()


def main():
    data = json.loads(RESULTS_FILE.read_text())
    results = data.get("results", data)

    # Load miniF2F for formal_statement lookup
    from datasets import load_dataset
    ds = load_dataset("cat-searcher/minif2f-lean4", split="test")
    stmt_map = {row.get("id", row.get("problem_name", "")): row["formal_statement"] for row in ds}

    proved_items = [(k, v) for k, v in results.items() if v.get("proved")]
    logger.info("Re-verifying %d claimed proofs...", len(proved_items))

    confirmed = 0
    rejected = 0
    corrected_results = dict(results)

    for i, (pid, v) in enumerate(proved_items, 1):
        proof = v.get("proof", "")
        formal = stmt_map.get(pid)
        if not formal:
            logger.warning("[%d/%d] %s — no formal statement found, SKIPPING", i, len(proved_items), pid)
            continue

        ok = verify_one(formal, proof)
        if ok:
            confirmed += 1
            status = "OK"
        else:
            rejected += 1
            status = "FALSE POSITIVE"
            corrected_results[pid] = {**v, "proved": False, "reverify": "false_positive"}

        logger.info("[%d/%d] %s — %s | proof: %s", i, len(proved_items), pid, status, repr(proof[:80]))

        # Print running count every 10
        if i % 10 == 0:
            print(f"=== [{i}/{len(proved_items)}] confirmed={confirmed} rejected={rejected} ===", flush=True)

    total = len([v for v in corrected_results.values() if isinstance(v, dict)])
    real_proved = sum(1 for v in corrected_results.values() if isinstance(v, dict) and v.get("proved"))

    print(f"\n=== RE-VERIFICATION COMPLETE ===")
    print(f"Claimed:   {len(proved_items)}/244")
    print(f"Confirmed: {confirmed}/244 = {100*confirmed/244:.1f}%")
    print(f"Rejected:  {rejected} false positives")

    out_data = {
        "summary": {"proved": real_proved, "total": total,
                    "pct": round(100 * real_proved / max(total, 1), 2)},
        "results": corrected_results,
    }
    OUTPUT_FILE.write_text(json.dumps(out_data, indent=2))
    logger.info("Written to %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
