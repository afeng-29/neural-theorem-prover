#!/apps/anaconda3/bin/python
"""
Download Lean-Workbook and prepare training data in the same format
used by DeepSeek-Prover-V1.5 (the published model).

DeepSeek-Prover-V1.5 SFT stage:
  - Source: InformalMath/lean-workbook (140K informal-formal problem pairs)
  - Only problems with VERIFIED Lean 4 proofs are used for SFT
  - Format: preamble + theorem_statement := by\n  <proof_body>

This script:
  1. Downloads Lean-Workbook from HuggingFace
  2. Filters to problems that have a non-sorry Lean 4 proof
  3. Writes JSONL in our training format (same as deepseek_lora_multistep.jsonl)

Output: data/lean_workbook_verified.jsonl
        data/lean_workbook_stats.json
"""

import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n"
    "\n"
    "set_option maxHeartbeats 400000\n"
    "\n"
    "open BigOperators Real Nat Topology Finset\n"
    "\n"
)


def has_sorry(text: str) -> bool:
    return bool(re.search(r"\bsorry\b", text))


def extract_proof_body(lean_proof: str) -> str:
    """
    Extract the tactic body from a full Lean proof string.
    Input might be: 'theorem foo : P := by\n  tactic'
                or: 'by\n  tactic'
                or: just the tactic body
    """
    # Remove preamble-style imports if present
    lines = lean_proof.strip().splitlines()
    body_lines = []
    in_proof = False
    for line in lines:
        if ":= by" in line or line.strip() == "by":
            in_proof = True
            continue
        if in_proof:
            body_lines.append(line)
    if body_lines:
        return "\n".join(body_lines).strip()
    # Fallback: return as-is
    return lean_proof.strip()


def build_training_example(formal_statement: str, proof_body: str) -> dict | None:
    """Build one JSONL training example matching our format."""
    # formal_statement: 'theorem NAME PARAMS : GOAL := sorry'
    # proof_body: the tactic sequence
    if not formal_statement or not proof_body:
        return None
    if has_sorry(proof_body):
        return None

    # Replace ':= sorry' with ':= by\n  <proof>'
    stmt = re.sub(r":=\s*sorry\s*$", "", formal_statement.strip())
    indented = "\n".join(f"  {line}" if line.strip() else "" for line in proof_body.splitlines())
    full_proof = f"{PREAMBLE}{stmt} := by\n{indented}"

    return {
        "prompt": PREAMBLE + stmt + " := by\n  ",
        "completion": proof_body,
        "full_proof": full_proof,
        "source": "lean_workbook",
        "n_lines": len([l for l in proof_body.splitlines() if l.strip()]),
    }


def main():
    out_dir = Path(__file__).parent
    out_path = out_dir / "lean_workbook_verified.jsonl"
    stats_path = out_dir / "lean_workbook_stats.json"

    logger.info("Downloading Lean-Workbook from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("InformalMath/lean-workbook", split="train")
        logger.info("Downloaded %d examples", len(ds))
    except Exception as e:
        logger.error("Failed to download Lean-Workbook: %s", e)
        logger.info("Trying alternative dataset path...")
        try:
            from datasets import load_dataset
            # Alternative: the lean-workbook-plus dataset with verified proofs
            ds = load_dataset("InformalMath/lean-workbook-plus", split="train")
            logger.info("Downloaded %d examples from lean-workbook-plus", len(ds))
        except Exception as e2:
            logger.error("Both dataset paths failed: %s", e2)
            sys.exit(1)

    # Print column names to understand the schema
    logger.info("Dataset columns: %s", ds.column_names)
    if len(ds) > 0:
        logger.info("Sample row keys: %s", list(ds[0].keys()))

    stats = {
        "total": len(ds),
        "has_proof": 0,
        "non_sorry": 0,
        "multi_step": 0,
        "written": 0,
    }

    examples = []
    for row in ds:
        # Try various column names for proof
        proof = (
            row.get("lean4_solution")
            or row.get("lean_solution")
            or row.get("proof")
            or row.get("formal_proof")
            or ""
        )
        formal = (
            row.get("formal_statement")
            or row.get("lean4_statement")
            or row.get("formal")
            or ""
        )

        if not proof or not formal:
            continue

        stats["has_proof"] += 1

        if has_sorry(proof):
            continue

        stats["non_sorry"] += 1

        proof_body = extract_proof_body(proof)
        n_lines = len([l for l in proof_body.splitlines() if l.strip()])

        if n_lines < 1:
            continue

        if n_lines >= 2:
            stats["multi_step"] += 1

        ex = build_training_example(formal, proof_body)
        if ex:
            examples.append(ex)
            stats["written"] += 1

    logger.info("Stats: %s", stats)
    logger.info("Writing %d training examples to %s", len(examples), out_path)

    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    stats_path.write_text(json.dumps(stats, indent=2))

    # Summary by proof length
    from collections import Counter
    length_dist = Counter(ex["n_lines"] for ex in examples)
    print("\nProof length distribution:")
    for length in sorted(length_dist)[:20]:
        print(f"  {length} lines: {length_dist[length]}")

    multi_step = [ex for ex in examples if ex["n_lines"] >= 2]
    print(f"\nTotal: {len(examples)} examples")
    print(f"Multi-step (≥2 lines): {len(multi_step)}")
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
