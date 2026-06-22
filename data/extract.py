"""
Step 5: Data extraction from mathlib4 using LeanDojo.

Traces all proofs in a given mathlib4 module and extracts (state, tactic, next_state)
triples for fine-tuning.

Usage:
    python data/extract.py --module Mathlib.Data.Nat.Basic --output data/nat_basic.jsonl

Output format (one JSON object per line):
    {
        "theorem_name": "Nat.add_comm",
        "statement": "∀ (n m : ℕ), n + m = m + n",
        "hypotheses": ["n : ℕ", "m : ℕ"],
        "steps": [
            {"state": "n m : ℕ\n⊢ n + m = m + n", "tactic": "ring", "next_state": ""},
        ],
        "full_proof": "ring",
        "source_module": "Mathlib.Data.Nat.Basic"
    }

Requires:
    - GITHUB_ACCESS_TOKEN environment variable
    - pip install lean-dojo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ProofStep:
    state: str
    tactic: str
    next_state: str


@dataclass
class TheoremRecord:
    theorem_name: str
    statement: str
    hypotheses: list[str]
    steps: list[dict]       # list of ProofStep as dicts
    full_proof: str
    source_module: str


def extract_module(
    module_name: str,
    lean_project_path: str | Path = "./lean_project",
    max_theorems: Optional[int] = None,
) -> list[TheoremRecord]:
    """
    Trace all proofs in `module_name` and return TheoremRecord objects.

    module_name:       Dotted mathlib4 module path, e.g. "Mathlib.Data.Nat.Basic"
    lean_project_path: Path to a Lake project that depends on mathlib4.
    max_theorems:      If set, stop after this many theorems (for quick tests).
    """
    if not os.environ.get("GITHUB_ACCESS_TOKEN"):
        raise EnvironmentError(
            "GITHUB_ACCESS_TOKEN is not set. LeanDojo needs it to access mathlib4.\n"
            "Get a token at https://github.com/settings/tokens (no special scopes needed).\n"
            "Then: export GITHUB_ACCESS_TOKEN=<your_token>"
        )

    try:
        from lean_dojo import LeanGitRepo, trace
    except ImportError:
        raise ImportError("lean_dojo not installed. Run: pip install lean-dojo")

    lean_project_path = Path(lean_project_path).resolve()
    logger.info("Extracting module: %s", module_name)
    logger.info("Lean project: %s", lean_project_path)

    # Trace the module — LeanDojo replays all proofs and records (state, tactic) pairs.
    # This is the slow step: first call downloads and caches the traced repo.
    logger.info("Tracing module (may take several minutes on first run)...")
    t_trace_start = time.monotonic()

    repo = LeanGitRepo(
        url="https://github.com/leanprover-community/mathlib4",
        commit="HEAD",   # LeanDojo will pin to a specific commit internally
    )
    traced_repo = trace(repo, dst_dir=str(lean_project_path / ".leandojo_cache"))

    logger.info("Trace complete in %.1fs", time.monotonic() - t_trace_start)

    # Find the traced file corresponding to the module
    module_file = module_name.replace(".", "/") + ".lean"
    traced_file = traced_repo.get_traced_file(module_file)

    if traced_file is None:
        raise ValueError(f"Module file not found in traced repo: {module_file}")

    records: list[TheoremRecord] = []
    failures = 0

    theorems = traced_file.get_traced_theorems()
    logger.info("Found %d theorems in %s", len(theorems), module_name)

    for i, traced_thm in enumerate(theorems):
        if max_theorems is not None and i >= max_theorems:
            break

        thm_name = traced_thm.theorem.full_name
        logger.debug("Processing theorem %d/%d: %s", i + 1, len(theorems), thm_name)

        try:
            record = _extract_theorem(traced_thm, module_name)
            if record is not None:
                records.append(record)
        except Exception as e:
            logger.warning("Failed to extract %s: %s", thm_name, e)
            failures += 1

    logger.info(
        "Extraction complete: %d theorems extracted, %d failures",
        len(records), failures,
    )
    return records


def _extract_theorem(traced_thm, module_name: str) -> Optional[TheoremRecord]:
    """Extract a TheoremRecord from a LeanDojo TracedTheorem object."""
    thm = traced_thm.theorem
    thm_name = thm.full_name

    # get_traced_tactics() returns List[TracedTactic] with .state_before,
    # .state_after (both plain strings in LeanDojo 4.x), and .tactic (str).
    if not traced_thm.has_tactic_proof():
        logger.debug("  %s: no tactic proof (term-mode or axiom)", thm_name)
        return None

    steps_raw = traced_thm.get_traced_tactics()

    if not steps_raw:
        logger.debug("  %s: no traced tactics", thm_name)
        return None

    steps = []
    tactics = []
    for step in steps_raw:
        state_str = step.state_before or ""
        next_state_str = step.state_after or ""
        tactic_str = step.tactic.strip()
        tactics.append(tactic_str)
        steps.append(ProofStep(
            state=state_str,
            tactic=tactic_str,
            next_state=next_state_str,
        ))

    # Parse hypotheses from the initial state
    hypotheses = _parse_hypotheses(steps[0].state if steps else "")
    # The statement is the initial goal (after ⊢)
    statement = _parse_goal(steps[0].state if steps else "")
    full_proof = "\n".join(tactics)

    return TheoremRecord(
        theorem_name=thm_name,
        statement=statement,
        hypotheses=hypotheses,
        steps=[asdict(s) for s in steps],
        full_proof=full_proof,
        source_module=module_name,
    )


def _parse_hypotheses(state: str) -> list[str]:
    """
    Extract hypothesis lines from a Lean proof state string.
    State format:
        h1 : T1
        h2 : T2
        ⊢ goal
    """
    lines = state.strip().splitlines()
    hyps = []
    for line in lines:
        line = line.strip()
        if line.startswith("⊢") or not line:
            break
        if " : " in line:
            hyps.append(line)
    return hyps


def _parse_goal(state: str) -> str:
    """Extract the goal (text after ⊢) from a proof state string."""
    for line in state.strip().splitlines():
        line = line.strip()
        if line.startswith("⊢"):
            return line[1:].strip()
    return state.strip()


def save_jsonl(records: list[TheoremRecord], output_path: str | Path):
    """Write records to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), output_path)


def print_summary(records: list[TheoremRecord], module_name: str):
    """Print extraction statistics."""
    if not records:
        print(f"\nNo records extracted from {module_name}")
        return

    proof_lengths = [len(r.steps) for r in records]
    avg_len = sum(proof_lengths) / len(proof_lengths)
    max_len = max(proof_lengths)
    min_len = min(proof_lengths)

    print(f"\n{'=' * 50}")
    print(f"Module: {module_name}")
    print(f"Theorems extracted: {len(records)}")
    print(f"Proof length (steps): avg={avg_len:.1f}, min={min_len}, max={max_len}")
    print(f"Total (state, tactic) pairs: {sum(proof_lengths)}")
    print(f"Sample theorem names:")
    for r in records[:5]:
        print(f"  {r.theorem_name} ({len(r.steps)} steps)")
    print(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract mathlib4 proof data using LeanDojo")
    parser.add_argument(
        "--module", default="Mathlib.Data.Nat.Basic",
        help="Mathlib4 module to extract (e.g. Mathlib.Data.Nat.Basic)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSONL file path (default: data/<module_stem>.jsonl)"
    )
    parser.add_argument(
        "--lean-project", default="./lean_project",
        help="Path to the Lean 4 project directory"
    )
    parser.add_argument(
        "--max-theorems", type=int, default=None,
        help="Maximum number of theorems to extract (for quick tests)"
    )
    args = parser.parse_args()

    if args.output is None:
        stem = args.module.split(".")[-1].lower()
        args.output = f"data/{stem}.jsonl"

    records = extract_module(
        module_name=args.module,
        lean_project_path=args.lean_project,
        max_theorems=args.max_theorems,
    )

    print_summary(records, args.module)

    if records:
        save_jsonl(records, args.output)
    else:
        logger.warning("No records extracted — output file not written.")
        sys.exit(1)


if __name__ == "__main__":
    main()
