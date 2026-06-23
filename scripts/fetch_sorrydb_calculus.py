"""
Fetch the SorryDB_2601 dataset and filter for analysis/calculus goals.

SorryDB tracks `sorry` placeholders from active Lean 4 math repositories.
We filter by file path keywords (Analysis, Calculus, MeasureTheory, Topology)
to get calculus/analysis-relevant goals for proof search evaluation.

Output: data/sorrydb_calculus.jsonl
Each line: {"id": ..., "repo": ..., "file_path": ..., "expected_type": ..., "state": ...}

Usage:
    python scripts/fetch_sorrydb_calculus.py [--output data/sorrydb_calculus.jsonl]

Requires: pip install datasets (already in requirements.txt via huggingface-hub)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Keywords that identify analysis/calculus files in Mathlib or related repos
CALCULUS_KEYWORDS = [
    "Calculus",
    "Analysis",
    "Deriv",
    "Differential",
    "MeasureTheory",
    "Integral",
    "Topology/Algebra",
    "Continuity",
    "Limit",
    "Tendsto",
    "NormedSpace",
    "Metric",
]

# Tactic-level keywords in the expected type that signal calculus content
TYPE_KEYWORDS = [
    "HasDerivAt",
    "HasFDerivAt",
    "Differentiable",
    "DifferentiableAt",
    "Continuous",
    "ContinuousAt",
    "ContinuousOn",
    "Filter.Tendsto",
    "HasStrictDerivAt",
    "HasStrictFDerivAt",
    "Integrable",
    "MeasureTheory",
    "NormedSpace",
    "IsCompact",
    "IsClosed",
    "IsOpen",
    "nhds",
    "limsup",
    "liminf",
]


def is_calculus_goal(record: dict) -> bool:
    """Return True if this sorry is likely a calculus/analysis goal."""
    file_path = record.get("file_path", "") or record.get("url", "") or ""
    expected_type = record.get("expected_type", "") or record.get("type", "") or ""

    # Match on file path
    for kw in CALCULUS_KEYWORDS:
        if kw.lower() in file_path.lower():
            return True

    # Match on type expression
    for kw in TYPE_KEYWORDS:
        if kw in expected_type:
            return True

    return False


def normalize_record(record: dict, idx: int) -> dict | None:
    """Normalize a SorryDB record into our standard format."""
    # SorryDB_2601 fields vary by version; try multiple key names
    expected_type = (
        record.get("expected_type")
        or record.get("type")
        or record.get("goal")
        or ""
    ).strip()

    if not expected_type:
        return None

    # Build a proof state string: hypotheses (if any) + ⊢ goal
    hypotheses = record.get("hypotheses") or record.get("context") or []
    if isinstance(hypotheses, str):
        hypotheses = [h.strip() for h in hypotheses.split("\n") if h.strip()]

    if hypotheses:
        state = "\n".join(hypotheses) + "\n⊢ " + expected_type
    else:
        state = "⊢ " + expected_type

    return {
        "id": record.get("id") or f"sorrydb_{idx:05d}",
        "repo": record.get("repo") or record.get("repository") or "",
        "file_path": record.get("file_path") or record.get("url") or "",
        "expected_type": expected_type,
        "hypotheses": hypotheses,
        "state": state,
        "raw": record,  # keep for debugging; strip before proof search
    }


def fetch_from_huggingface(output_path: Path):
    """Download SorryDB_2601 from HuggingFace and filter."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets")
        raise

    print("Downloading cat-searcher/SorryDB_2601 from HuggingFace...")
    # SorryDB is small enough to fit in memory
    ds = load_dataset("cat-searcher/SorryDB_2601", split="train", trust_remote_code=True)
    print(f"Total records in SorryDB_2601: {len(ds)}")

    filtered = []
    for idx, record in enumerate(ds):
        if is_calculus_goal(record):
            norm = normalize_record(record, idx)
            if norm:
                filtered.append(norm)

    print(f"Calculus/analysis records: {len(filtered)} / {len(ds)}")
    return filtered


def fetch_from_json(json_path: str, output_path: Path):
    """Load from a local SorryDB JSON dump."""
    with open(json_path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "sorries" in data:
        records = data["sorries"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unexpected JSON structure in {json_path}")

    filtered = [
        norm for idx, r in enumerate(records)
        if is_calculus_goal(r)
        for norm in [normalize_record(r, idx)]
        if norm
    ]
    print(f"Calculus/analysis records: {len(filtered)} / {len(records)}")
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sorrydb_calculus.jsonl")
    parser.add_argument("--local-json", default=None,
                        help="Path to a local SorryDB JSON dump (skip HuggingFace download)")
    parser.add_argument("--max-goals", type=int, default=None,
                        help="Limit to this many goals (for quick testing)")
    parser.add_argument("--show-types", action="store_true",
                        help="Print all unique expected_type prefixes for inspection")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.local_json:
        records = fetch_from_json(args.local_json, Path(args.output))
    else:
        records = fetch_from_huggingface(Path(args.output))

    if args.max_goals:
        records = records[: args.max_goals]

    if args.show_types:
        print("\n--- Expected type prefixes (first 60 chars) ---")
        for r in records[:30]:
            print(" ", r["expected_type"][:60])

    # Write to JSONL (strip 'raw' field to keep output clean)
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            r_out = {k: v for k, v in r.items() if k != "raw"}
            f.write(json.dumps(r_out, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} calculus/analysis goals to {out_path}")
    print(f"\nNext step: run proof search comparison on these goals:")
    print(f"  sbatch scripts/run_sorrydb_eval.sh")


if __name__ == "__main__":
    main()
