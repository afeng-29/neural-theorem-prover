"""
Fetch the SorryDB_2601 dataset and filter for analysis/calculus goals.

SorryDB tracks `sorry` placeholders from active Lean 4 math repositories.
Data lives in the SorryDB GitHub repo (not HuggingFace):
  https://github.com/SorryDB/SorryDB/tree/master/data/SorryDB_2601/

We download the full dataset JSON, filter by file path keywords
(Analysis, Calculus, MeasureTheory, Topology, Deriv), extract the proof
state from debug_info.goal, and write to a flat JSONL.

Output: data/sorrydb_calculus.jsonl
Each line: {"id": ..., "repo": ..., "file_path": ..., "state": ...,
            "expected_type": ..., "hypotheses": [...]}

Usage:
    python scripts/fetch_sorrydb_calculus.py [--output data/sorrydb_calculus.jsonl]
"""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.request
from pathlib import Path

SORRYDB_FULL_URL = (
    "https://raw.githubusercontent.com/SorryDB/SorryDB/master/"
    "data/SorryDB_2601/SorryDB_2601.json"
)
SORRYDB_EVAL_URL = (
    "https://raw.githubusercontent.com/SorryDB/SorryDB/master/"
    "data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json"
)

# File path substrings that identify analysis/calculus content
CALCULUS_PATH_KEYWORDS = [
    "Calculus",
    "Analysis",
    "Deriv",
    "Differential",
    "MeasureTheory",
    "Integral",
    "Topology",
    "Continuity",
    "NormedSpace",
    "Metric",
]

# Proof-state keywords (fallback when path doesn't match)
CALCULUS_GOAL_KEYWORDS = [
    "HasDerivAt",
    "HasFDerivAt",
    "Differentiable",
    "DifferentiableAt",
    "Continuous",
    "ContinuousAt",
    "ContinuousOn",
    "Filter.Tendsto",
    "Integrable",
    "MeasureTheory",
    "limsup",
    "liminf",
    "nhds",
]


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    # Try cluster cert bundle first, fall back to system defaults
    for bundle in [
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/certs/ca-certificates.crt",
    ]:
        if Path(bundle).exists():
            ctx.load_verify_locations(bundle)
            return ctx
    return ctx


def fetch_json(url: str) -> dict:
    print(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "sorrydb-fetch/1.0"})
    with urllib.request.urlopen(req, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode("utf-8"))


def is_calculus(record: dict) -> bool:
    path = record.get("location", {}).get("path", "")
    for kw in CALCULUS_PATH_KEYWORDS:
        if kw in path:
            return True
    goal = record.get("debug_info", {}).get("goal", "")
    for kw in CALCULUS_GOAL_KEYWORDS:
        if kw in goal:
            return True
    return False


def parse_goal(goal_str: str) -> tuple[str, list[str]]:
    """
    Split a LeanDojo-style proof state into (expected_type, hypotheses).

    The goal string looks like:
        h1 : T1\nh2 : T2\n⊢ expected_type
    """
    lines = goal_str.strip().split("\n")
    goal_line_idx = next(
        (i for i, l in enumerate(lines) if l.lstrip().startswith("⊢")), len(lines) - 1
    )
    hyps = [l.strip() for l in lines[:goal_line_idx] if l.strip()]
    goal_part = lines[goal_line_idx].lstrip()
    expected_type = goal_part[1:].strip() if goal_part.startswith("⊢") else goal_part
    return expected_type, hyps


def normalize(record: dict) -> dict | None:
    goal_str = record.get("debug_info", {}).get("goal", "").strip()
    if not goal_str:
        return None
    expected_type, hyps = parse_goal(goal_str)
    if not expected_type:
        return None
    return {
        "id": record.get("id", ""),
        "repo": record.get("repo", {}).get("remote", ""),
        "file_path": record.get("location", {}).get("path", ""),
        "url": record.get("debug_info", {}).get("url", ""),
        "state": goal_str,
        "expected_type": expected_type,
        "hypotheses": hyps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sorrydb_calculus.jsonl")
    parser.add_argument("--eval-split", action="store_true",
                        help="Use the 1000-sorry evaluation split instead of full 2601")
    parser.add_argument("--max-goals", type=int, default=None)
    parser.add_argument("--show-types", action="store_true",
                        help="Print first 30 expected_type values for inspection")
    args = parser.parse_args()

    url = SORRYDB_EVAL_URL if args.eval_split else SORRYDB_FULL_URL
    data = fetch_json(url)

    # Schema: {"documentation": "...", "sorries": [...]}
    sorries = data.get("sorries", data) if isinstance(data, dict) else data
    print(f"Total sorries in dataset: {len(sorries)}")

    filtered = []
    for rec in sorries:
        if is_calculus(rec):
            norm = normalize(rec)
            if norm:
                filtered.append(norm)

    print(f"Calculus/analysis sorries: {len(filtered)} / {len(sorries)}")

    if args.max_goals:
        filtered = filtered[: args.max_goals]

    if args.show_types:
        print("\n--- Sample expected_types ---")
        for r in filtered[:20]:
            print(f"  [{r['file_path'].split('/')[-1]}] {r['expected_type'][:80]}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in filtered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(filtered)} goals to {args.output}")


if __name__ == "__main__":
    main()
