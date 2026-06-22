"""
Download and filter the LeanDojo Mathlib tactic dataset for calculus theorems.

Source: cat-searcher/leandojo-benchmark-4-random (HuggingFace)
  - train: 250,814 examples  →  ~6,734 calculus
  - validation:   4,260     →  ~53 calculus
  - test:         4,506     →  ~117 calculus

Output: data/calculus/train.jsonl, val.jsonl, test.jsonl
Each line: {"state": "...", "tactic": "...", "full_name": "...", "file_path": "..."}

To add more domains later, run this script again with --domain pointing at a
different Mathlib path prefix (e.g. Algebra, NumberTheory, Topology).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_ID = "cat-searcher/leandojo-benchmark-4-random"

DOMAIN_FILTERS = {
    "calculus":      lambda fp: "Calculus" in fp,
    "algebra":       lambda fp: "Algebra" in fp and "Calculus" not in fp,
    "number_theory": lambda fp: "NumberTheory" in fp,
    "topology":      lambda fp: "Topology" in fp,
    "analysis":      lambda fp: "Analysis" in fp,
    "all":           lambda fp: True,
}


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d examples → %s", len(records), path)


def filter_and_convert(split_name: str, domain_fn) -> list[dict]:
    logger.info("Loading %s split from %s ...", split_name, DATASET_ID)
    ds = load_dataset(DATASET_ID, split=split_name)
    logger.info("  Total: %d — filtering for domain ...", len(ds))

    records = []
    for ex in ds:
        if domain_fn(ex["file_path"]):
            records.append({
                "state":     ex["state"],
                "tactic":    ex["tactic"],
                "full_name": ex["full_name"],
                "file_path": ex["file_path"],
            })

    logger.info("  After filter: %d examples", len(records))
    return records


def print_stats(records: list[dict], split: str) -> None:
    module_counts = Counter(
        r["file_path"].split("/")[-1].replace(".lean", "") for r in records
    )
    logger.info("--- %s top modules ---", split)
    for mod, cnt in module_counts.most_common(10):
        logger.info("  %4d  %s", cnt, mod)


def main():
    parser = argparse.ArgumentParser(description="Prepare tactic data by domain")
    parser.add_argument(
        "--domain",
        default="calculus",
        choices=list(DOMAIN_FILTERS.keys()),
        help="Which Mathlib domain to extract (default: calculus)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/calculus",
        help="Directory for output JSONL files",
    )
    args = parser.parse_args()

    domain_fn = DOMAIN_FILTERS[args.domain]
    out_dir = Path(args.output_dir)

    splits = {"train": "train", "val": "validation", "test": "test"}
    for out_name, hf_name in splits.items():
        records = filter_and_convert(hf_name, domain_fn)
        print_stats(records, out_name)
        save_jsonl(records, out_dir / f"{out_name}.jsonl")

    logger.info("Done. Data saved to %s/", out_dir)
    logger.info("Next step:")
    logger.info("  python training/finetune.py \\")
    logger.info("    --train-data %s/train.jsonl \\", out_dir)
    logger.info("    --val-data   %s/val.jsonl \\", out_dir)
    logger.info("    --test-data  %s/test.jsonl \\", out_dir)
    logger.info("    --base-model models/pretrained/leandojo-lean4-tacgen-byt5-small \\")
    logger.info("    --output-dir models/finetuned/calculus/")


if __name__ == "__main__":
    main()
