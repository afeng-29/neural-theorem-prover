"""
Build LoRA training data from DeepSeek's validation-split eval results.

Reads the re-verified validation result JSON, extracts (theorem, proof) pairs
for every problem that was confirmed proved, and writes JSONL in the format:
  {"prompt": "<preamble + formal_statement up to ':= by\n  '>", "completion": "<proof_body>"}
"""
import json, re, sys, logging
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

MINIF2F_PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n"
    "set_option maxHeartbeats 400000\n"
    "open BigOperators Real Nat Topology Finset\n\n"
)


def formal_to_prompt(formal_statement: str) -> str:
    """Convert formal_statement (ends ':= sorry') to generation prompt."""
    return MINIF2F_PREAMBLE + re.sub(r":=\s*sorry\s*$", ":= by\n  ", formal_statement.strip())


def main():
    if len(sys.argv) < 3:
        print("Usage: make_deepseek_train_data.py <result_json> <output_jsonl>")
        sys.exit(1)

    result_file = sys.argv[1]
    out_file = sys.argv[2]

    with open(result_file) as f:
        data = json.load(f)

    results = data["results"]
    proved_pids = [pid for pid, r in results.items() if r.get("proved") and r.get("proof")]
    logger.info("%d proved problems in %s", len(proved_pids), result_file)

    ds = {row["id"]: row for row in load_dataset("cat-searcher/minif2f-lean4", split="validation")}

    written = 0
    with open(out_file, "w") as out:
        for pid in proved_pids:
            r = results[pid]
            prob = ds.get(pid)
            if not prob:
                logger.warning("  %s not in validation dataset — skipping", pid)
                continue
            prompt = formal_to_prompt(prob["formal_statement"])
            completion = r["proof"]
            out.write(json.dumps({"prompt": prompt, "completion": completion,
                                  "id": pid}) + "\n")
            written += 1

    logger.info("Wrote %d training pairs to %s", written, out_file)


if __name__ == "__main__":
    main()
