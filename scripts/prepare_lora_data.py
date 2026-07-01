"""
Prepare LoRA training data for DeepSeek-Prover from two sources:
  1. internlm/Lean-Workbook: ~10K proved Lean4 theorems (single-tactic whole proofs)
  2. Our verified miniF2F proofs (from results JSON files)

Output: data/deepseek_lora_train_v2.jsonl
Format: {"prompt": "<preamble + theorem stmt>", "completion": "<proof body>", "id": "..."}
"""
import json, re, argparse
from pathlib import Path

PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n"
    "set_option maxHeartbeats 400000\n"
    "open BigOperators Real Nat Topology Finset\n\n"
)


def load_lean_workbook():
    from datasets import load_dataset
    print("Loading internlm/Lean-Workbook ...")
    ds = load_dataset("internlm/Lean-Workbook", split="train", trust_remote_code=True)

    examples = []
    for ex in ds:
        if ex["status"] != "proved":
            continue
        if ex["state_after"].strip() != "no goals":
            continue  # keep only single-tactic whole proofs

        formal = ex["formal_statement"].strip()
        tactic = ex["tactic"].strip()
        eid = ex["id"]

        # Strip the trailing := by sorry and build the prompt
        # Strip the trailing `:= by sorry` (may have varying whitespace)
        formal_clean, n_subs = re.subn(r"\s*:=\s*by\s+sorry\s*$", "", formal)
        formal_clean = formal_clean.strip()
        if n_subs == 0:
            # Pattern not found — can't parse cleanly, skip
            continue

        prompt = PREAMBLE + formal_clean + " := by\n"
        completion = "  " + tactic

        examples.append({"prompt": prompt, "completion": completion, "id": eid,
                          "source": "lean_workbook"})

    print(f"  Lean-Workbook: {len(examples)} single-tactic proofs")
    return examples


def load_minif2f_proofs(result_files):
    from datasets import load_dataset
    print("Loading miniF2F dataset for formal statements ...")
    ds = {}
    for split in ("test", "validation"):
        for row in load_dataset("cat-searcher/minif2f-lean4", split=split):
            ds[row["id"]] = row

    examples = []
    for fpath in result_files:
        p = Path(fpath)
        if not p.exists():
            print(f"  SKIP (not found): {fpath}")
            continue
        data = json.loads(p.read_text())
        for pid, r in data["results"].items():
            if not r.get("proved"):
                continue
            proof = r.get("proof") or ""
            if not proof.strip():
                continue
            row = ds.get(pid)
            if not row:
                continue
            formal = row["formal_statement"].strip()
            # Build prompt: preamble + statement ending at ":= by\n"
            formal_clean = re.sub(r"\s*:=\s*by\b.*$", "", formal, flags=re.DOTALL).strip()
            prompt = PREAMBLE + formal_clean + " := by\n"
            # Indent completion body
            body = proof.strip()
            if not body.startswith(" "):
                body = "  " + body.replace("\n", "\n  ")
            examples.append({"prompt": prompt, "completion": body, "id": pid,
                              "source": p.stem})
        print(f"  {p.stem}: {sum(1 for r in data['results'].values() if r.get('proved'))} proved loaded")

    print(f"  miniF2F total: {len(examples)} proofs")
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/deepseek_lora_train_v2.jsonl")
    parser.add_argument("--no-lean-workbook", action="store_true")
    parser.add_argument("--result-files", nargs="*", default=[
        "results/minif2f_deepseek_base_test.json",
        "results/minif2f_deepseek_valid.json",
        "results/minif2f_byt5_ft_test.json",
    ])
    args = parser.parse_args()

    all_examples = []

    if not args.no_lean_workbook:
        all_examples.extend(load_lean_workbook())

    all_examples.extend(load_minif2f_proofs(args.result_files))

    # Deduplicate by id
    seen = set()
    deduped = []
    for ex in all_examples:
        if ex["id"] not in seen:
            seen.add(ex["id"])
            deduped.append(ex)

    # Shuffle for good mixing
    import random
    random.seed(42)
    random.shuffle(deduped)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ex in deduped:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(deduped)} training examples to {out}")
    sources = {}
    for ex in deduped:
        sources[ex["source"]] = sources.get(ex["source"], 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()
