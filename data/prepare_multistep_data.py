#!/apps/anaconda3/bin/python
"""
Prepare multi-step proof training data for DeepSeek LoRA fine-tuning.

Sources (in priority order):
1. miniF2F VALIDATION proofs proved by the base model (67 verified, multi-step competition math)
2. miniF2F TEST proofs proved by the base model (60 verified) — optional, causes test leakage
3. Existing Lean-Workbook training data filtered to multi-step proofs (3+ tactic lines)

Output: data/deepseek_lora_multistep.jsonl
"""
import sys, json, re
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).parent.parent
PREAMBLE = (
    "import Mathlib\nimport Aesop\n"
    "set_option maxHeartbeats 400000\n"
    "open BigOperators Real Nat Topology Finset\n\n"
)

def count_real_lines(proof: str) -> int:
    """Count non-comment, non-blank tactic lines."""
    return len([
        l for l in proof.strip().splitlines()
        if l.strip() and not l.strip().startswith("--")
    ])

def formal_to_prompt(formal_statement: str) -> str | None:
    """
    Convert 'theorem NAME ARGS : GOAL := sorry' to a training prompt.
    Returns None if we can't parse it.
    """
    formal = formal_statement.strip()
    # Strip trailing ':= sorry' / ':=\n  sorry'
    body = re.sub(r":=\s*sorry\s*$", "", formal).strip()
    if not body.startswith("theorem "):
        return None
    return PREAMBLE + body + " := by\n"

def load_minif2f_proofs(result_file: Path, split_label: str) -> list[dict]:
    """Load verified proofs from a minif2f eval result JSON."""
    data = json.loads(result_file.read_text())
    results = data.get("results", data)
    examples = []
    for pid, v in results.items():
        if not isinstance(v, dict) or not v.get("proved") or not v.get("proof"):
            continue
        proof = v["proof"]
        n_lines = count_real_lines(proof)
        examples.append({
            "pid": pid,
            "proof": proof,
            "n_lines": n_lines,
            "source": f"minif2f_{split_label}",
        })
    return examples

def load_minif2f_formal_statements(split: str) -> dict[str, str]:
    """Load formal_statement by problem id from HuggingFace cache."""
    from datasets import load_dataset
    ds = load_dataset("cat-searcher/minif2f-lean4", split=split, trust_remote_code=True)
    return {prob["id"]: prob["formal_statement"] for prob in ds}


def main():
    examples = []

    # ── Source 1: miniF2F validation proofs ──────────────────────────────────
    valid_file = ROOT / "results" / "minif2f_deepseek_valid.json"
    if valid_file.exists():
        valid_proofs = load_minif2f_proofs(valid_file, "valid")
        print(f"Validation proofs: {len(valid_proofs)} proved")
        print(f"  Loading formal statements from dataset...")
        valid_formals = load_minif2f_formal_statements("validation")
        added = 0
        for ep in valid_proofs:
            formal = valid_formals.get(ep["pid"])
            if not formal:
                continue
            prompt = formal_to_prompt(formal)
            if not prompt:
                continue
            examples.append({
                "prompt": prompt,
                "completion": "\n" + ep["proof"].lstrip("\n"),
                "id": ep["pid"],
                "source": ep["source"],
                "n_proof_lines": ep["n_lines"],
            })
            added += 1
        print(f"  Added {added} validation examples")
    else:
        print(f"WARNING: {valid_file} not found, skipping validation proofs")

    # ── Source 2: miniF2F test proofs (comment out to avoid leakage) ─────────
    # test_file = ROOT / "results" / "minif2f_deepseek_base_test.json"
    # if test_file.exists():
    #     test_proofs = load_minif2f_proofs(test_file, "test")
    #     test_formals = load_minif2f_formal_statements("test")
    #     for ep in test_proofs:
    #         formal = test_formals.get(ep["pid"])
    #         if not formal: continue
    #         prompt = formal_to_prompt(formal)
    #         if not prompt: continue
    #         examples.append({"prompt": prompt, "completion": "\n" + ep["proof"].lstrip("\n"),
    #                           "id": ep["pid"], "source": ep["source"],
    #                           "n_proof_lines": ep["n_lines"]})

    # ── Source 3: Lean-Workbook multi-step (3+ real tactic lines) ────────────
    lw_file = ROOT / "data" / "deepseek_lora_train_v2.jsonl"
    if lw_file.exists():
        lw_all = [json.loads(l) for l in lw_file.read_text().splitlines() if l.strip()]
        multi_step = [
            d for d in lw_all
            if count_real_lines(d.get("completion", "")) >= 3
        ]
        print(f"Lean-Workbook multi-step (3+ lines): {len(multi_step)} / {len(lw_all)}")
        for d in multi_step:
            examples.append({
                "prompt": d["prompt"],
                "completion": d["completion"],
                "id": d.get("id", ""),
                "source": "lean_workbook_multistep",
                "n_proof_lines": count_real_lines(d["completion"]),
            })
    else:
        print(f"WARNING: {lw_file} not found, skipping Lean-Workbook")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nTotal training examples: {len(examples)}")
    src_counts = Counter(e["source"] for e in examples)
    for src, n in src_counts.most_common():
        print(f"  {src}: {n}")
    line_dist = Counter(e["n_proof_lines"] for e in examples)
    print(f"\nProof length distribution:")
    for k in sorted(line_dist)[:20]:
        print(f"  {k} lines: {line_dist[k]}")

    out = ROOT / "data" / "deepseek_lora_multistep.jsonl"
    with out.open("w") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
