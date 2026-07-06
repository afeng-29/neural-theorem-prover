"""
MA-ProofBench evaluation — DeepSeek-Prover-V1.5-RL (whole-proof + MCTS) and ByT5.

Dataset: openbmb/MA-ProofBench (200 Lean 4 problems)
  level1: 100 undergraduate problems
  level2: 100 PhD-level problems

Each problem has its own `header` field (import + open statements), so the
preamble is per-problem rather than the fixed miniF2F preamble.

Usage (batch job):
    grid_run --grid_submit=batch --grid_gpu --grid_mem=100G --grid_ncpus=8 \\
        /user/af3698/neural-theorem-prover/scripts/run_ma_proofbench_eval.sh

For ByT5 only (no GPU needed):
    grid_run --grid_submit=batch --grid_mem=20G --grid_ncpus=8 \\
        scripts/run_ma_proofbench_eval.py --model-type byt5-pretrained \\
        --lean-project lean_project --output results/ma_proofbench_byt5.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

ELAN_ENV = {
    **os.environ,
    "PATH": f"{Path.home() / '.elan' / 'bin'}:{os.environ.get('PATH', '')}",
}

# Appended to every per-problem header to allow more elaboration time.
EXTRA_OPTIONS = "\nset_option maxHeartbeats 400000\n"


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_ma_proofbench() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openbmb/MA-ProofBench", split="test")
    problems = []
    for row in ds:
        stmt = row.get("formal_statement", "").strip()
        header = row.get("header", "import Mathlib").strip()

        # The formal_statement field sometimes embeds the header text inline.
        # Normalise: strip any leading import/open lines so we keep only the
        # theorem declaration.
        stmt = _strip_header_prefix(stmt)
        if not stmt:
            continue

        problems.append({
            "id":               row.get("id", ""),
            "level":            row.get("split", ""),   # "level1" or "level2"
            "informal_stmt":    row.get("informal_stmt", ""),
            "formal_statement": stmt,           # theorem ... := sorry
            "header":           header,          # import Mathlib [+ open ...]
            "topic":            row.get("topic", ""),
            "tag":              row.get("tag", ""),
        })
    logger.info("Loaded %d MA-ProofBench problems", len(problems))
    return problems


def _strip_header_prefix(stmt: str) -> str:
    """Remove any leading import/open/set_option lines from a formal_statement."""
    lines = stmt.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("import ") or s.startswith("open ") or s.startswith("set_option "):
            start = i + 1
        elif s.startswith("theorem ") or s.startswith("lemma ") or s.startswith("example"):
            start = i
            break
    return "\n".join(lines[start:]).strip()


def _build_preamble(header: str) -> str:
    """Construct the per-problem preamble: header + maxHeartbeats option."""
    return header.rstrip() + EXTRA_OPTIONS


def _base_stmt(formal_statement: str) -> str:
    """Strip ':= sorry' / ':= by sorry' / ':= by\\n  sorry' to get the bare theorem signature."""
    s = formal_statement.strip()
    # Handle `:= by\n  sorry`, `:= by sorry`, `:= sorry` in any whitespace variant
    s = re.sub(r":=\s*by\s+sorry\s*$", "", s, flags=re.DOTALL).strip()
    s = re.sub(r":=\s*sorry\s*$", "", s).strip()
    return s


# ── Lean verification ─────────────────────────────────────────────────────────

def verify_proofs(
    lean_project: Path,
    preamble: str,
    formal_statement: str,
    proof_bodies: list[str],
    timeout: int = 120,
) -> list[bool]:
    """
    Verify proof_bodies in parallel via 'lake env lean' (one file per body).
    Returns a bool list: True = compiles cleanly with no sorry.
    """
    if not proof_bodies:
        return []

    base = _base_stmt(formal_statement)

    def _verify_one(i: int, body: str) -> bool:
        unique_stmt = re.sub(r"(theorem\s+\S+|lemma\s+\S+)", rf"\1_b{i}", base, count=1)
        stmt_with_by = f"{unique_stmt} := by"
        file_lines = preamble.rstrip().splitlines() + [""]
        for line in stmt_with_by.splitlines():
            file_lines.append(line)
        for ln in body.splitlines():
            s = ln.strip()
            file_lines.append(f"  {s}" if s else "")
        src = "\n".join(file_lines) + "\n"

        safe_i = f"mapb_{i}"
        fpath = lean_project / f"ProofGoals_{safe_i}.lean"
        try:
            fpath.write_text(src)
            result = subprocess.run(
                ["lake", "env", "lean", fpath.name],
                cwd=lean_project, capture_output=True, text=True,
                timeout=timeout, env=ELAN_ENV,
            )
            out = result.stdout + result.stderr
            return result.returncode == 0 and "uses 'sorry'" not in out
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            logger.warning("_verify_one[%d] error: %s", i, e)
            return False
        finally:
            fpath.unlink(missing_ok=True)

    # Cap at 4 workers — each lean process loads ~8-12 GB of Mathlib oleans
    with ThreadPoolExecutor(max_workers=min(4, len(proof_bodies))) as ex:
        return list(ex.map(lambda a: _verify_one(*a), enumerate(proof_bodies)))


# ── DeepSeek whole-proof generation ──────────────────────────────────────────

def deepseek_generate(model, preamble: str, formal_statement: str,
                      n: int, max_new_tokens: int) -> list[str]:
    """Generate n whole-proof bodies for formal_statement using DeepSeek."""
    import torch

    base = _base_stmt(formal_statement)
    prompt = preamble.rstrip() + "\n\n" + base + " := by\n  "

    model._ensure_loaded()
    device = next(model._model.parameters()).device
    inputs = model._tokenizer(
        prompt, return_tensors="pt", max_length=2048, truncation=True,
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    proofs: list[str] = []
    remaining = n
    while remaining > 0:
        bs = min(model._generate_batch, remaining)
        try:
            outputs = model._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                num_return_sequences=bs,
                pad_token_id=model._tokenizer.eos_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            import gc; gc.collect()
            torch.cuda.empty_cache()
            logger.warning("OOM at batch_size=%d — aborting generation", bs)
            break
        for seq in outputs:
            text = model._tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            text = text.replace("Ġ", " ").replace("Ċ", "\n")
            cleaned = _clean_proof(text)
            if cleaned:
                proofs.append(cleaned)
        remaining -= bs
    return proofs


def _clean_proof(text: str) -> str:
    text = re.sub(r"```[^\n]*\n?", "", text)
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            lines.append(ln)
            continue
        if re.search(
            r"(complete the following|lean 4 code|fill in|your answer|solution:|"
            r"step \d+:|^The (theorem|proof|answer)|^Note:)",
            s, re.IGNORECASE,
        ):
            break
        if re.search(r"\bsorry\b", s):
            return ""
        lines.append(ln)
    return "\n".join(lines).rstrip()


# ── ByT5 tactic generation ────────────────────────────────────────────────────

FALLBACK_TACTICS = [
    "norm_num", "ring", "omega", "simp", "linarith", "nlinarith", "aesop",
    "field_simp", "continuity", "measurability", "positivity",
    "exact?", "apply?", "simp [*]", "norm_cast", "push_neg", "contrapose!",
]


def byt5_generate_tactics(model, formal_statement: str, top_k: int) -> list[str]:
    from prover.tactic_model import CausalLMTacticModel
    if not isinstance(model, CausalLMTacticModel):
        # ByT5 tactic model
        proof_state = formal_statement  # rough approximation
        candidates = model.predict_tactics(proof_state, top_k=top_k)
        return [c.tactic for c in candidates]
    return []


# ── Per-problem evaluation ────────────────────────────────────────────────────

def eval_problem(problem: dict, model, lean_project: Path, tree_prover, args) -> dict:
    t_start = time.monotonic()
    pid = problem["id"]
    formal = problem["formal_statement"]
    preamble = _build_preamble(problem["header"])

    # ── Phase 1: MCTS tree search (DeepSeek only) ─────────────────────────
    if args.model_type == "deepseek" and args.tree_timeout > 0:
        logger.info("  Phase 1: tree search (timeout=%.0fs)", args.tree_timeout)
        # Update prover preamble for this problem
        tree_prover._preamble = preamble
        result = tree_prover.search(model=model, formal_statement=formal)
        if result.proved:
            elapsed = time.monotonic() - t_start
            logger.info("  PROVED via tree search in %.1fs", elapsed)
            return {
                "id": pid, "proved": True, "proof": result.proof,
                "method": "tree_search", "level": problem["level"],
                "elapsed_seconds": elapsed,
            }

    # ── Phase 2: Whole-proof sampling (DeepSeek) or single tactic (ByT5) ─
    remaining = args.timeout - (time.monotonic() - t_start)
    if remaining < 10:
        return _not_proved(problem, time.monotonic() - t_start)

    if args.model_type == "deepseek":
        logger.info("  Phase 2: whole-proof sampling (up to %d samples)", args.samples)
        batch = 16
        samples_done = 0
        while samples_done < args.samples:
            if time.monotonic() - t_start > args.timeout:
                break
            n = min(batch, args.samples - samples_done)
            proofs = deepseek_generate(
                model, preamble, formal, n=n, max_new_tokens=min(args.max_new_tokens, 512),
            )
            samples_done += n
            if not proofs:
                continue
            seen: set[str] = set()
            unique = [p for p in proofs if not (p in seen or seen.add(p))]
            results = verify_proofs(lean_project, preamble, formal, unique, timeout=90)
            for ok, proof in zip(results, unique):
                if ok:
                    elapsed = time.monotonic() - t_start
                    logger.info("  PROVED via sampling (%d samples) in %.1fs", samples_done, elapsed)
                    return {
                        "id": pid, "proved": True, "proof": proof,
                        "method": "whole_proof", "level": problem["level"],
                        "elapsed_seconds": elapsed,
                    }

    elif args.model_type in ("byt5-pretrained", "byt5-ft"):
        logger.info("  ByT5: generating %d tactic candidates", args.top_k)
        tactics = byt5_generate_tactics(model, formal, top_k=args.top_k)
        tactics = list(dict.fromkeys(tactics + FALLBACK_TACTICS))
        results = verify_proofs(lean_project, preamble, formal, tactics, timeout=60)
        for ok, tac in zip(results, tactics):
            if ok:
                elapsed = time.monotonic() - t_start
                logger.info("  PROVED via tactic '%s' in %.1fs", tac, elapsed)
                return {
                    "id": pid, "proved": True, "proof": tac,
                    "method": "tactic", "level": problem["level"],
                    "elapsed_seconds": elapsed,
                }

    return _not_proved(problem, time.monotonic() - t_start)


def _not_proved(problem: dict, elapsed: float) -> dict:
    return {
        "id": problem["id"], "proved": False, "proof": None,
        "method": None, "level": problem["level"],
        "elapsed_seconds": elapsed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["deepseek", "byt5-pretrained", "byt5-ft"],
                        default="deepseek")
    parser.add_argument("--model-path", default="models/pretrained/deepseek-prover-v1.5-rl")
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--lean-project", default="lean_project")
    parser.add_argument("--output", default="results/ma_proofbench_eval.json")
    parser.add_argument("--samples", type=int, default=200,
                        help="Whole-proof samples per theorem (Phase 2, DeepSeek only)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300,
                        help="Total wall-clock budget per theorem (seconds)")
    parser.add_argument("--tree-timeout", type=float, default=120,
                        help="Budget for MCTS tree search (0 to disable)")
    parser.add_argument("--tree-width", type=int, default=8)
    parser.add_argument("--tree-depth", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=32,
                        help="Tactics per ByT5 call")
    parser.add_argument("--level", choices=["level1", "level2", "all"], default="all",
                        help="Which difficulty tier to evaluate")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    lean_project = Path(args.lean_project).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if args.resume and output_path.exists():
        try:
            saved = json.loads(output_path.read_text())
            existing = saved.get("results", saved)
            logger.info("Resuming: %d problems already done", len(existing))
        except Exception as e:
            logger.warning("Could not load existing results: %s", e)

    problems = load_ma_proofbench()
    if args.level != "all":
        problems = [p for p in problems if p["level"] == args.level]
    logger.info("Evaluating %d problems (level=%s)", len(problems), args.level)

    # Load model
    if args.model_type == "deepseek":
        from prover.tactic_model import DeepSeekProverModel
        model = DeepSeekProverModel(model_id=args.model_path, lora_adapter=args.lora_adapter)
        model._ensure_loaded()
        n_params = sum(p.numel() for p in model._model.parameters())
        if n_params < 5e9:
            model._generate_batch = 16
            logger.info("4-bit model (%dM params): batch_size=%d", n_params // 1e6, model._generate_batch)
    elif args.model_type == "byt5-pretrained":
        from prover.tactic_model import TacticModel
        model = TacticModel.load_pretrained(args.model_path)
    else:
        from prover.tactic_model import TacticModel
        model = TacticModel.load_finetuned(args.model_path)

    tree_prover = None
    if args.model_type == "deepseek" and args.tree_timeout > 0:
        from prover.mcts import TreeSearchProver
        tree_prover = TreeSearchProver(
            lean_project_path=lean_project,
            width=args.tree_width,
            max_depth=args.tree_depth,
            tree_timeout=args.tree_timeout,
        )

    results: dict = dict(existing)
    t_run = time.monotonic()

    for idx, problem in enumerate(problems, 1):
        pid = str(problem["id"])
        level = problem["level"]
        if pid in results:
            proved = results[pid].get("proved", False)
            logger.info("[%d/%d] %s (%s) — already done (proved=%s)", idx, len(problems), pid, level, proved)
            continue

        logger.info("[%d/%d] %s (%s) %s", idx, len(problems), pid, level,
                    problem["informal_stmt"][:60])
        try:
            result = eval_problem(problem, model, lean_project, tree_prover, args)
        except Exception as e:
            logger.warning("Error on %s: %s", pid, e)
            result = _not_proved(problem, 0.0)

        results[pid] = result

        proved_l1 = sum(1 for v in results.values() if isinstance(v, dict) and v.get("proved") and v.get("level") == "level1")
        proved_l2 = sum(1 for v in results.values() if isinstance(v, dict) and v.get("proved") and v.get("level") == "level2")
        done_l1 = sum(1 for v in results.values() if isinstance(v, dict) and v.get("level") == "level1")
        done_l2 = sum(1 for v in results.values() if isinstance(v, dict) and v.get("level") == "level2")
        logger.info(
            "  → %s | L1: %d/%d  L2: %d/%d  (%.0fs elapsed)",
            "PROVED" if result["proved"] else "failed",
            proved_l1, done_l1, proved_l2, done_l2,
            time.monotonic() - t_run,
        )

        # Checkpoint every 10 problems
        if idx % 10 == 0:
            _save(output_path, results)

    _save(output_path, results)

    # Final summary
    all_proved = [v for v in results.values() if isinstance(v, dict) and v.get("proved")]
    l1_proved = sum(1 for v in all_proved if v.get("level") == "level1")
    l2_proved = sum(1 for v in all_proved if v.get("level") == "level2")
    total_l1 = sum(1 for v in results.values() if isinstance(v, dict) and v.get("level") == "level1")
    total_l2 = sum(1 for v in results.values() if isinstance(v, dict) and v.get("level") == "level2")

    print("\n=== MA-ProofBench Results ===")
    print(f"Level 1 (undergrad): {l1_proved}/{total_l1} = {100*l1_proved/max(total_l1,1):.1f}%")
    print(f"Level 2 (PhD):       {l2_proved}/{total_l2} = {100*l2_proved/max(total_l2,1):.1f}%")
    print(f"Overall:             {l1_proved+l2_proved}/{total_l1+total_l2} = {100*(l1_proved+l2_proved)/max(total_l1+total_l2,1):.1f}%")
    print(f"Results written to: {output_path}")


def _save(path: Path, results: dict):
    proved = sum(1 for v in results.values() if isinstance(v, dict) and v.get("proved"))
    total = len([v for v in results.values() if isinstance(v, dict)])
    out = {
        "summary": {
            "proved": proved, "total": total,
            "pct": round(100 * proved / max(total, 1), 2),
            "level1": {
                "proved": sum(1 for v in results.values() if isinstance(v, dict) and v.get("proved") and v.get("level") == "level1"),
                "total":  sum(1 for v in results.values() if isinstance(v, dict) and v.get("level") == "level1"),
            },
            "level2": {
                "proved": sum(1 for v in results.values() if isinstance(v, dict) and v.get("proved") and v.get("level") == "level2"),
                "total":  sum(1 for v in results.values() if isinstance(v, dict) and v.get("level") == "level2"),
            },
        },
        "results": results,
    }
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
