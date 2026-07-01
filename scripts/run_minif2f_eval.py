"""
Evaluate theorem provers on miniF2F-Lean4 benchmark.

Dataset: cat-searcher/minif2f-lean4 (244 test, 244 validation problems)
Each problem: AMC/AIME/IMO competition math formalized in Lean 4 with `sorry`.

Models supported:
  deepseek      DeepSeek-Prover-V1.5-RL (whole-proof generation)
  byt5-pretrained  ByT5-small pretrained (1-step tactic)
  byt5-ft          ByT5 analysis FT (1-step tactic)

Verification: subprocess lake build (no REPL), same as 24-theorem benchmark.

Usage:
  python scripts/run_minif2f_eval.py --model-type deepseek --split test \\
      --lean-project lean_project/ --model-path models/pretrained/deepseek-prover-v1.5-rl \\
      --top-k 8 --timeout 300 --output results/minif2f_deepseek_test.json
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── miniF2F preamble (same as _DEEPSEEK_PROVER_HEADER in tactic_model.py) ────
MINIF2F_PREAMBLE = """\
import Mathlib
import Aesop

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Finset

"""

# ── Fallback tactics to try for ByT5 (common competition-math closers) ───────
FALLBACK_TACTICS = [
    "norm_num", "ring", "omega", "decide", "simp", "linarith", "aesop",
    "field_simp; ring", "nlinarith", "positivity", "norm_cast; norm_num",
    "simp [mul_comm]", "simp [add_comm]", "ring_nf; norm_num",
]


# ── Lean file building and verification ───────────────────────────────────────

def _build_lean_file(problems: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """
    Build a single Lean file with N theorems (one per proof candidate).

    Each `problems` entry has:
      thm_name: str
      formal_body: str  — everything between `theorem name` and `:= sorry`,
                           e.g. "(b h v : ℝ) ... : v = 65"
      proof_body: str   — indented proof (may be multi-line)

    Returns (lean_src, thm_ranges) where thm_ranges[i] = (start_line, end_line).
    """
    lines: list[str] = MINIF2F_PREAMBLE.splitlines()
    lines.append("")  # blank after preamble
    ranges: list[tuple[int, int]] = []
    for p in problems:
        start = len(lines) + 1  # 1-indexed
        thm_header = f"theorem {p['thm_name']} {p['formal_body']} := by"
        for hline in thm_header.splitlines():
            lines.append(hline)
        for tline in p["proof_body"].splitlines():
            lines.append("  " + tline.lstrip())
        end = len(lines)
        ranges.append((start, end))
        lines.append("")
    return "\n".join(lines), ranges


def verify_candidates(
    candidates: list[dict],
    lean_project: Path,
    timeout: int = 120,
) -> list[bool]:
    """
    Verify N proof candidates in a single lake build call.
    Each candidate: {thm_name, formal_body, proof_body}
    Returns list[bool] parallel to candidates.
    """
    if not candidates:
        return []

    goals_path = lean_project / "ProofGoals.lean"
    original = goals_path.read_text(encoding="utf-8") if goals_path.exists() else ""

    lean_src, thm_ranges = _build_lean_file(candidates)
    try:
        goals_path.write_text(lean_src, encoding="utf-8")

        elan_bin = Path.home() / ".elan" / "bin"
        env = {**os.environ, "PATH": f"{elan_bin}:{os.environ.get('PATH', '')}"}

        result = subprocess.run(
            ["lake", "build", "TheoremProver"],
            cwd=lean_project,
            capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        output = result.stdout + result.stderr

        # Lean 4 error recovery can insert sorry implicitly (producing exit 0 + warning,
        # not an error). Reject any build that uses sorry even without a compile error.
        if result.returncode == 0 and "error:" not in output.lower() and "uses 'sorry'" not in output:
            return [True] * len(candidates)

        error_lines: set[int] = set()
        for m in re.finditer(r"error:.*?ProofGoals\.lean:(\d+):\d+:", output):
            error_lines.add(int(m.group(1)))

        if not error_lines:
            logger.debug("lake build failed but no line numbers parsed; output: %s", output[:400])
            return [False] * len(candidates)

        results = []
        for start, end in thm_ranges:
            results.append(not any(start <= ln <= end for ln in error_lines))
        return results

    except subprocess.TimeoutExpired:
        logger.warning("lake build timed out after %ds", timeout)
        return [False] * len(candidates)
    except Exception as e:
        logger.warning("lake build error: %s", e)
        return [False] * len(candidates)
    finally:
        goals_path.write_text(original, encoding="utf-8")


# ── Parse miniF2F formal_statement ────────────────────────────────────────────

def parse_formal_statement(formal: str) -> tuple[str, str]:
    """
    Split 'theorem NAME BODY := sorry' into (NAME, BODY_without_sorry).
    BODY includes params and hypotheses up to (but not including) ':= sorry'.
    """
    formal = formal.strip()
    # Remove trailing `:= sorry` or `:=\n  sorry`
    body = re.sub(r":=\s*sorry\s*$", "", formal).strip()
    # Extract theorem name
    m = re.match(r"theorem\s+(\S+)\s*(.*)", body, re.DOTALL)
    if m:
        return m.group(1), m.group(2).strip()
    return "unknown", body


def formal_to_proof_state(formal: str) -> str:
    """
    Construct an approximate initial proof state string for ByT5.
    Parses theorem args and goal from the formal_statement.
    """
    thm_name, body = parse_formal_statement(formal)
    # Find the goal: last ":" not inside parens/brackets before ":= sorry"
    # Strip the theorem name prefix to get the parameter/goal part
    # Simple heuristic: last `:` at depth 0
    depth = 0
    last_colon = -1
    for i, ch in enumerate(body):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            last_colon = i

    if last_colon == -1:
        return body  # fallback

    params_part = body[:last_colon].strip()
    goal_part = body[last_colon + 1:].strip()

    # Parse individual params like "(b h v : ℝ)" "(h₀ : ...)"
    state_lines = []
    for pm in re.finditer(r"\(([^()]+)\)", params_part):
        state_lines.append(pm.group(1).strip())
    state_lines.append(f"⊢ {goal_part}")
    return "\n".join(state_lines)


# ── DeepSeek proof generation ─────────────────────────────────────────────────

def deepseek_generate_from_formal(
    model,
    formal: str,
    n: int,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> list[str]:
    """
    Generate n proof bodies for a miniF2F formal_statement using DeepSeek.
    Returns list of raw proof body strings (may be multi-line).
    """
    import torch
    # Build prompt: preamble + formal_statement with `:= sorry` → `:= by\n  `
    prompt_text = MINIF2F_PREAMBLE + re.sub(r":=\s*sorry\s*$", ":= by\n  ", formal.strip())

    model._ensure_loaded()
    device = next(model._model.parameters()).device
    inputs = model._tokenizer(
        prompt_text, return_tensors="pt",
        max_length=2048, truncation=True,
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
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=bs,
                pad_token_id=model._tokenizer.eos_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if bs == 1:
                logger.warning("OOM even with batch_size=1, aborting generation")
                break
            model._generate_batch = max(1, model._generate_batch // 2)
            logger.warning("OOM: reducing batch_size to %d", model._generate_batch)
            continue
        for seq in outputs:
            text = model._tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            # LlamaTokenizerFast byte-level BPE: Ġ=space (U+0120), Ċ=newline (U+010A).
            # decode() sometimes leaves these as literal chars — convert explicitly.
            text = text.replace('Ġ', ' ').replace('Ċ', '\n')
            proof = _clean_deepseek_proof(text)
            if proof:
                proofs.append(proof)
        remaining -= bs

    return proofs


def _clean_deepseek_proof(text: str) -> str:
    """Extract and clean proof body from DeepSeek's generated text."""
    # Remove Markdown code fences
    text = re.sub(r"```[^\n]*\n?", "", text)
    lines = []
    for ln in text.splitlines():
        stripped = ln.strip()
        if not stripped:
            lines.append(ln)  # preserve blank lines (indentation structure)
            continue
        # Stop at clear meta-commentary or English prose (not valid Lean)
        if re.search(r"(complete the following|lean 4 code|fill in|your answer|solution:|step \d+:|^The (theorem|proof|answer)|^Note:)", stripped, re.IGNORECASE):
            break
        # Stop at lines that use LaTeX $ notation with English surrounding words
        if re.search(r'\$[A-Za-z_]', stripped) and re.search(r'\b(be|the|of|all|set|define|show|prove|note|let)\b', stripped, re.IGNORECASE):
            break
        # Stop at lines with excessive non-ASCII (raw BPE markers or natural language)
        non_ascii = sum(1 for c in stripped if ord(c) > 127)
        if non_ascii > len(stripped) * 0.25:
            break
        # Stop at sorry
        if re.search(r"\bsorry\b", stripped):
            return ""
        lines.append(ln)
    return "\n".join(lines).rstrip()


# ── ByT5 proof generation ─────────────────────────────────────────────────────

def byt5_generate_tactics(
    model,
    formal: str,
    top_k: int,
) -> list[str]:
    """
    Generate top-k tactic candidates for ByT5 given a miniF2F formal_statement.
    Returns list of tactic strings to try as 1-step proofs.
    """
    proof_state = formal_to_proof_state(formal)
    candidates = model.predict_tactics(proof_state, top_k=top_k)
    # Combine model candidates with universal fallbacks
    all_tactics = [c.tactic for c in candidates if c.tactic]
    # Add fallbacks not already present
    seen = set(all_tactics)
    for t in FALLBACK_TACTICS:
        if t not in seen:
            all_tactics.append(t)
    return all_tactics


# ── Main evaluation loop ───────────────────────────────────────────────────────

def load_checkpoint(output_path: Path) -> dict:
    if output_path.exists():
        try:
            return json.loads(output_path.read_text())
        except Exception:
            pass
    return {}


def save_results(output_path: Path, results: dict, summary: dict):
    output_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))


def run_eval(args):
    from datasets import load_dataset

    lean_project = Path(args.lean_project).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading miniF2F (%s split)...", args.split)
    ds = load_dataset("cat-searcher/minif2f-lean4", split=args.split, trust_remote_code=True)
    problems = list(ds)
    if args.n_problems:
        problems = problems[:args.n_problems]
    logger.info("Evaluating %d problems", len(problems))

    # Load checkpoint (for --resume)
    checkpoint = load_checkpoint(output_path) if args.resume else {}
    done = checkpoint.get("results", {})

    # Load model
    logger.info("Loading model: %s", args.model_type)
    model = _load_model(args)

    results: dict = dict(done)
    n_proved = sum(1 for r in results.values() if r["proved"])

    for idx, prob in enumerate(problems):
        pid = prob["id"]
        if pid in results:
            logger.info("[%d/%d] %s — SKIPPED (resume)", idx + 1, len(problems), pid)
            continue

        formal = prob["formal_statement"]
        thm_name, formal_body = parse_formal_statement(formal)

        # Skip problems whose formal statement is entirely commented out — they produce
        # a parse error at the theorem header, causing Lean to stop and falsely pass
        # all subsequent candidates in the same batch file.
        if formal_body.strip().startswith("--"):
            logger.info("[%d/%d] %s — SKIPPED (commented-out formal statement)", idx + 1, len(problems), pid)
            results[pid] = {"id": pid, "proved": False, "proof": None, "elapsed_seconds": 0.0,
                            "informal_stmt": prob.get("informal_stmt", "")}
            continue

        logger.info("[%d/%d] %s", idx + 1, len(problems), pid)

        t0 = time.time()
        proved = False
        winning_proof = None

        try:
            if args.model_type == "deepseek":
                proof_bodies = deepseek_generate_from_formal(
                    model, formal, n=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                )
                if proof_bodies:
                    candidates = [
                        {"thm_name": f"{thm_name}_{i}", "formal_body": formal_body, "proof_body": pb}
                        for i, pb in enumerate(proof_bodies)
                    ]
                    successes = verify_candidates(candidates, lean_project, timeout=args.timeout)
                    for ok, cand in zip(successes, candidates):
                        if ok:
                            proved = True
                            winning_proof = cand["proof_body"]
                            break

            else:  # byt5-pretrained or byt5-ft
                tactics = byt5_generate_tactics(model, formal, top_k=args.top_k)
                if tactics:
                    candidates = [
                        {"thm_name": f"{thm_name}_{i}", "formal_body": formal_body, "proof_body": t}
                        for i, t in enumerate(tactics)
                    ]
                    successes = verify_candidates(candidates, lean_project, timeout=args.timeout)
                    for ok, cand in zip(successes, candidates):
                        if ok:
                            proved = True
                            winning_proof = cand["proof_body"]
                            break

        except Exception as e:
            logger.warning("Error on %s: %s", pid, e)

        # Re-verify any batch-reported proof with a single-candidate build to eliminate
        # false positives caused by multi-line predictions or other batch artifacts.
        if proved and winning_proof is not None:
            rechk = [{"thm_name": f"{thm_name}_rechk", "formal_body": formal_body, "proof_body": winning_proof}]
            try:
                ok_single = verify_candidates(rechk, lean_project, timeout=args.timeout)
                if not ok_single[0]:
                    logger.warning("  !! Batch false positive caught: %s  proof: %s", pid, winning_proof[:80])
                    proved = False
                    winning_proof = None
            except Exception as e2:
                logger.warning("  !! Re-verify error for %s: %s — marking failed", pid, e2)
                proved = False
                winning_proof = None

        elapsed = time.time() - t0
        if proved:
            n_proved += 1
            logger.info("  PROVED in %.1fs  proof: %s", elapsed, winning_proof[:80])
        else:
            logger.info("  FAILED in %.1fs", elapsed)

        results[pid] = {
            "id": pid,
            "proved": proved,
            "proof": winning_proof,
            "elapsed_seconds": elapsed,
            "informal_stmt": prob.get("informal_stmt", ""),
        }

        # Print running summary every 10 problems
        total_done = len(results)
        if total_done % 10 == 0 or total_done == len(problems):
            rate = n_proved / total_done * 100
            logger.info(
                "=== PROGRESS [%d/%d] proved=%d (%.1f%%) ===",
                total_done, len(problems), n_proved, rate,
            )

        # Save checkpoint after every problem
        summary = {
            "model_type": args.model_type,
            "split": args.split,
            "n_attempted": total_done,
            "n_proved": n_proved,
            "pass_rate": n_proved / total_done if total_done else 0.0,
        }
        save_results(output_path, results, summary)

    total = len(results)
    rate = n_proved / total * 100 if total else 0
    logger.info("FINAL: %d/%d proved (%.1f%%)", n_proved, total, rate)
    return n_proved, total


def _load_model(args):
    """Load the appropriate model backend."""
    if args.model_type == "deepseek":
        from prover.tactic_model import DeepSeekProverModel
        model = DeepSeekProverModel(
            model_id=args.model_path,
            load_in_4bit=args.load_in_4bit,
            lora_adapter=args.lora_adapter,
        )
        model._ensure_loaded()
        return model

    else:  # byt5
        from prover.tactic_model import TacticModel
        model = TacticModel(model_path=args.model_path)
        model._ensure_loaded()
        return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate on miniF2F benchmark")
    parser.add_argument("--model-type", required=True,
                        choices=["deepseek", "byt5-pretrained", "byt5-ft"],
                        help="Which model to evaluate")
    parser.add_argument("--model-path", required=True,
                        help="Path to model (pretrained or finetuned)")
    parser.add_argument("--lean-project", default="lean_project/",
                        help="Path to Lean 4 project directory")
    parser.add_argument("--split", default="test", choices=["test", "validation"],
                        help="miniF2F split to evaluate")
    parser.add_argument("--n-problems", type=int, default=None,
                        help="Limit to first N problems (default: all)")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Number of proof candidates per problem")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max tokens for DeepSeek generation")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Seconds per lake build call")
    parser.add_argument("--output", required=True,
                        help="Output JSON path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load DeepSeek in 4-bit NF4 (reduces VRAM)")
    parser.add_argument("--lora-adapter", default=None,
                        help="Path to QLoRA LoRA adapter directory (optional)")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
