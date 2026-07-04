#!/apps/anaconda3/bin/python
"""
miniF2F evaluation using tree-search + high-sample-count whole-proof generation.

Two-phase approach per theorem (same as published DeepSeek-Prover-V1.5-RL protocol):
  Phase 1 — Tree search (BFS, depth ≤ 6, width 8):
    Generate single-tactic candidates, prune invalid branches via batch lake build,
    repeat. Finds structured multi-step proofs efficiently.
  Phase 2 — Whole-proof fallback (if tree search times out):
    Generate up to `--samples` complete proofs and batch-verify.
    Scaled to use the model's full GPU budget.

The published 60.2% uses ~3200 samples total per theorem. We target 400-800
which should reach ~50-55% on one A40 within 20-25 hours.

Usage (batch job):
    grid_run --grid_submit=batch --grid_gpu --grid_mem=60G \\
        /user/af3698/neural-theorem-prover/scripts/run_minif2f_mcts_eval.py \\
        --model-path models/pretrained/deepseek-prover-v1.5-rl \\
        --lean-project lean_project \\
        --output results/minif2f_mcts_eval.json \\
        --split test --samples 400 --tree-timeout 120 --timeout 300

For array jobs (split across multiple GPUs), use --start-idx / --end-idx.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so we can import prover.*
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

MINIF2F_PREAMBLE = """\
import Mathlib
import Aesop

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Finset

"""


def load_minif2f(split: str) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("cat-searcher/minif2f-lean4", split=split)
    problems = []
    for row in ds:
        stmt = row.get("formal_statement", "")
        if not stmt.strip():
            continue
        problems.append({
            "id": row.get("id", row.get("problem_name", "")),
            "formal_statement": stmt,
            "informal_stmt": row.get("informal_stmt", ""),
        })
    logger.info("Loaded %d problems from miniF2F (%s split)", len(problems), split)
    return problems


def parse_formal_statement(formal: str) -> tuple[str, str]:
    """Return (thm_name, body_without_sorry)."""
    body = formal.strip()
    if body.startswith("theorem "):
        body = body[len("theorem "):]
    m = re.match(r"(\S+)\s*(.*)", body, re.DOTALL)
    if not m:
        return "unnamed", body.rstrip(":= sorry").strip()
    name = m.group(1)
    rest = m.group(2).rstrip()
    rest = re.sub(r":=\s*sorry\s*$", "", rest).strip()
    return name, rest


def generate_whole_proofs(model, formal_statement: str, n: int, max_new_tokens: int) -> list[str]:
    """
    Generate n complete proof bodies for formal_statement.
    Returns list of raw proof body strings.
    """
    import torch

    prompt = MINIF2F_PREAMBLE + re.sub(r":=\s*sorry\s*$", ":= by\n  ", formal_statement.strip())

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
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # NEVER reduce to batch=8 — it deadlocks in torch.multinomial on CUDA 12.2.
            # Abort and return whatever proofs were collected so far.
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
    """Extract proof body, stopping at prose or sorry."""
    text = re.sub(r"```[^\n]*\n?", "", text)
    lines = []
    for ln in text.splitlines():
        stripped = ln.strip()
        if not stripped:
            lines.append(ln)
            continue
        if re.search(
            r"(complete the following|lean 4 code|fill in|your answer|solution:|"
            r"step \d+:|^The (theorem|proof|answer)|^Note:)",
            stripped, re.IGNORECASE,
        ):
            break
        if re.search(r'\$[A-Za-z_]', stripped) and re.search(
            r'\b(be|the|of|all|set|define|show|prove|note|let)\b', stripped, re.IGNORECASE
        ):
            break
        non_ascii = sum(1 for c in stripped if ord(c) > 127)
        if non_ascii > len(stripped) * 0.25:
            break
        if re.search(r"\bsorry\b", stripped):
            return ""
        lines.append(ln)
    return "\n".join(lines).rstrip()


def verify_whole_proofs(lean_project: Path, formal_statement: str, proof_bodies: list[str]) -> list[bool]:
    """Batch-verify proof bodies via lake build. Returns per-proof bool list."""
    import subprocess, os

    if not proof_bodies:
        return []

    elan_env = {
        **os.environ,
        "PATH": f"{Path.home() / '.elan' / 'bin'}:{os.environ.get('PATH', '')}",
    }
    base = re.sub(r":=\s*sorry\s*$", "", formal_statement.strip())
    file_lines = [MINIF2F_PREAMBLE.rstrip(), ""]
    ranges: list[tuple[int, int]] = []

    for body in proof_bodies:
        start = len(file_lines) + 1
        file_lines.append(f"{base} := by")
        for line in body.splitlines():
            s = line.strip()
            file_lines.append(f"  {s}" if s else "")
        end = len(file_lines)
        ranges.append((start, end))
        file_lines.append("")

    src = "\n".join(file_lines)
    goals_path = lean_project / "ProofGoals.lean"
    original = goals_path.read_text() if goals_path.exists() else None

    try:
        goals_path.write_text(src)
        proc = subprocess.Popen(
            ["lake", "build", "TheoremProver"],
            cwd=lean_project, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=elan_env, start_new_session=True,
        )
        import signal as _sig
        try:
            raw_out, raw_err = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            try:
                import os as _os
                _os.killpg(proc.pid, _sig.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.communicate()
            logger.warning("Whole-proof verify timed out")
            return [False] * len(proof_bodies)
        out = raw_out.decode(errors="replace") + raw_err.decode(errors="replace")

        if proc.returncode == 0 and "error:" not in out.lower() and "uses 'sorry'" not in out:
            return [True] * len(proof_bodies)

        error_lines: set[int] = set()
        for m in re.finditer(r"error:.*?ProofGoals\.lean:(\d+):\d+:", out):
            error_lines.add(int(m.group(1)))

        return [
            not any(s <= ln <= e for ln in error_lines)
            for s, e in ranges
        ]
    finally:
        if original is not None:
            goals_path.write_text(original)
        elif goals_path.exists():
            goals_path.unlink()


def eval_problem(
    problem: dict,
    model,
    lean_project: Path,
    tree_prover,
    args,
) -> dict:
    """Run tree search + whole-proof fallback for one miniF2F problem."""
    t_start = time.monotonic()
    formal = problem["formal_statement"]

    # ── Phase 1: Tree search ─────────────────────────────────────────────────
    if args.tree_timeout > 0:
        logger.info("  Phase 1: tree search (timeout=%.0fs)", args.tree_timeout)
        result = tree_prover.search(model=model, formal_statement=formal)
        if result.proved:
            elapsed = time.monotonic() - t_start
            logger.info("  PROVED via tree search in %.1fs: %s", elapsed, result.proof[:80])
            return {
                "id": problem["id"],
                "proved": True,
                "proof": result.proof,
                "method": "tree_search",
                "elapsed_seconds": elapsed,
            }

    # ── Phase 2: Whole-proof sampling ────────────────────────────────────────
    remaining_budget = args.timeout - (time.monotonic() - t_start)
    if remaining_budget < 10:
        elapsed = time.monotonic() - t_start
        return {"id": problem["id"], "proved": False, "proof": None, "elapsed_seconds": elapsed}

    logger.info("  Phase 2: whole-proof sampling (up to %d samples)", args.samples)
    # Cap max_new_tokens at 256 for whole-proof: batch=16×256 tokens fits in ~7.5GB KV headroom.
    # Longer sequences (2048) OOM at batch=16, and batch=8 deadlocks on CUDA 12.2.
    whole_proof_max_tokens = min(args.max_new_tokens, 256)
    batch = 16  # match model._generate_batch; verify 16 at a time for lake build stability
    samples_done = 0
    while samples_done < args.samples:
        if time.monotonic() - t_start > args.timeout:
            break
        n = min(batch, args.samples - samples_done)
        proofs = generate_whole_proofs(
            model, formal, n=n, max_new_tokens=whole_proof_max_tokens,
        )
        samples_done += n
        if not proofs:
            continue
        # De-duplicate
        seen: set[str] = set()
        unique = [p for p in proofs if not (p in seen or seen.add(p))]
        results = verify_whole_proofs(lean_project, formal, unique)
        for ok, proof in zip(results, unique):
            if ok:
                elapsed = time.monotonic() - t_start
                logger.info(
                    "  PROVED via sampling (%d samples) in %.1fs: %s",
                    samples_done, elapsed, proof[:80],
                )
                return {
                    "id": problem["id"],
                    "proved": True,
                    "proof": proof,
                    "method": "whole_proof",
                    "samples_tried": samples_done,
                    "elapsed_seconds": elapsed,
                }

    elapsed = time.monotonic() - t_start
    return {
        "id": problem["id"],
        "proved": False,
        "proof": None,
        "method": "whole_proof",
        "samples_tried": samples_done,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/pretrained/deepseek-prover-v1.5-rl")
    parser.add_argument("--lean-project", default="lean_project")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="results/minif2f_mcts_eval.json")
    parser.add_argument("--samples", type=int, default=400,
                        help="Max whole-proof samples per theorem (Phase 2)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max tokens per proof attempt")
    parser.add_argument("--timeout", type=float, default=300,
                        help="Total wall-clock budget per theorem (seconds)")
    parser.add_argument("--tree-timeout", type=float, default=120,
                        help="Budget for tree search phase (0 to disable)")
    parser.add_argument("--tree-width", type=int, default=8,
                        help="Tactic candidates per tree-search expansion")
    parser.add_argument("--tree-depth", type=int, default=6,
                        help="Max tree search depth (tactics)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index for array-job splitting")
    parser.add_argument("--end-idx", type=int, default=-1,
                        help="End index (exclusive) for array-job splitting (-1 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip problems already in --output")
    args = parser.parse_args()

    from prover.tactic_model import DeepSeekProverModel
    from prover.mcts import TreeSearchProver

    lean_project = Path(args.lean_project).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    existing: dict = {}
    if args.resume and output_path.exists():
        try:
            saved = json.loads(output_path.read_text())
            existing = saved.get("results", saved)
            logger.info("Resuming: %d problems already done", len(existing))
        except Exception as e:
            logger.warning("Could not load existing results: %s", e)

    problems = load_minif2f(args.split)

    # Slice for array jobs
    end = len(problems) if args.end_idx < 0 else args.end_idx
    problems = problems[args.start_idx:end]

    logger.info("Evaluating %d problems (indices %d–%d)", len(problems), args.start_idx, end)

    # Load model
    logger.info("Loading DeepSeek-Prover from %s", args.model_path)
    model = DeepSeekProverModel(model_id=args.model_path)
    model._ensure_loaded()

    # 4-bit models can handle larger batches (model ~3.9B vs 7B, more KV headroom)
    # Detect by parameter count: BF16 = ~7B params reported, 4-bit = ~3.9B
    n_params = sum(p.numel() for p in model._model.parameters())
    if n_params < 5e9:
        model._generate_batch = 16
        logger.info("4-bit model detected (%dM params): using batch_size=%d", n_params//1e6, model._generate_batch)

    tree_prover = TreeSearchProver(
        lean_project_path=lean_project,
        width=args.tree_width,
        max_depth=args.tree_depth,
        tree_timeout=args.tree_timeout,
    )

    results: dict = dict(existing)
    t_run_start = time.monotonic()

    for idx, problem in enumerate(problems, 1):
        pid = problem["id"]
        global_idx = idx + args.start_idx
        if pid in results:
            logger.info("[%d/%d] %s — already done (proved=%s)", global_idx, len(problems) + args.start_idx, pid, results[pid].get("proved"))
            continue

        proved_so_far = sum(1 for v in results.values() if v.get("proved"))
        done_so_far = len(results)
        logger.info(
            "[%d/%d] %s  (proved so far: %d/%d = %.1f%%)",
            global_idx, len(problems) + args.start_idx, pid,
            proved_so_far, done_so_far,
            100 * proved_so_far / max(done_so_far, 1),
        )

        res = eval_problem(problem, model, lean_project, tree_prover, args)
        res["informal_stmt"] = problem.get("informal_stmt", "")
        results[pid] = res

        proved_total = sum(1 for v in results.values() if v.get("proved"))
        total_done = len(results)
        logger.info(
            "  → %s  (total: %d/%d proved = %.1f%%)",
            "PROVED" if res["proved"] else "FAILED",
            proved_total, total_done,
            100 * proved_total / max(total_done, 1),
        )

        # Save after each problem
        out_data = {
            "summary": {
                "proved": proved_total,
                "total": total_done,
                "pct": round(100 * proved_total / max(total_done, 1), 2),
            },
            "results": results,
        }
        output_path.write_text(json.dumps(out_data, indent=2))

    proved = sum(1 for v in results.values() if v.get("proved"))
    total = len(results)
    elapsed = time.monotonic() - t_run_start
    print(f"\n=== FINAL: {proved}/{total} = {100*proved/max(total,1):.1f}% ({elapsed/3600:.1f}h) ===")


if __name__ == "__main__":
    main()
