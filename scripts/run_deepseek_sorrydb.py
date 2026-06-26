"""
Run DeepSeek-Prover-V1.5-RL on SorryDB calculus/analysis goals.

Reads data/sorrydb_calculus.jsonl (produced by fetch_sorrydb_calculus.py),
runs whole-proof generation + batch subprocess verification (no REPL),
and saves results to results/sorrydb_deepseek.json.

Usage:
    python scripts/run_deepseek_sorrydb.py \
        --goals-file data/sorrydb_calculus.jsonl \
        --max-goals  100 \
        --timeout    300 \
        --top-k      32 \
        --load-in-4bit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from prover import ProofSearch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_goals(path: str, max_goals: int | None = None) -> list[dict]:
    goals = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("expected_type"):
                goals.append(rec)
            if max_goals and len(goals) >= max_goals:
                break
    return goals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goals-file", default="data/sorrydb_calculus.jsonl")
    parser.add_argument("--lean-project", default="lean_project/")
    parser.add_argument("--model-path",
                        default="models/pretrained/deepseek-prover-v1.5-rl")
    parser.add_argument("--max-goals", type=int, default=100,
                        help="Limit to first N goals (default 100)")
    parser.add_argument("--skip-goals", type=int, default=0,
                        help="Skip the first N goals (for resuming after crash)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-goal timeout in seconds (default 300)")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit (fits 16GB V100)")
    parser.add_argument("--output", default="results/sorrydb_deepseek.json")
    args = parser.parse_args()

    goals = load_goals(args.goals_file, args.max_goals)
    if args.skip_goals:
        goals = goals[args.skip_goals:]
        logger.info("Skipping first %d goals (resuming from goal %d)", args.skip_goals, args.skip_goals + 1)
    logger.info("Loaded %d SorryDB goals from %s", len(goals), args.goals_file)

    prover = ProofSearch(
        model_path=args.model_path,
        lean_project=args.lean_project,
        top_k=args.top_k,
        model_type="deepseek",
        load_in_4bit=args.load_in_4bit,
    )

    Path("results").mkdir(exist_ok=True)
    results = []
    n_success = 0
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.json")

    def _save_checkpoint():
        n = len(results)
        cp = {
            "model": args.model_path, "goals_file": args.goals_file,
            "n_goals_attempted": n, "timeout": args.timeout, "top_k": args.top_k,
            "summary": {"success": n_success, "total": n,
                        "success_rate": n_success / max(n, 1)},
            "results": results,
        }
        checkpoint_path.write_text(json.dumps(cp, indent=2), encoding="utf-8")

    for i, goal in enumerate(goals):
        goal_id = goal.get("id", f"goal_{i:05d}")
        theorem = goal["expected_type"]
        hypotheses = goal.get("hypotheses", [])
        repo = goal.get("repo", "")

        logger.info("=" * 60)
        logger.info("[%d/%d] %s", i + 1, len(goals), goal_id[:60])
        logger.info("Theorem: %s", theorem[:100])
        logger.info("Repo: %s", repo)

        try:
            r = prover.prove(theorem=theorem, hypotheses=hypotheses, timeout=args.timeout)
            verified, proof, elapsed, nodes, error = r.verified, r.proof, r.elapsed_seconds, r.search_nodes_expanded, r.error
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM on goal %d — skipping", i + 1)
            import gc; gc.collect(); torch.cuda.empty_cache()
            verified, proof, elapsed, nodes, error = False, None, 0.0, 0, "OOM"

        status = "SUCCESS" if verified else "FAILED"
        logger.info("Result: %s in %.1fs (%d nodes)", status, elapsed, nodes)
        if proof:
            logger.info("Proof: %s", proof[:120])

        if verified:
            n_success += 1

        results.append({
            "id": goal_id,
            "repo": repo,
            "file_path": goal.get("file_path", ""),
            "expected_type": theorem,
            "hypotheses": hypotheses,
            "success": verified,
            "proof": proof,
            "nodes_expanded": nodes,
            "elapsed_seconds": elapsed,
            "error": error,
        })
        _save_checkpoint()  # write after every goal — safe if SLURM kills the job

    n = len(results)
    logger.info("=" * 60)
    logger.info("SUMMARY: %d / %d proved (%.1f%%)", n_success, n, 100 * n_success / max(n, 1))

    if n_success > 0:
        logger.info("Proved goals:")
        for r in results:
            if r["success"]:
                logger.info("  [%s] %s", r["id"][:40], r["expected_type"][:80])
                logger.info("    proof: %s", str(r["proof"])[:100])

    output = {
        "model": args.model_path,
        "goals_file": args.goals_file,
        "n_goals": n,
        "timeout": args.timeout,
        "top_k": args.top_k,
        "load_in_4bit": args.load_in_4bit,
        "summary": {
            "success": n_success,
            "total": n,
            "success_rate": n_success / max(n, 1),
        },
        "results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
