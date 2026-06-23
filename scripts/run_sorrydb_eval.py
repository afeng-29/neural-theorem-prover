"""
Run proof search on SorryDB calculus/analysis goals with both models.

Reads data/sorrydb_calculus.jsonl (produced by fetch_sorrydb_calculus.py),
runs best-first search with both the pretrained and fine-tuned models,
and saves results to results/sorrydb_comparison.json.

Usage:
    python scripts/run_sorrydb_eval.py \
        --goals-file data/sorrydb_calculus.jsonl \
        --max-goals  50 \
        --timeout    120 \
        --top-k      32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from prover import ProofSearch


def load_goals(path: str, max_goals: int | None = None) -> list[dict]:
    goals = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("expected_type"):
                goals.append(rec)
            if max_goals and len(goals) >= max_goals:
                break
    return goals


def run_model(model_path: str, goals: list[dict], lean_project: str,
              timeout: float, top_k: int) -> list[dict]:
    prover = ProofSearch(model_path=model_path, lean_project=lean_project, top_k=top_k)

    # Prepare all theorems in one batch for LeanDojo caching
    batch_items = [(g["expected_type"], g.get("hypotheses", [])) for g in goals]
    prover.prepare_theorem_batch(batch_items)

    results = []
    for i, goal in enumerate(goals):
        goal_id = goal.get("id", f"goal_{i:05d}")
        print(f"  [{i+1}/{len(goals)}] {goal_id[:50]}...", end=" ", flush=True)
        r = prover.prove(
            theorem=goal["expected_type"],
            hypotheses=goal.get("hypotheses", []),
            timeout=timeout,
        )
        status = "OK" if r.verified else "FAIL"
        print(f"{status} ({r.elapsed_seconds:.0f}s)")
        results.append({
            "id": goal_id,
            "repo": goal.get("repo", ""),
            "file_path": goal.get("file_path", ""),
            "expected_type": goal["expected_type"],
            "success": r.verified,
            "proof": r.proof,
            "nodes_expanded": r.search_nodes_expanded,
            "elapsed_seconds": r.elapsed_seconds,
            "error": r.error,
        })
    return results


def print_summary(label: str, results: list[dict]):
    n = len(results)
    success = sum(r["success"] for r in results)
    print(f"\n{label}: {success}/{n} proved ({success/n:.1%})")
    if success > 0:
        print("  Proved goals:")
        for r in results:
            if r["success"]:
                print(f"    [{r['id'][:40]}] {r['expected_type'][:60]}")
                print(f"      proof: {r['proof']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goals-file", default="data/sorrydb_calculus.jsonl")
    parser.add_argument("--lean-project", default="lean_project/")
    parser.add_argument("--pretrained",
                        default="models/pretrained/leandojo-lean4-tacgen-byt5-small")
    parser.add_argument("--finetuned", default="models/finetuned/calculus/")
    parser.add_argument("--max-goals", type=int, default=50,
                        help="Limit to first N goals (default 50)")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--model", choices=["both", "pretrained", "finetuned"],
                        default="both")
    args = parser.parse_args()

    goals = load_goals(args.goals_file, args.max_goals)
    print(f"Loaded {len(goals)} SorryDB calculus goals from {args.goals_file}")

    Path("results").mkdir(exist_ok=True)
    output: dict = {
        "goals_file": args.goals_file,
        "n_goals": len(goals),
        "timeout": args.timeout,
        "top_k": args.top_k,
    }

    pre_results = ft_results = None

    if args.model in ("both", "pretrained"):
        print(f"\n{'='*70}")
        print(f"PRETRAINED  ({args.pretrained})")
        print("="*70)
        pre_results = run_model(args.pretrained, goals, args.lean_project,
                                args.timeout, args.top_k)
        print_summary("Pretrained", pre_results)
        n = len(pre_results)
        s = sum(r["success"] for r in pre_results)
        output["pretrained"] = {
            "model": args.pretrained,
            "success": s,
            "total": n,
            "success_rate": s / n,
            "results": pre_results,
        }

    if args.model in ("both", "finetuned"):
        print(f"\n{'='*70}")
        print(f"FINE-TUNED  ({args.finetuned})")
        print("="*70)
        ft_results = run_model(args.finetuned, goals, args.lean_project,
                               args.timeout, args.top_k)
        print_summary("Fine-tuned", ft_results)
        n = len(ft_results)
        s = sum(r["success"] for r in ft_results)
        output["finetuned"] = {
            "model": args.finetuned,
            "success": s,
            "total": n,
            "success_rate": s / n,
            "results": ft_results,
        }

    if pre_results and ft_results:
        pre_s = sum(r["success"] for r in pre_results)
        ft_s  = sum(r["success"] for r in ft_results)
        n = len(goals)
        print(f"\n{'='*70}")
        print("SORRYDB COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"  Pretrained: {pre_s}/{n}  ({pre_s/n:.1%})")
        print(f"  Fine-tuned: {ft_s}/{n}  ({ft_s/n:.1%})")
        print(f"  Delta:      {ft_s-pre_s:+d}  ({(ft_s-pre_s)/n:+.1%})")
        output["delta"] = {
            "success_diff": ft_s - pre_s,
            "success_rate_diff": (ft_s - pre_s) / n,
        }

    out_path = "results/sorrydb_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
