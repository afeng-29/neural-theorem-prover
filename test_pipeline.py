"""
Step 4: Run baseline tests and save results to results/baseline_results.json.

Usage:
    python test_pipeline.py [--model-path PATH] [--lean-project PATH] [--timeout N]

Requires:
    - GITHUB_ACCESS_TOKEN environment variable
    - Lean project built (cd lean_project && lake build)
    - Model weights at models/pretrained/ (or downloaded on first run)
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEST_THEOREMS: list[tuple[str, list[str], str]] = [
    # ── Easy (likely 1 tactic) ──────────────────────────────────────────────
    ("n + 0 = n",                   ["n : ℕ"],          "nat_add_zero"),
    ("0 + n = n",                   ["n : ℕ"],          "nat_zero_add"),
    ("n + m = m + n",               ["n m : ℕ"],        "nat_add_comm"),
    ("(n + m) + k = n + (m + k)",   ["n m k : ℕ"],      "nat_add_assoc"),
    ("P ∧ Q → Q ∧ P",               ["P Q : Prop"],     "prop_and_comm"),
    ("P → P",                       ["P : Prop"],       "prop_id"),
    # ── Harder (likely 2-4 tactics, multi-step proof trace) ─────────────────
    # Implication chain — needs intro hpq hqr hp; exact hqr (hpq hp)
    ("(P → Q) → (Q → R) → P → R",  ["P Q R : Prop"],   "prop_impl_trans"),
    # Or-commutativity — needs intro + cases + left/right
    ("P ∨ Q → Q ∨ P",              ["P Q : Prop"],     "prop_or_comm"),
    # Nat multiplication by 2 — ring or rw + arithmetic
    ("2 * n = n + n",               ["n : ℕ"],          "nat_mul_two"),
    # Monotonicity of addition — needs linarith / omega / exact Nat.add_le_add_right
    ("n ≤ m → n + k ≤ m + k",      ["n m k : ℕ"],      "nat_add_le_add_right"),
    # Double negation — needs Classical.byContradiction or tauto
    ("¬¬P → P",                     ["P : Prop"],       "prop_double_neg"),
    # Distributive law — intro + rcases + constructor + exact
    ("P ∧ (Q ∨ R) → (P ∧ Q) ∨ (P ∧ R)", ["P Q R : Prop"], "prop_and_distrib"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None,
                        help="Path to model checkpoint or HuggingFace model id")
    parser.add_argument("--lean-project", default="./lean_project",
                        help="Path to the Lean 4 project directory")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Per-theorem timeout in seconds")
    parser.add_argument("--max-depth", type=int, default=20,
                        help="Max proof search depth")
    parser.add_argument("--top-k", type=int, default=32,
                        help="Tactic candidates per step")
    parser.add_argument("--model-type", default="byt5",
                        choices=["byt5", "deepseek", "causal"],
                        help="byt5: ByT5-small ReProver (default); "
                             "deepseek: DeepSeek-Prover-V1.5-RL (7B, needs GPU); "
                             "causal: generic causal LM")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load DeepSeek model in 4-bit quantization (fits 16GB GPU)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip actual Lean interaction (for smoke-testing imports)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip proof search; just verify known-good proofs via subprocess "
                             "(no GITHUB_ACCESS_TOKEN needed; first call takes 2-4 min cold start)")
    args = parser.parse_args()

    # GITHUB_ACCESS_TOKEN is NOT needed — we trace a local repo, not a GitHub URL.
    # LeanDojo only uses it for rate-limited GitHub API calls on remote repos.

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if args.dry_run:
        logger.info("Dry run — skipping Lean interaction, writing stub results")
        _write_stub_results(results_dir)
        return

    from prover import ProofSearch
    from prover.tactic_model import TacticModel

    logger.info("Initializing ProofSearch (model_type=%s)...", args.model_type)
    prover = ProofSearch(
        model_path=args.model_path,
        lean_project=args.lean_project,
        top_k=args.top_k,
        model_type=args.model_type,
        load_in_4bit=args.load_in_4bit,
    )

    if args.verify_only:
        logger.info("Running in --verify-only mode (subprocess, no token needed)")
        _run_verify_only(prover, results_dir, args)
        return

    # Pre-write all theorems into ProofGoals.lean in one commit so every
    # prove() session shares the same cached LeanDojo trace (one 26-min
    # lake build instead of 26 min × N theorems).
    batch_items = [(t, h) for t, h, _lbl in TEST_THEOREMS]
    logger.info("Preparing %d theorems in one batch commit...", len(batch_items))
    prover.prepare_theorem_batch(batch_items)

    all_results = []
    n_success = 0

    for theorem, hypotheses, label in TEST_THEOREMS:
        logger.info("=" * 60)
        logger.info("Theorem [%s]: %s", label, theorem)
        logger.info("Hypotheses: %s", hypotheses)

        result = prover.prove(
            theorem=theorem,
            hypotheses=hypotheses,
            timeout=args.timeout,
            max_depth=args.max_depth,
        )

        status = "SUCCESS" if result.verified else "FAILED"
        if result.verified:
            n_success += 1

        logger.info("Result: %s in %.2fs (%d nodes expanded)",
                    status, result.elapsed_seconds, result.search_nodes_expanded)
        if result.proof:
            logger.info("Proof:\n  %s", result.proof.replace("\n", "\n  "))
        if result.error:
            logger.info("Error: %s", result.error)

        all_results.append({
            "label": label,
            "theorem": theorem,
            "hypotheses": hypotheses,
            "success": result.verified,
            "proof": result.proof,
            "steps": [
                {"state": s, "tactic": t} for s, t in result.steps
            ],
            "nodes_expanded": result.search_nodes_expanded,
            "elapsed_seconds": result.elapsed_seconds,
            "error": result.error,
        })

    # Write results
    output_path = results_dir / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": args.model_path or TacticModel.DEFAULT_MODEL_ID,
                "lean_project": args.lean_project,
                "timeout": args.timeout,
                "max_depth": args.max_depth,
                "summary": {
                    "total": len(TEST_THEOREMS),
                    "success": n_success,
                    "failed": len(TEST_THEOREMS) - n_success,
                    "success_rate": n_success / len(TEST_THEOREMS),
                },
                "results": all_results,
            },
            f,
            indent=2,
        )

    logger.info("=" * 60)
    logger.info("SUMMARY: %d / %d proved", n_success, len(TEST_THEOREMS))
    logger.info("Results saved to %s", output_path)


def _run_verify_only(prover, results_dir: Path, args):
    """
    Verify known-good proofs via subprocess (no GITHUB_ACCESS_TOKEN needed).

    Uses batch verification: all theorems are written to one file and built
    in a single lake build call (~26 min), instead of 26 min per theorem.
    Falls back to individual checks if the batch build fails.
    """
    KNOWN_PROOFS: dict[str, list[str]] = {
        "nat_add_zero": ["simp"],
        "nat_zero_add": ["simp"],
        "nat_add_comm":  ["ring"],
        "nat_add_assoc": ["ring"],
        "prop_and_comm": ["intro h", "exact ⟨h.2, h.1⟩"],
        "prop_id":       ["intro h", "exact h"],
    }

    all_results = []
    n_success = 0

    # Build batch items: (theorem, hypotheses, proof_tactics, thm_name)
    batch_items = []
    for theorem, hypotheses, label in TEST_THEOREMS:
        proof_tactics = KNOWN_PROOFS.get(label, ["sorry"])
        thm_name = label
        batch_items.append((theorem, hypotheses, proof_tactics, thm_name))

    logger.info(
        "Verifying %d theorems via single batch lake build (~26 min)...",
        len(TEST_THEOREMS),
    )
    t_batch_start = time.time()
    batch_results = prover.verify_proofs_batch(batch_items)
    t_batch_elapsed = time.time() - t_batch_start
    logger.info("Batch build finished in %.1fs", t_batch_elapsed)

    for i, (theorem, hypotheses, label) in enumerate(TEST_THEOREMS):
        proof_tactics = KNOWN_PROOFS.get(label, ["sorry"])
        verified = batch_results[i]
        if verified:
            n_success += 1
        status = "SUCCESS" if verified else "FAILED"
        logger.info("  [%s] %s: %s", label, theorem, status)

        all_results.append({
            "label": label,
            "theorem": theorem,
            "hypotheses": hypotheses,
            "success": verified,
            "proof": "\n".join(proof_tactics) if verified else None,
            "steps": [],
            "nodes_expanded": 0,
            "elapsed_seconds": t_batch_elapsed / len(TEST_THEOREMS),
            "error": "" if verified else f"Lean rejected proof: {proof_tactics}",
        })

    output_path = results_dir / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "mode": "verify-only (subprocess)",
                "lean_project": args.lean_project,
                "summary": {
                    "total": len(TEST_THEOREMS),
                    "success": n_success,
                    "failed": len(TEST_THEOREMS) - n_success,
                    "success_rate": n_success / len(TEST_THEOREMS),
                },
                "results": all_results,
            },
            f,
            indent=2,
        )
    logger.info("=" * 60)
    logger.info("SUMMARY: %d / %d verified", n_success, len(TEST_THEOREMS))
    logger.info("Results saved to %s", output_path)


def _write_stub_results(results_dir: Path):
    """Write placeholder results for dry-run / CI smoke test."""
    from prover.tactic_model import TacticModel

    output = {
        "model": TacticModel.DEFAULT_MODEL_ID,
        "lean_project": "./lean_project",
        "timeout": 60.0,
        "max_depth": 20,
        "summary": {"total": len(TEST_THEOREMS), "success": 0, "failed": len(TEST_THEOREMS),
                    "success_rate": 0.0, "note": "dry-run"},
        "results": [
            {
                "label": label,
                "theorem": thm,
                "hypotheses": hyps,
                "success": False,
                "proof": None,
                "steps": [],
                "nodes_expanded": 0,
                "elapsed_seconds": 0.0,
                "error": "dry-run",
            }
            for thm, hyps, label in TEST_THEOREMS
        ],
    }
    path = results_dir / "baseline_results.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Stub results written to %s", path)


if __name__ == "__main__":
    main()
