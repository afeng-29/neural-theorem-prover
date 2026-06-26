"""
Compare pretrained vs fine-tuned model on end-to-end proof success.

Runs best-first proof search with both model checkpoints on the calculus
theorems in ProofGoals.lean (Group A–D), then prints a per-theorem table
and saves to results/proof_search_comparison.json.

Requires:
    - GITHUB_ACCESS_TOKEN environment variable (for LeanDojo tracing)
    - lake build completed in lean_project/

Usage:
    python scripts/compare_proof_search.py \
        --lean-project lean_project/ \
        --pretrained   models/pretrained/leandojo-lean4-tacgen-byt5-small \
        --finetuned    models/finetuned/calculus/ \
        --timeout      120 \
        --top-k        32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from prover import ProofSearch

# ── Calculus theorems for comparison ──────────────────────────────────────────
# Each entry: (label, theorem_str, hypotheses_list, difficulty)
CALCULUS_THEOREMS: list[tuple[str, str, list[str], str]] = [
    # Group A: Continuity
    ("continuous_const",
     "Continuous (fun _ : ℝ => c)", ["c : ℝ"], "easy"),
    ("continuous_id",
     "Continuous (id : ℝ → ℝ)", [], "easy"),
    ("continuous_add",
     "Continuous (fun x => f x + g x)",
     ["f g : ℝ → ℝ", "hf : Continuous f", "hg : Continuous g"], "easy"),
    ("continuous_mul",
     "Continuous (fun x => f x * g x)",
     ["f g : ℝ → ℝ", "hf : Continuous f", "hg : Continuous g"], "easy"),
    ("continuous_comp",
     "Continuous (f ∘ g)",
     ["f g : ℝ → ℝ", "hf : Continuous f", "hg : Continuous g"], "easy"),
    ("continuous_neg",
     "Continuous (fun x => -f x)",
     ["f : ℝ → ℝ", "hf : Continuous f"], "easy"),
    ("continuousAt_of_continuous",
     "ContinuousAt f x",
     ["f : ℝ → ℝ", "hf : Continuous f", "x : ℝ"], "easy"),
    ("continuousAt_const",
     "ContinuousAt (fun _ : ℝ => c) x",
     ["c x : ℝ"], "easy"),

    # Group B: Differentiability
    ("differentiable_const",
     "Differentiable ℝ (fun _ : ℝ => c)", ["c : ℝ"], "easy"),
    ("differentiable_id",
     "Differentiable ℝ (id : ℝ → ℝ)", [], "easy"),
    ("differentiable_add",
     "Differentiable ℝ (fun x => f x + g x)",
     ["f g : ℝ → ℝ", "hf : Differentiable ℝ f", "hg : Differentiable ℝ g"], "easy"),
    ("differentiable_neg",
     "Differentiable ℝ (fun x => -f x)",
     ["f : ℝ → ℝ", "hf : Differentiable ℝ f"], "easy"),
    ("differentiable_comp",
     "Differentiable ℝ (f ∘ g)",
     ["f g : ℝ → ℝ", "hf : Differentiable ℝ f", "hg : Differentiable ℝ g"], "medium"),
    ("differentiable_mul",
     "Differentiable ℝ (fun x => f x * g x)",
     ["f g : ℝ → ℝ", "hf : Differentiable ℝ f", "hg : Differentiable ℝ g"], "medium"),

    # Group C: HasDerivAt
    ("hasDerivAt_const",
     "HasDerivAt (fun _ => c) 0 x", ["x c : ℝ"], "easy"),
    ("hasDerivAt_id",
     "HasDerivAt id 1 x", ["x : ℝ"], "easy"),
    ("hasDerivAt_add",
     "HasDerivAt (fun x => f x + g x) (f' + g') x",
     ["f g : ℝ → ℝ", "f' g' x : ℝ",
      "hf : HasDerivAt f f' x", "hg : HasDerivAt g g' x"], "easy"),
    ("hasDerivAt_const_mul",
     "HasDerivAt (fun x => c * f x) (c * f') x",
     ["f : ℝ → ℝ", "f' x c : ℝ", "hf : HasDerivAt f f' x"], "medium"),
    ("hasDerivAt_neg",
     "HasDerivAt (fun x => -f x) (-f') x",
     ["f : ℝ → ℝ", "f' x : ℝ", "hf : HasDerivAt f f' x"], "easy"),
    ("hasDerivAt_differentiableAt",
     "DifferentiableAt ℝ f x",
     ["f : ℝ → ℝ", "f' x : ℝ", "hf : HasDerivAt f f' x"], "easy"),
    ("hasDerivAt_deriv",
     "deriv f x = f'",
     ["f : ℝ → ℝ", "f' x : ℝ", "hf : HasDerivAt f f' x"], "medium"),

    # Group D: Filter.Tendsto
    ("tendsto_const",
     "Filter.Tendsto (fun _ => c) l (nhds c)",
     ["c : ℝ", "l : Filter ℝ"], "easy"),
    ("tendsto_of_continuousAt",
     "Filter.Tendsto f (nhds x) (nhds (f x))",
     ["f : ℝ → ℝ", "x : ℝ", "hf : ContinuousAt f x"], "easy"),
    ("tendsto_add",
     "Filter.Tendsto (fun x => f x + g x) l (nhds (a + b))",
     ["f g : ℝ → ℝ", "l : Filter ℝ", "a b : ℝ",
      "hf : Filter.Tendsto f l (nhds a)",
      "hg : Filter.Tendsto g l (nhds b)"], "medium"),
]


def run_search(model_path: str, lean_project: str, timeout: float,
               top_k: int, log_tactics: bool = False,
               model_type: str = "byt5", load_in_4bit: bool = False,
               use_subprocess: bool = False) -> list[dict]:
    prover = ProofSearch(
        model_path=model_path,
        lean_project=lean_project,
        top_k=top_k,
        model_type=model_type,
        load_in_4bit=load_in_4bit,
        use_subprocess=use_subprocess,
    )

    if not use_subprocess:
        # Batch-prepare theorems so LeanDojo traces them in one lake build
        batch_items = [(thm, hyps) for _, thm, hyps, _ in CALCULUS_THEOREMS]
        prover.prepare_theorem_batch(batch_items)

    results = []
    for label, thm, hyps, difficulty in CALCULUS_THEOREMS:
        print(f"  [{label}] ...", end=" ", flush=True)
        r = prover.prove(thm, hyps, timeout=timeout)
        status = "OK" if r.verified else "FAIL"
        print(f"{status} ({r.elapsed_seconds:.0f}s, {r.search_nodes_expanded} nodes)")

        if log_tactics and r.root_tactics:
            n_err = sum(1 for t in r.root_tactics if t["elaboration"] == "error")
            n_ok  = sum(1 for t in r.root_tactics if t["elaboration"] in ("success", "complete"))
            print(f"    root tactics: {len(r.root_tactics)} total, {n_ok} elaborated, {n_err} errored")
            print(f"    top-5: {[t['tactic'] for t in r.root_tactics[:5]]}")

        entry: dict = {
            "label": label,
            "theorem": thm,
            "hypotheses": hyps,
            "difficulty": difficulty,
            "success": r.verified,
            "proof": r.proof,
            "nodes_expanded": r.search_nodes_expanded,
            "elapsed_seconds": r.elapsed_seconds,
            "error": r.error,
        }
        if log_tactics:
            entry["tactics_tried"] = [
                {"tactic": t["tactic"], "log_prob": t["log_prob"]}
                for t in r.root_tactics
            ]
            entry["elaboration_results"] = [
                {k: v for k, v in t.items() if k != "log_prob"}
                for t in r.root_tactics
            ]
        results.append(entry)
    return results


def print_comparison(pre_results: list[dict], ft_results: list[dict]):
    pre_map = {r["label"]: r for r in pre_results}
    ft_map  = {r["label"]: r for r in ft_results}

    header = f"{'Label':<35} {'Pre':>5} {'FT':>5}"
    print("\n" + "=" * 50)
    print("PROOF SEARCH COMPARISON  (pretrained vs fine-tuned)")
    print("=" * 50)
    print(header)
    print("-" * 50)

    pre_total = ft_total = 0
    for label, _, _, difficulty in CALCULUS_THEOREMS:
        p = pre_map[label]["success"]
        f = ft_map[label]["success"]
        pre_total += p
        ft_total  += f
        icon_p = "✓" if p else "✗"
        icon_f = "✓" if f else "✗"
        flag = " ← FT gain" if f and not p else (" ← FT lost" if p and not f else "")
        print(f"  {label:<33} {icon_p:>5} {icon_f:>5}{flag}")

    n = len(CALCULUS_THEOREMS)
    print("-" * 50)
    print(f"  {'TOTAL':<33} {pre_total}/{n:>2} {ft_total}/{n:>2}")
    print(f"  {'Success rate':<33} {pre_total/n:>5.0%} {ft_total/n:>5.0%}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lean-project", default="lean_project/")
    parser.add_argument("--pretrained",
                        default="models/pretrained/leandojo-lean4-tacgen-byt5-small")
    parser.add_argument("--finetuned", default="models/finetuned/calculus/")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--model", choices=["both", "pretrained", "finetuned"],
                        default="both",
                        help="Which model(s) to run (use 'pretrained' or 'finetuned' to run only one)")
    parser.add_argument("--output", default="results/proof_search_comparison.json",
                        help="Path to write JSON results (default: results/proof_search_comparison.json)")
    parser.add_argument("--log-tactics", action="store_true",
                        help="Include tactics_tried and elaboration_results for the root node in output JSON")
    parser.add_argument("--model-type", default="byt5",
                        choices=["byt5", "deepseek", "causal"],
                        help="Model backend: byt5 (default), deepseek (7B), or causal")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load DeepSeek model in 4-bit quantization (fits 16GB GPU)")
    parser.add_argument("--use-subprocess", action="store_true",
                        help="ByT5/causal: skip REPL, try each top-k tactic as 1-step "
                             "proof via lake build subprocess. Clean numbers without REPL.")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    pre_results = ft_results = None

    if args.model in ("both", "pretrained"):
        print("\n" + "=" * 70)
        print(f"Running PRETRAINED model  ({args.pretrained})  [{args.model_type}]")
        print("=" * 70)
        pre_results = run_search(args.pretrained, args.lean_project,
                                 args.timeout, args.top_k, args.log_tactics,
                                 args.model_type, args.load_in_4bit,
                                 args.use_subprocess)

    if args.model in ("both", "finetuned"):
        print("\n" + "=" * 70)
        print(f"Running FINE-TUNED model  ({args.finetuned})  [{args.model_type}]")
        print("=" * 70)
        ft_results = run_search(args.finetuned, args.lean_project,
                                args.timeout, args.top_k, args.log_tactics,
                                args.model_type, args.load_in_4bit,
                                args.use_subprocess)

    if pre_results and ft_results:
        print_comparison(pre_results, ft_results)

    output = {
        "n_theorems": len(CALCULUS_THEOREMS),
        "timeout": args.timeout,
        "top_k": args.top_k,
        "model_type": args.model_type,
    }
    if pre_results:
        n = len(pre_results)
        pre_success = sum(r["success"] for r in pre_results)
        output["pretrained"] = {
            "model": args.pretrained,
            "success": pre_success,
            "total": n,
            "success_rate": pre_success / n,
            "results": pre_results,
        }
    if ft_results:
        n = len(ft_results)
        ft_success = sum(r["success"] for r in ft_results)
        output["finetuned"] = {
            "model": args.finetuned,
            "success": ft_success,
            "total": n,
            "success_rate": ft_success / n,
            "results": ft_results,
        }

    out_path = args.output
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
