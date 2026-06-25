"""
Sanity check: what tactics does the analysis model generate for a trivial goal?

Goal: "c : ℝ\n⊢ Continuous (fun _ : ℝ => c)"  (continuous_const theorem)

Usage:
    python scripts/sanity_check_tactics.py [--model models/finetuned/analysis/]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from prover.tactic_model import TacticModel
from prover.lean_interface import format_state_for_model

KNOWN_CORRECT = ["exact continuous_const", "fun_prop", "continuity",
                 "exact fun _ => rfl", "simp"]

GOAL = "c : ℝ\n⊢ Continuous (fun _ : ℝ => c)"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/finetuned/analysis/",
                        help="Path to model checkpoint (default: analysis FT)")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    model = TacticModel(model_path=args.model, device="cpu")
    model_input = format_state_for_model(GOAL)

    print(f"Model : {args.model}")
    print(f"Goal  : {GOAL!r}")
    print(f"Input : {model_input!r}")
    print()

    candidates = model.predict_tactics(model_input, top_k=args.top_k, num_beams=args.top_k)

    print(f"Top-{args.top_k} generated tactics:")
    print(f"{'Rank':<6} {'LogProb':>9}  Tactic")
    print("-" * 70)
    found_correct: list[str] = []
    for i, c in enumerate(candidates, 1):
        marker = " ✓" if c.tactic in KNOWN_CORRECT else ""
        print(f"  {i:<4} {c.log_prob:>9.4f}  {c.tactic}{marker}")
        if c.tactic in KNOWN_CORRECT:
            found_correct.append(c.tactic)

    print()
    if found_correct:
        print(f"FOUND correct tactics in top-{args.top_k}: {found_correct}")
    else:
        print(f"NONE of the known-correct tactics appeared in top-{args.top_k}.")
        print(f"Known-correct set: {KNOWN_CORRECT}")

    # Also check partial matches (e.g. "exact continuous_const" as substring)
    all_tactics = [c.tactic for c in candidates]
    partial = []
    for kc in KNOWN_CORRECT:
        for t in all_tactics:
            if kc in t or t in kc:
                partial.append((kc, t))
    if partial:
        print(f"Partial matches (known ⊆ generated or vice versa): {partial}")


if __name__ == "__main__":
    main()
