"""
Compare pretrained vs fine-tuned model on top-k tactic prediction accuracy.

Runs evaluate_top_k() from training/finetune.py on the same 117 test examples
with both model checkpoints, then prints a side-by-side table and saves to
results/tactic_comparison.json.

Usage:
    python scripts/compare_tactic_models.py \
        --test-data data/calculus/test.jsonl \
        --pretrained models/pretrained/leandojo-lean4-tacgen-byt5-small \
        --finetuned  models/finetuned/calculus/ \
        --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.finetune import evaluate_top_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", nargs="+",
                        default=["data/calculus/test.jsonl"])
    parser.add_argument("--pretrained",
                        default="models/pretrained/leandojo-lean4-tacgen-byt5-small")
    parser.add_argument("--finetuned",
                        default="models/finetuned/calculus/")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("Evaluating PRETRAINED model...")
    pre = evaluate_top_k(
        model_path=args.pretrained,
        test_paths=args.test_data,
        k=args.top_k,
        n_samples=args.n_samples,
    )

    print("=" * 70)
    print("Evaluating FINE-TUNED model...")
    ft = evaluate_top_k(
        model_path=args.finetuned,
        test_paths=args.test_data,
        k=args.top_k,
        n_samples=args.n_samples,
    )

    print("\n" + "=" * 70)
    print("TACTIC PREDICTION COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<28} {'Pretrained':>12} {'Fine-tuned':>12} {'Delta':>10}")
    print("-" * 65)

    top1_pre = pre["top1_exact_match"]
    top1_ft  = ft["top1_exact_match"]
    topk_pre = pre[f"top{args.top_k}_exact_match"]
    topk_ft  = ft[f"top{args.top_k}_exact_match"]

    print(f"{'Top-1 exact match':<28} {top1_pre:>12.2%} {top1_ft:>12.2%} {top1_ft-top1_pre:>+10.2%}")
    print(f"{'Top-' + str(args.top_k) + ' exact match':<28} {topk_pre:>12.2%} {topk_ft:>12.2%} {topk_ft-topk_pre:>+10.2%}")
    print(f"{'n_samples':<28} {pre['n_samples']:>12} {ft['n_samples']:>12}")
    print("=" * 70)

    gain_top1 = top1_ft - top1_pre
    gain_topk = topk_ft - topk_pre
    if gain_top1 > 0:
        print(f"\nFine-tuning improved top-1 by {gain_top1:+.2%} ({gain_top1/max(top1_pre,1e-9):.1f}x relative)")
        print(f"Fine-tuning improved top-{args.top_k} by {gain_topk:+.2%}")
    else:
        print(f"\nNo gain from fine-tuning on top-1 ({gain_top1:+.2%})")

    out = {
        "test_data": args.test_data,
        "top_k": args.top_k,
        "pretrained": {
            "model": args.pretrained,
            "top1_exact_match": top1_pre,
            f"top{args.top_k}_exact_match": topk_pre,
            "n_samples": pre["n_samples"],
        },
        "finetuned": {
            "model": args.finetuned,
            "top1_exact_match": top1_ft,
            f"top{args.top_k}_exact_match": topk_ft,
            "n_samples": ft["n_samples"],
        },
        "delta": {
            "top1_exact_match": gain_top1,
            f"top{args.top_k}_exact_match": gain_topk,
        },
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/tactic_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results/tactic_comparison.json")


if __name__ == "__main__":
    main()
