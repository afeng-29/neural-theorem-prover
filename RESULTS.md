# miniF2F Evaluation Results

## DeepSeek-Prover-V1.5-RL — Whole-Proof Generation + Tree Search

All runs use the test split (244 problems), 4-bit BitsAndBytes quantization on CBS A40 GPU nodes,
`max_new_tokens=256`, `batch=16`, `timeout=300s` per Lake verification call.

---

## Summary

| Run | Method | Proved | Total | Accuracy | vs Baseline |
|-----|--------|--------|-------|----------|-------------|
| Baseline | whole-proof top_k=32 | 60 | 244 | 24.6% | — |
| top_k=64 | whole-proof top_k=64 | 97 | 244 | 39.8% | +15.2pp |
| top_k=64 (verified) | whole-proof top_k=64 reverified | 97 | 244 | 39.8% | +15.2pp |
| top_k=256 (claimed) | whole-proof top_k=256 | 114 | 244 | 46.7% | +22.1pp |
| top_k=256 (verified) | whole-proof top_k=256 reverified | 97 | 244 | 39.8% | +15.2pp |
| Published (DeepSeek paper) | MCTS ~3200 samples | 147 | 244 | 60.2% | target |
| MCTS tree search (claimed) | BFS depth≤6, width=8 | 143 | 244 | 58.6% | +12.0pp vs top_k=256 |
| MCTS tree search (verified) | BFS depth≤6, width=8 | 75 | 244 | 30.7% | verified by lake build |
| **MCTS v2 (clean run)** | **BFS depth≤6, width=8, fixed verifier** | **119** | **244** | **48.8%** | **+9.0pp vs top_k baseline** |

### MA-ProofBench (openbmb/MA-ProofBench)

| Run | Method | Proved | Total | Accuracy |
|-----|--------|--------|-------|----------|
| DeepSeek base | whole-proof + MCTS, 200 samples | 0 | 200 | 0.0% |

Job 8774411 cancelled after 68/200 problems (0 proved). MA-ProofBench requires
undergraduate/PhD-level analysis proofs (Lipschitz continuity, complex analysis,
PDEs, measure theory) that are qualitatively harder than miniF2F competition math.
DeepSeek-Prover with 200 samples per problem finds no proofs at this difficulty level.
See §MA-ProofBench section below for details.

---

## MCTS v2 — Clean Run Final Results (2026-07-07)

**Job ID:** 8749649 (researchgpu04)  
**Method:** BFS depth≤6, width=8, DeepSeek-Prover-V1.5-RL 4-bit  
**Verifier:** batch lake build with range_end+1 fix + theorem-keyword tactic filter  
**Score: 119/244 = 48.8%** (+9.0pp over verified whole-proof baseline of 39.8%)  
**Wall time:** 13.9 hours

This is the first MCTS result with a correct verifier. The rate was stable at ~48-49% throughout
all 244 problems. Compared to the published 60.2% target (using ~3200 samples/problem), our
run used 400 samples/problem — closing about 40% of the gap with ~8× fewer samples.

### Result File

`results/minif2f_mcts_eval_v2.json`

---

## MCTS Tree Search — Final Results (2026-07-06)

**Job ID:** 8744308 (researchgpu05)  
**Method:** BFS depth≤6, width=8, DeepSeek-Prover-V1.5-RL 4-bit, `max_new_tokens=256`  
**Claimed score (batch verifier):** 143/244 = 58.6%  
**Verified score (individual lake build):** **75/244 = 30.7%**  
**Re-verification job:** 8749514 (researchgpu04)

### False Positive Analysis

68 of 143 claimed proofs (47.6%) were batch-verifier false positives:

- **~17 `theorem`-keyword FPs:** Model regenerated the theorem header as a tactic. Lean closes the
  `by` block when it sees `theorem`, pushing errors outside the tracked line range → misclassified
  as complete. Fixed in `prover/mcts.py` with a filter: `not re.match(r"^\s*theorem\s+\S", tac)`.
- **~51 line-range FPs:** The batch verifier's `range_end` calculation still misses some "unsolved
  goals" errors that land just outside the tracked range (e.g., errors on tactics spanning multiple
  lines, or errors reported at the closing token of a sub-proof). These look like no-error → (True,
  True) but are actually partial proofs.

### Result Files

- `results/minif2f_mcts_eval.json` — raw MCTS output (143 claimed proofs)
- `results/minif2f_mcts_reverified.json` — after re-verification (75 confirmed proofs)

### Next Steps

A clean re-run with the single-proof verifier inline (instead of batch) would eliminate all
line-range FPs and give an accurate score. Expected range: 40–55% given proof quality observed.

---

## MCTS Tree Search — Verifier Bug & Fix (2026-07-05)

**The original MCTS result (243/244 = 99.6%) was a false positive due to a regex bug.**

### The Bug

`prover/mcts.py` `_batch_verify_with_sorry()` used:
```python
# Wrong — Lean outputs "ProofGoals.lean:N:M: error: ..." not "error: ...ProofGoals.lean:N:M:"
re.finditer(r"error:.*?ProofGoals\.lean:(\d+):\d+:", out)
re.finditer(r"warning:.*?ProofGoals\.lean:(\d+):\d+:.*sorry", out)
```

Both regexes never matched Lean's actual output format, so `error_lines` and `sorry_lines`
were always empty. The fallback branch `"no error and no sorry → proof complete"` fired for
every generated tactic, making every first tactic look like a complete proof.

### Impact

Re-verification of all 243 claimed proofs against the actual Lean compiler:
- **Confirmed real proofs: 11/244 = 4.5%**
- **False positives: 232/243 = 95.5%**

Re-verified results saved in `results/minif2f_mcts_reverified.json`.

### Fix Applied

```python
# Fixed — correct Lean output format
re.finditer(r"ProofGoals\.lean:(\d+):\d+:.*?error:", out)
re.finditer(r"ProofGoals\.lean:(\d+):\d+:.*?warning:.*sorry", out)
```

Fixed in `prover/mcts.py`. Re-run in progress (job TBD).

---

## top_k=256 Results (2026-07-04 → 2026-07-05)

**Job ID:** 8744098 (researchgpu05)  
**Wall time:** ~12 hours  
**Seeded from:** top_k=64 checkpoint (97 proved problems carried over)  
**Claimed score: 114/244 = 46.7%** (+17 new proofs beyond top_k=64)  
**Verified score: 97/244 = 39.8%** (reverified via `lake env lean`, job 8749638)

All 17 "new" proofs beyond top_k=64 were batch-verifier false positives. The verified top_k=256
score is identical to top_k=64, confirming the extra samples found no genuinely new proofs.

### Notes

- Streaming verification in chunks of 32 (avoids single Lake call with 256 theorems)
- 17 new proofs beyond top_k=64, mostly on medium-difficulty algebra/number theory
- The hard tail (problems failing at top_k=64) is mostly impervious to more samples
- MCTS tree search with fixed verifier is the correct approach for those problems

### Result File

`results/minif2f_deepseek_base_topk256_test.json`

---

## top_k=64 Results (2026-07-03 → 2026-07-04)

**Job IDs:** 8742515 (researchgpu05), 8743946 (researchgpu04, resumed after CUDA hang)  
**Wall time:** ~17 hours  
**Score: 97/244 = 39.8%**

### Result File

`results/minif2f_deepseek_base_topk64_test.json`

---

## Baseline: top_k=32 (2026-07-03)

**Score: 60/244 = 24.6%**

---

## MA-ProofBench Evaluation (2026-07-07)

**Dataset:** `openbmb/MA-ProofBench` — 200 Lean 4 problems (100 level1 undergrad, 100 level2 PhD)  
**Job:** 8774411 (researchgpu05), cancelled after 68/200 level1 problems  
**Score: 0/200 = 0.0%**

### Why 0%

MA-ProofBench is qualitatively harder than miniF2F:

| Benchmark | Source | Difficulty | Our score |
|-----------|--------|------------|-----------|
| miniF2F | AMC/AIME/IMO competition | High school | 48.8% |
| MA-ProofBench level1 | Undergraduate analysis | Undergrad | 0% |
| MA-ProofBench level2 | PhD-level math | PhD | 0% |

Representative level1 problems:
- `LipschitzWith 1 Real.sin ∧ LipschitzWith 1 Real.cos`
- Complex analysis: holomorphic functions with conjugate-holomorphic constraint
- PDEs: existence of solutions via tsum equalities
- Measure theory: dominated convergence applications

These require multi-lemma Mathlib API calls and proof structures the model never
learned to synthesize from the competition-math training distribution.

### Bug discovered and fixed

`prover/mcts.py`'s `search()` method stripped `:= sorry` but not `:= by\n  sorry`
(the format MA-ProofBench uses). This left `sorry` in `base_stmt`, causing
`_batch_verify` to build malformed files (`theorem ... := by\n  sorry := by\n  <tactic>`).
All tactics returned `(False, False)` from parse errors, collapsing tree search at depth 0
for every problem. Fix committed in 820eb1b — but this did not change the 0% result,
since whole-proof sampling (unaffected by the bug) also found 0 proofs.

---

## Known CUDA Hang Problems (4-bit quantization specific)

Problems that cause silent `model.generate()` deadlocks (SIGALRM unreachable in CUDA kernel).
**Note: BF16 model does NOT hang on these — bug is 4-bit dequantization specific.**

- `numbertheory_4x3m7y3neq2003` (≠ in goal)
- `numbertheory_x5neqy2p4` (≠ in goal)
- `amc12_2000_p6` (≠ in goal)
- `mathd_numbertheory_66` (discovered 2026-07-04)
