# miniF2F Evaluation Results

## DeepSeek-Prover-V1.5-RL — Whole-Proof Generation

All runs use the test split (244 problems), 4-bit BitsAndBytes quantization on CBS A40 GPU nodes,
`max_new_tokens=256`, `batch=16`, `timeout=300s` per Lake verification call.

---

## Summary

| Run | top_k | Proved | Total | Accuracy | vs Baseline |
|-----|-------|--------|-------|----------|-------------|
| Baseline | 32 | 60 | 244 | 24.6% | — |
| **top_k=64** | **64** | **97** | **244** | **39.8%** | **+15.2pp (+37 problems)** |
| Published (DeepSeek paper) | ~3200 | 147 | 244 | 60.2% | target |

---

## top_k=64 Results (2026-07-03 → 2026-07-04)

**Job IDs:** 8742515 (researchgpu05), 8743946 (researchgpu04, resumed after CUDA hang)  
**Wall time:** ~17 hours  
**Score: 97/244 = 39.8%**

### Category Breakdown

| Category | Proved |
|----------|--------|
| mathd (algebra + numbertheory) | 74 |
| algebra | 6 |
| amc12b | 5 |
| amc12a | 4 |
| induction | 3 |
| aime | 2 |
| imo | 2 |
| amc12 | 1 |
| **Total** | **97** |

### Notable Proofs

- `imo_1984_p6` — proved with `linarith` (hard IMO inequality)
- `imo_1964_p2` — proved with `nlinarith [sq_nonneg (a-b), sq_nonneg (b-c), sq_nonneg (c-a)]`
- `aime_1983_p2`, `aime_1989_p8` — AIME competition problems proved
- Most proofs use short tactics: `linarith`, `nlinarith`, `omega`, `norm_num`, `ring`, `field_simp`

### Known CUDA Hang Problems (pre-failed)

Problems that cause silent `model.generate()` deadlocks (SIGALRM unreachable in CUDA kernel):
- `numbertheory_4x3m7y3neq2003` (≠ in goal)
- `numbertheory_x5neqy2p4` (≠ in goal)
- `amc12_2000_p6` (≠ in goal)
- `mathd_numbertheory_66` (discovered 2026-07-04)

### Result File

`results/minif2f_deepseek_base_topk64_test.json`

---

## Next Steps

1. **Scale sampling** — run top_k=256 or top_k=512 to approach ~50%
2. **MCTS search** — use `prover/mcts.py` for tree-search guided generation (targets published 60.2%)
3. **Fine-tuning** — LoRA on proved problems to improve hit rate on harder AMC/AIME/IMO
