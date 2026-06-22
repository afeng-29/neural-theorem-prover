# Neural Theorem Proving: System Survey

*Evaluated 2026-06-18. Criteria: open-source, Lean 4 compatible, locally runnable.*

---

## Candidate Systems

### 1. ReProver / LeanDojo (Yang et al., 2023) — PRIMARY RECOMMENDATION

**Repo:** https://github.com/lean-dojo/ReProver  
**Paper:** "LeanDojo: Theorem Proving with Retrieval-Augmented Language Models" (NeurIPS 2023)

| Criterion | Status |
|-----------|--------|
| Open source | Yes — MIT license |
| Lean 4 compatible | Yes — trained on mathlib4 |
| Pretrained weights | Yes — `kaiyuy/leandojo-lean4-tacgen-byt5-small` on HuggingFace |
| Local inference | Yes — CPU-feasible (small model, ~300M params) |
| LeanDojo integration | Native — designed around it |

**Architecture:** ByT5-based seq2seq model. Given a proof state string (goal + local context), it generates tactic strings. Retrieval augments the input with relevant premises from mathlib.

**Pretrained checkpoint:** `kaiyuy/leandojo-lean4-tacgen-byt5-small`  
Also available: `kaiyuy/leandojo-lean4-retriever-byt5-small` (premise retriever)

**Verdict:** Best option. It is the only system specifically designed for Lean 4 tactic prediction with a clean Python API, open weights, and a maintained codebase. LeanDojo provides the Lean 4 subprocess interface we need.

---

### 2. DeepSeek-Prover / DeepSeekMath

**Repo:** https://github.com/deepseek-ai/DeepSeek-Prover-V1.5  
**HuggingFace:** `deepseek-ai/DeepSeek-Prover-V1.5-RL`, `deepseek-ai/DeepSeek-Prover-V1.5-SFT`

| Criterion | Status |
|-----------|--------|
| Open source | Weights available, training code partially open |
| Lean 4 compatible | Yes — generates complete Lean 4 proofs |
| Pretrained weights | Yes — 7B parameter models |
| Local inference | Yes, but 7B requires ~14 GB VRAM for bfloat16 |

**Architecture:** Causal LM (DeepSeek-Math-7B base). Generates whole-proof completions rather than state-conditioned tactic prediction. Uses Monte Carlo Tree Search (MCTS) in V1.5-RL variant.

**Verdict:** Powerful but heavyweight. Good fallback if ReProver fails; the 7B size makes CPU inference impractical (>30 min per proof). Useful once GPU is available.

---

### 3. InternLM-Math

**HuggingFace:** `internlm/internlm2-math-plus-7b`, `internlm/internlm2-math-plus-1_8b`

| Criterion | Status |
|-----------|--------|
| Open source | Weights available |
| Lean 4 compatible | Partial — generates Lean 4 but not state-conditioned |
| Pretrained weights | Yes — 1.8B and 7B variants |
| Local inference | 1.8B is CPU-feasible |

**Verdict:** Not specifically optimized for tactic prediction; generates complete proofs. No LeanDojo integration. Lower priority than ReProver.

---

### 4. Llemma (EleutherAI, 2023)

**HuggingFace:** `EleutherAI/llemma_7b`, `EleutherAI/llemma_34b`  
**Paper:** "Llemma: An Open Language Model for Mathematics"

| Criterion | Status |
|-----------|--------|
| Open source | Yes — Apache 2.0 |
| Lean 4 compatible | Not specifically — general math LM |
| Pretrained weights | Yes — 7B and 34B |
| Local inference | 7B feasible with quantization |

**Verdict:** General math reasoning model, not a theorem prover. Requires significant prompt engineering to produce valid Lean 4 tactics, and no interaction with Lean's type checker during search. Not suitable as a primary system.

---

### 5. COPRA (Thakur et al., 2023)

**Repo:** https://github.com/IBM/COPRA  
**Paper:** "In-context Learning for Automated Proof Generation"

| Criterion | Status |
|-----------|--------|
| Open source | Code available, but requires OpenAI API key |
| Lean 4 compatible | Yes — Lean 4 and Isabelle variants |
| Pretrained weights | N/A — prompts GPT-4/Claude |
| Local inference | No — API-dependent |

**Verdict:** Not suitable for local deployment or fine-tuning. Useful as a reference for prompt design.

---

## Recommendation

**Start with ReProver (LeanDojo).** Reasons:

1. Only system with a state-conditioned tactic model specifically trained on Lean 4 / mathlib4
2. Smallest viable checkpoint (~300M params, CPU-runnable)
3. LeanDojo provides the Lean 4 subprocess wrapper we need for proof search
4. Active maintenance and documented API
5. Clean separation between tactic model, retriever, and search — easy to replace components for fine-tuning experiments

**Fallback:** If ReProver environment setup fails (e.g., Lean version mismatch), use `deepseek-ai/DeepSeek-Prover-V1.5-SFT` with whole-proof generation and Lean verification only (no interactive search).

---

## Key Dependencies and Gotchas

- **LeanDojo requires `GITHUB_ACCESS_TOKEN`** to clone mathlib4 and its dependencies. Set this before any `lean_dojo` import.
- **Lean 4 / mathlib4 first-build takes 20–40 minutes** — it compiles all of mathlib. This only happens once; subsequent runs use the cache at `~/.elan` and `~/.cache/mathlib`.
- **LeanDojo pins Lean versions** — the version in `lean-toolchain` must match what LeanDojo expects. As of LeanDojo 1.x, this is `leanprover/lean4:v4.3.0` or later. Check `lean_dojo.__version__` and the LeanDojo changelog if you hit version mismatches.
- **ReProver checkpoint format:** The HuggingFace checkpoint is a standard `transformers` model, loadable with `AutoModelForSeq2SeqLM`. No custom loading code needed.
