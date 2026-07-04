"""
Tactic prediction model wrapper.

Supports two backends:
  1. ReProver (ByT5-based seq2seq) — default, best quality
  2. CausalLM fallback — any HuggingFace causal LM prompted to generate tactics

The interface is the same for both: given a proof state string, return a list
of (tactic, log_prob) pairs ranked by model confidence.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import re
import torch

logger = logging.getLogger(__name__)

# LeanDojo training data annotates retrieved premises with <a>lemma_name</a> tags.
# Strip these before passing tactics to Lean — they are metadata, not syntax.
_TAG_RE = re.compile(r"</?a>")


@dataclass
class TacticCandidate:
    tactic: str
    log_prob: float  # higher = more confident

    def __lt__(self, other: "TacticCandidate") -> bool:
        return self.log_prob < other.log_prob


class TacticModel:
    """
    Wrapper around the ReProver ByT5 tactic generation model.

    model_path: path to a local HuggingFace checkpoint directory, or a
                HuggingFace model id string (e.g. 'kaiyuy/leandojo-lean4-tacgen-byt5-small').
    device:     'cuda', 'mps', or 'cpu'. Defaults to best available.
    """

    DEFAULT_MODEL_ID = "kaiyuy/leandojo-lean4-tacgen-byt5-small"

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: Optional[str] = None,
    ):
        if model_path is None:
            model_path = self.DEFAULT_MODEL_ID

        self.model_path = str(model_path)
        self.device = device or self._best_device()
        self._model = None
        self._tokenizer = None

    # ── Lazy loading ───────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import warnings
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging as hf_logging

        logger.info("Loading tactic model from %s on %s", self.model_path, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # The ReProver ByT5 checkpoint ships with lm_head.weight ≠ shared.weight.
        # Transformers 5.x warns about this but correctly leaves them untied.
        # Suppress the warning since we already set tie_word_embeddings=False in config.
        prev_hf_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*tie.*word.*embed.*", category=UserWarning)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                dtype=torch.float16 if self.device != "cpu" else torch.float32,
            ).to(self.device)
        hf_logging.set_verbosity(prev_hf_level)

        self._model.eval()
        logger.info("Tactic model loaded (%d parameters)", sum(p.numel() for p in self._model.parameters()))

    # ── Prediction ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_tactics(
        self,
        state: str,
        top_k: int = 32,
        num_beams: int = 32,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ) -> list[TacticCandidate]:
        """
        Given a proof state string, return up to `top_k` tactic candidates
        sorted by log probability (most confident first).

        state:          The current proof state (goal string from LeanDojo).
        top_k:          Number of candidates to return.
        num_beams:      Beam width for beam search. Should be >= top_k.
        max_new_tokens: Maximum tactic length in tokens.
        temperature:    Softmax temperature (1.0 = no change, <1 = sharper).
        """
        self._ensure_loaded()

        inputs = self._tokenizer(
            state,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
        ).to(self.device)

        outputs = self._model.generate(
            **inputs,
            num_beams=max(num_beams, top_k),
            num_return_sequences=top_k,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
            temperature=temperature,
        )

        # Decode sequences
        sequences = outputs.sequences
        # sequences_scores gives the normalized log-prob per sequence
        scores = outputs.sequences_scores.cpu().float().tolist()

        candidates = []
        seen: set[str] = set()
        for seq, score in zip(sequences, scores):
            tactic = _TAG_RE.sub("", self._tokenizer.decode(seq, skip_special_tokens=True)).strip()
            if tactic and tactic not in seen:
                seen.add(tactic)
                candidates.append(TacticCandidate(tactic=tactic, log_prob=float(score)))

        # Sort descending by log_prob
        candidates.sort(key=lambda c: c.log_prob, reverse=True)
        return candidates[:top_k]

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _best_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self):
        """Free GPU/MPS memory by deleting the model from device."""
        if self._model is not None:
            del self._model
            self._model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()


# ── CausalLM fallback ──────────────────────────────────────────────────────────

_CAUSAL_PROMPT_TEMPLATE = """\
You are a Lean 4 theorem prover. Given the current proof state, output ONLY the next tactic.
Do not explain. Do not include the proof state. One tactic per line.

Proof state:
{state}

Next tactic:"""

# DeepSeek-Prover-V1.5-RL was trained for WHOLE-PROOF completion, not next-tactic
# prediction from a bare proof state.  The correct prompt is a Lean 4 file with
# imports + theorem declaration ending at `:= by\n`, and the model generates the
# tactic body.  We extract each line of each generated proof as a TacticCandidate
# so the existing step-by-step REPL loop can try them sequentially.
_DEEPSEEK_PROVER_HEADER = """\
import Mathlib
import Aesop

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Finset

"""


class CausalLMTacticModel:
    """
    Fallback: use a causal LM (e.g. deepseek-ai/deepseek-math-7b-base) to generate tactics.
    Much slower and less accurate than ReProver for tactic prediction, but works without
    the ByT5 seq2seq setup.

    model_id: HuggingFace model id or local path.
    """

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or TacticModel._best_device()
        self._pipeline = None

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline

        logger.info("Loading causal LM %s for tactic prediction", self.model_id)
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    @torch.no_grad()
    def predict_tactics(
        self,
        state: str,
        top_k: int = 8,
        max_new_tokens: int = 128,
        **kwargs,
    ) -> list[TacticCandidate]:
        self._ensure_loaded()

        prompt = _CAUSAL_PROMPT_TEMPLATE.format(state=state.strip())
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=top_k,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
        )

        candidates = []
        seen: set[str] = set()
        # Assign uniform log-probs since we don't have beam scores here
        for i, out in enumerate(outputs):
            text = out["generated_text"].strip()
            # Take only the first line (one tactic)
            tactic = text.split("\n")[0].strip()
            if tactic and tactic not in seen:
                seen.add(tactic)
                # Penalize later outputs slightly so we have an ordering
                candidates.append(TacticCandidate(tactic=tactic, log_prob=-float(i)))

        return candidates


# ── DeepSeek-Prover-V1.5 ──────────────────────────────────────────────────────

def _extract_deepseek_tactics(text: str) -> list[str]:
    """
    Extract tactic lines from a DeepSeek-Prover whole-proof completion.

    DeepSeek generates proof body lines after `example ... := by\n`.
    We keep non-empty stripped lines until we hit a stop marker (new
    theorem/import declaration, markdown fence, or a blank-line + keyword
    that signals the proof ended).

    The DeepSeek tokenizer encodes space as Ġ (U+0120) and newline as Ċ
    (U+010A) rather than converting them back to ASCII; normalise first.
    """
    # Reverse the BPE byte-as-unicode encoding used by GPT-2/DeepSeek tokenizers.
    # Ġ (U+0120) → space (0x20), Ċ (U+010A) → newline (0x0A).
    text = text.replace("Ġ", " ").replace("Ċ", "\n")

    _STOP_PREFIXES = ("import ", "theorem ", "lemma ", "def ", "open ",
                      "#check", "#eval", "```", "--", "/-", "example")
    # English prose indicators: 4-bit quantization sometimes drops newlines,
    # causing the model's English explanation to concatenate with the tactic.
    # Reject lines that look like English sentences rather than Lean tactics.
    _PROSE_WORDS = frozenset([
        "the", "is", "are", "was", "were", "can", "that", "this", "we", "for",
        "to", "of", "in", "on", "it", "its", "be", "by", "an", "or", "if",
        "use", "uses", "used", "since", "which", "as", "shows", "means",
        "prove", "proof", "function", "tactic", "automatically", "given",
        "here", "where", "how", "what",
    ])

    def _looks_like_prose(s: str) -> bool:
        if len(s) > 150:
            return True
        if "\\(" in s or "\\[" in s:  # LaTeX math
            return True
        words = s.split()
        if len(words) >= 4:
            lowercase_words = [w.rstrip(".,;:") for w in words if w[0:1].islower()]
            prose_count = sum(1 for w in lowercase_words if w in _PROSE_WORDS)
            if prose_count >= 2:
                return True
        return False

    def _is_garbage_line(s: str) -> bool:
        """True if this line should be DROPPED — it's model hallucination, not Lean."""
        # Non-ASCII characters: always garbage from the 4-bit model
        if any(ord(c) > 127 for c in s):
            return True
        # Markdown-style headers/bullets (### Explanation, * item, - item, 1. step)
        if re.match(r"^[#\*\-]", s):
            return True
        # Numbered list items: "1. Something" or "1) Something"
        if re.match(r"^\d+[.)]\s", s):
            return True
        # Separator lines: ===, ---, all punctuation
        if re.match(r"^[=\-_]{3,}$", s):
            return True
        # Lines that are just keywords/labels ("Step-by-Step", "Proof:", etc.)
        if re.match(r"^[A-Z][A-Za-z\s-]+:?\s*$", s) and len(s.split()) <= 5:
            return True
        # LLM meta-prompts that leaked into the output
        if re.search(r"(complete the following|lean 4 code|fill in|your answer|solution:)", s, re.IGNORECASE):
            return True
        return False

    # Whole-proof bodies that contain `sorry` are not real proofs — reject entire proof
    # (sorry is syntactically valid in Lean 4 and passes lake build, but is an axiom skip)
    _SORRY_RE = re.compile(r"\bsorry\b")

    tactics = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in _STOP_PREFIXES):
            break
        if _is_garbage_line(stripped):
            continue
        if _looks_like_prose(stripped):
            continue
        tactics.append(stripped)

    if any(_SORRY_RE.search(t) for t in tactics):
        return []  # proof uses sorry — not a real proof
    return tactics


class DeepSeekProverModel:
    """
    Tactic model backed by deepseek-ai/DeepSeek-Prover-V1.5-RL (7B params).

    The model was trained for WHOLE-PROOF completion from a Lean 4 theorem
    declaration (`:= by` prefix), not step-by-step tactic chat.  Use
    generate_proofs() to get full proof candidates; ProofSearch routes
    DeepSeek through _prove_deepseek_whole_proof() which tries each
    generated script via the REPL.

    Requires ~14 GB VRAM in fp16 (fits a V100-32GB).
    """

    DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-Prover-V1.5-RL"

    def __init__(
        self,
        model_id: str | None = None,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        lora_adapter: str | None = None,
    ):
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.device = device or TacticModel._best_device()
        self.load_in_4bit = load_in_4bit
        self.lora_adapter = lora_adapter  # path to saved PEFT LoRA adapter dir
        self._model = None
        self._tokenizer = None
        self._generate_batch = 4  # conservative default; bumped to 16 after 4-bit detection in eval scripts

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import time
        import random
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        # Transformers 5.x caching_allocator_warmup pre-allocates ~7GB on the GPU
        # as a CUDA cache hint. On shared nodes the CUDA virtual address space is
        # already fragmented by other processes, causing "CUDA driver error: out of
        # memory" even when physical VRAM is ample. The warmup is purely a
        # performance optimization; patching it to a no-op is safe.
        try:
            import transformers.modeling_utils as _tmu
            if hasattr(_tmu, "caching_allocator_warmup"):
                _tmu.caching_allocator_warmup = lambda *a, **kw: None
        except Exception:
            pass

        logger.info("Loading DeepSeek-Prover from %s on %s", self.model_id, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        # Shared GPU nodes often have CUDA VMA exhausted by other tenants even when
        # 4-bit loading needs ~8GB free (3.5GB weights + BF16 embedding overhead).
        # Empirically confirmed: job loaded successfully with 11GB free on researchgpu05.
        NEED_GB = 8
        SLEEP_S = 300  # 5 minutes between checks/retries

        for wc in range(12):  # wait up to 1 hour (12 × 5 min)
            if not torch.cuda.is_available():
                break
            free_mem, _ = torch.cuda.mem_get_info(0)
            if free_mem >= NEED_GB * 1024 ** 3:
                break
            jitter = random.uniform(0, 30)
            logger.info(
                "%.1fGB GPU free < %dGB needed — sleeping %.0fs (wait %d/24)",
                free_mem / 1024 ** 3, NEED_GB, SLEEP_S + jitter, wc + 1,
            )
            time.sleep(SLEEP_S + jitter)

        quantized = False
        for attempt in range(6):  # retry loading up to 6 times on GPU OOM
            try:
                quantized = self._load_model_to_gpu(AutoModelForCausalLM, BitsAndBytesConfig)
                break  # success
            except RuntimeError as e:
                _oom = ("out of memory" in str(e).lower()
                        or "cuda driver error" in str(e).lower())
                if not _oom or attempt >= 5:
                    raise
                if self._model is not None:
                    del self._model
                    self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                jitter = random.uniform(0, 60)
                logger.warning(
                    "GPU load failed (attempt %d/6): %s — retrying in %.0fs",
                    attempt + 1, e, SLEEP_S + jitter,
                )
                time.sleep(SLEEP_S + jitter)

        if self.lora_adapter:
            from peft import PeftModel
            logger.info("Loading LoRA adapter from %s", self.lora_adapter)
            self._model = PeftModel.from_pretrained(self._model, self.lora_adapter)
            if not quantized:
                self._model = self._model.to(torch.bfloat16)

        self._model.eval()
        n = sum(p.numel() for p in self._model.parameters())
        logger.info("DeepSeek-Prover loaded (%.1fB parameters)", n / 1e9)

    def _load_model_to_gpu(self, AutoModelForCausalLM, BitsAndBytesConfig) -> bool:
        """Load model to GPU; returns True if 4-bit quantized, False if BF16."""
        # Smoke-test: verify the CUDA allocator can service any allocation at all.
        # Catches broken expandable_segments or CUDA context issues before the
        # expensive model load attempt.
        if torch.cuda.is_available():
            try:
                _t = torch.zeros(1024, device='cuda:0')
                del _t
                torch.cuda.empty_cache()
            except RuntimeError as e:
                raise RuntimeError(f"CUDA allocator smoke test failed ({e}); "
                                   "GPU node may have broken CUDA context") from e

        # Proactively skip BF16 (14GB) if GPU doesn't have 16GB free.
        # Avoids failed .to("cuda") which can corrupt the CUDA context.
        _use_4bit = self.load_in_4bit
        if not _use_4bit and torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info(0)
            if free_mem < 16 * 1024 ** 3:
                logger.info(
                    "%.1fGB GPU free < 16GB — using 4-bit directly",
                    free_mem / 1024 ** 3,
                )
                _use_4bit = True

        quantized = _use_4bit
        if _use_4bit:
            # device_map={"": 0}: all layers on GPU 0, no CPU spillover.
            # BnB 4-bit needs ~3.5GB VRAM; transformers 4.x quantizes each layer
            # before GPU placement so peak usage stays near that.
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True,
                device_map={"": 0},
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            )
        else:
            # Load to CPU first, then move to GPU in one shot.
            # In transformers 4.46.3 there is no caching_allocator_warmup.
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
            )
            cuda_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            try:
                self._model = self._model.to(cuda_dev)
            except RuntimeError as e:
                _oom = ("out of memory" in str(e).lower()
                        or "cuda driver error" in str(e).lower())
                if not _oom:
                    raise
                # GPU filled up between check and .to() — fall to 4-bit
                logger.warning("BF16 .to(cuda) failed — falling back to 4-bit")
                del self._model
                self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, trust_remote_code=True,
                    device_map={"": 0},
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                )
                quantized = True

        # Fail fast if model landed on CPU — inference would be 100-1000× slower
        _dev = str(next(self._model.parameters()).device)
        if _dev == "cpu":
            raise RuntimeError(
                "Model loaded onto CPU instead of GPU — GPU VMA likely exhausted. "
                "Will retry after sleeping."
            )
        logger.info("Model device: %s", _dev)
        return quantized

    def _build_prompt(self, theorem: str, hypotheses: list[str],
                      prior_tactics: list[str] | None = None) -> str:
        """Format a Lean 4 file header + theorem stub for whole-proof completion."""
        hyp_str = " ".join(f"({h})" for h in hypotheses) if hypotheses else ""
        indent = "  "
        tactic_lines = ""
        if prior_tactics:
            tactic_lines = "\n".join(f"{indent}{t}" for t in prior_tactics) + "\n"
        return (
            f"{_DEEPSEEK_PROVER_HEADER}"
            f"example {hyp_str} : {theorem} := by\n"
            f"{tactic_lines}"
            f"{indent}"
        )

    @torch.no_grad()
    def generate_proofs(
        self,
        theorem: str,
        hypotheses: list[str],
        n: int = 32,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        prior_tactics: list[str] | None = None,
    ) -> list[list[str]]:
        """
        Generate n complete proof attempts.

        Returns a list of tactic-line sequences (one sequence per attempt).
        The prompt ends with `  ` (two-space indent after `by\n`) so the
        model generates the indented proof body directly.

        prior_tactics: tactics already applied (for mid-proof continuation).
        """
        self._ensure_loaded()
        prompt_text = self._build_prompt(theorem, hypotheses, prior_tactics)
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            prompt_text, return_tensors="pt",
            max_length=2048, truncation=True,  # prevent OOM on long proof states
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate in small batches to limit KV-cache VRAM (32 at once OOMs on ≤16GB GPUs).
        # self._generate_batch persists across goals so we don't retry OOM sizes each call.
        proofs: list[list[str]] = []
        remaining = n
        while remaining > 0:
            bs = min(self._generate_batch, remaining)
            try:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=bs,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            except torch.cuda.OutOfMemoryError:
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()
                if bs == 1:
                    logger.warning("generate_proofs: OOM with batch_size=1, aborting")
                    break
                self._generate_batch = max(1, self._generate_batch // 2)
                logger.warning("generate_proofs: OOM, reducing batch_size to %d", self._generate_batch)
                continue
            for seq in outputs:
                text = self._tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                tactics = _extract_deepseek_tactics(text)
                if tactics:
                    proofs.append(tactics)
            remaining -= bs
        logger.info("DeepSeek generated %d non-empty proof scripts", len(proofs))
        return proofs

    @torch.no_grad()
    def generate_next_tactics(
        self,
        formal_statement: str,
        applied_tactics: list[str],
        n: int = 8,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> list[str]:
        """
        Given the theorem and tactics applied so far, generate n candidate
        SINGLE next tactics (one tactic per candidate, not whole proofs).

        Used by the tree-search prover to expand nodes in the proof tree.

        formal_statement: 'theorem NAME PARAMS : GOAL := sorry'
        applied_tactics:  list of tactics already applied (may be empty)
        Returns: list of tactic strings (de-duplicated, first non-empty line each)
        """
        self._ensure_loaded()

        # Build prompt: preamble + theorem stub + tactics applied so far
        # Model continues from the next indented line (next tactic)
        base = re.sub(r":=\s*sorry\s*$", "", formal_statement.strip())
        prompt = _DEEPSEEK_PROVER_HEADER + base + " := by\n"
        if applied_tactics:
            prompt += "\n".join(f"  {t}" for t in applied_tactics) + "\n"
        prompt += "  "  # indent for the next tactic

        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            prompt, return_tensors="pt", max_length=2048, truncation=True
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        raw_lines: list[str] = []
        remaining = n
        while remaining > 0:
            bs = min(self._generate_batch, remaining)
            try:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=bs,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            except torch.cuda.OutOfMemoryError:
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()
                if bs == 1:
                    break
                self._generate_batch = max(1, self._generate_batch // 2)
                continue
            for seq in outputs:
                text = self._tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                text = text.replace("Ġ", " ").replace("Ċ", "\n")
                # Take only the first non-empty, non-comment line = next tactic
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith("--"):
                        raw_lines.append(stripped)
                        break
            remaining -= bs

        # De-duplicate while preserving order
        seen: set[str] = set()
        tactics: list[str] = []
        for t in raw_lines:
            if t not in seen and not re.search(r"\bsorry\b", t):
                seen.add(t)
                tactics.append(t)
        return tactics

    @torch.no_grad()
    def predict_tactics(
        self,
        state: str,
        top_k: int = 32,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> list[TacticCandidate]:
        """
        Fallback: step-by-step tactic prediction from bare proof state.
        Used only when theorem context is unavailable; ProofSearch calls
        generate_proofs() instead via _prove_deepseek_whole_proof().
        """
        self._ensure_loaded()
        device = next(self._model.parameters()).device
        # Minimal Lean-style prompt without full file header
        prompt = f"-- proof state:\n{state.strip()}\n-- next tactic:\n  "
        inputs = self._tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=top_k,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        candidates = []
        seen: set[str] = set()
        for i, seq in enumerate(outputs):
            text = self._tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for line in text.split("\n"):
                line = line.strip()
                if line and not line.startswith(("#", "-", "`")):
                    if line not in seen:
                        seen.add(line)
                        candidates.append(TacticCandidate(tactic=line, log_prob=-float(i)))
                    break

        return candidates

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            if self.device == "cuda":
                import torch
                torch.cuda.empty_cache()
