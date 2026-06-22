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

import torch

logger = logging.getLogger(__name__)


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
            tactic = self._tokenizer.decode(seq, skip_special_tokens=True).strip()
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
