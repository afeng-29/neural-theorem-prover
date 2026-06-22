"""
Fine-tuning scaffold for the ReProver tactic model.

STATUS: Ready to run — uncomment the main() call at the bottom, point at data,
        and ensure a GPU is available.

Training task:
    Sequence-to-sequence: proof state string → next tactic string.
    Base model: ByT5-small (same architecture as the ReProver pretrained checkpoint).
    This means you can initialize from the pretrained checkpoint and fine-tune on
    domain-specific data, rather than training from scratch.

To run:
    python training/finetune.py \
        --train-data data/nat_basic.jsonl \
        --val-split 0.1 \
        --base-model models/pretrained/leandojo-lean4-tacgen-byt5-small \
        --output-dir models/finetuned/ \
        --epochs 5 \
        --batch-size 16

Requires: GPU (at least 16 GB VRAM for batch size 16 with ByT5-small).
For CPU-only: reduce batch size to 1 and expect very slow training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Lean 4 special tokens ──────────────────────────────────────────────────────
# ByT5 operates on raw UTF-8 bytes, so no special tokenizer handling is needed
# for Unicode math symbols. These are listed here for documentation and for use
# with other tokenizers (e.g., T5, GPT-based) that may need added tokens.
LEAN4_SPECIAL_SYMBOLS = [
    "⊢",   # turnstile (goal separator)
    "→",   # function type / implication
    "↔",   # iff
    "∀",   # forall
    "∃",   # exists
    "λ",   # lambda
    "ℕ",   # natural numbers
    "ℤ",   # integers
    "ℚ",   # rationals
    "ℝ",   # reals
    "ℂ",   # complex
    "·",   # function application dot
    "⟨",   # angle bracket open
    "⟩",   # angle bracket close
    "∧",   # and
    "∨",   # or
    "¬",   # not
    "≤",   # less-or-equal
    "≥",   # greater-or-equal
    "≠",   # not equal
    "∈",   # element of
    "∉",   # not element of
    "⊆",   # subset
    "∩",   # intersection
    "∪",   # union
    "∑",   # sum
    "∏",   # product
    "α",   # type variable (commonly used in Lean)
    "β",
    "γ",
]


# ── Dataset ────────────────────────────────────────────────────────────────────

class TacticDataset(Dataset):
    """
    Reads JSONL files produced by data/extract.py.
    Each proof step becomes one (input, target) training example:
        input:  proof state string (before the tactic)
        target: tactic string
    """

    def __init__(
        self,
        jsonl_paths: list[str | Path],
        tokenizer,
        max_input_length: int = 2048,
        max_target_length: int = 256,
        include_premises: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.include_premises = include_premises
        self.examples: list[tuple[str, str]] = []

        for path in jsonl_paths:
            self._load_jsonl(Path(path))

        logger.info("Loaded %d training examples from %d files", len(self.examples), len(jsonl_paths))

    def _load_jsonl(self, path: Path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for step in record.get("steps", []):
                    state = step.get("state", "").strip()
                    tactic = step.get("tactic", "").strip()
                    if state and tactic:
                        self.examples.append((state, tactic))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        state, tactic = self.examples[idx]

        model_inputs = self.tokenizer(
            state,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tactic,
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics_factory(tokenizer):
    """
    Returns a metrics function for Seq2SeqTrainer.
    Computes top-1 exact-match tactic accuracy (per token sequence).
    """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # predictions shape: (batch, seq_len) — argmax already applied by Trainer
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 (padding label) with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Top-1 exact match
        exact_match = sum(
            p.strip() == l.strip()
            for p, l in zip(decoded_preds, decoded_labels)
        ) / max(len(decoded_labels), 1)

        return {"exact_match": exact_match}

    return compute_metrics


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    train_data: list[str | Path],
    base_model: str,
    output_dir: str | Path,
    val_split: float = 0.1,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    warmup_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    fp16: bool = True,
    seed: int = 42,
):
    """
    Fine-tune the tactic model on (state, tactic) pairs from `train_data`.

    train_data:   List of JSONL file paths from data/extract.py.
    base_model:   Path to pretrained checkpoint or HuggingFace model id.
    output_dir:   Where to save the fine-tuned model.
    val_split:    Fraction of data held out for validation.
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer from %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Build full dataset, then split
    full_dataset = TacticDataset(train_data, tokenizer)
    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    logger.info("Train: %d examples, Val: %d examples", n_train, n_val)

    logger.info("Loading model from %s", base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d (%.1fM)", n_params, n_params / 1e6)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if fp16 else None,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        fp16=fp16 and torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=256,
        generation_num_beams=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        logging_steps=50,
        dataloader_num_workers=4,
        seed=seed,
        report_to="none",    # set to "wandb" if tracking experiments
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving best model to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Fine-tuning complete.")

    # Save training metrics
    metrics = trainer.evaluate()
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Final eval metrics: %s", metrics)

    return trainer


# ── Top-10 accuracy evaluation ─────────────────────────────────────────────────

def evaluate_top_k_accuracy(
    model_path: str | Path,
    test_data: list[str | Path],
    k: int = 10,
    n_samples: int = 500,
    device: Optional[str] = None,
):
    """
    Evaluate top-k tactic prediction accuracy on a held-out set.
    Loads model from model_path and checks whether the ground-truth tactic
    appears in the top-k beam search outputs.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from prover.tactic_model import TacticModel

    model = TacticModel(model_path=model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Load test examples
    examples: list[tuple[str, str]] = []
    for path in test_data:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                for step in rec.get("steps", []):
                    state = step.get("state", "").strip()
                    tactic = step.get("tactic", "").strip()
                    if state and tactic:
                        examples.append((state, tactic))

    random.shuffle(examples)
    examples = examples[:n_samples]

    top1_correct = 0
    topk_correct = 0

    for state, true_tactic in examples:
        candidates = model.predict_tactics(state, top_k=k)
        pred_tactics = [c.tactic.strip() for c in candidates]

        if pred_tactics and pred_tactics[0] == true_tactic:
            top1_correct += 1
        if true_tactic in pred_tactics:
            topk_correct += 1

    n = len(examples)
    results = {
        "n_samples": n,
        "top1_accuracy": top1_correct / n,
        f"top{k}_accuracy": topk_correct / n,
    }
    logger.info("Top-1 accuracy: %.3f, Top-%d accuracy: %.3f", results["top1_accuracy"], k, results[f"top{k}_accuracy"])
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune the tactic model")
    parser.add_argument("--train-data", nargs="+", required=True,
                        help="JSONL files from data/extract.py")
    parser.add_argument("--base-model", default="models/pretrained/leandojo-lean4-tacgen-byt5-small",
                        help="Pretrained checkpoint path or HuggingFace model id")
    parser.add_argument("--output-dir", default="models/finetuned/",
                        help="Where to save the fine-tuned model")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    train(
        train_data=args.train_data,
        base_model=args.base_model,
        output_dir=args.output_dir,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=not args.no_fp16,
    )


# ── Uncomment the line below to enable training when this script is run directly ──
# if __name__ == "__main__":
#     main()
