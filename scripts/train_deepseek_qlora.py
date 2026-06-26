"""
QLoRA fine-tuning of DeepSeek-Prover-V1.5-RL on calculus tactic data.

Training format (causal completion, loss on tactic only):
  -- State:\n{proof_state}\n-- Tactic:\n{tactic}\n

Data: data/calculus/train.jsonl  (6,734 examples from Mathlib calculus files)
Model: models/pretrained/deepseek-prover-v1.5-rl  (7B, loaded in 4-bit NF4)
LoRA: r=16, alpha=32, all linear projection layers
Output: models/finetuned/deepseek-qlora-calculus/

Usage:
    python scripts/train_deepseek_qlora.py [--epochs 3] [--lr 2e-4]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Strip LeanDojo <a>lemma</a> annotation tags — not valid Lean syntax
_TAG_RE = re.compile(r"</?a>")

PROMPT_PREFIX = "-- State:\n"
PROMPT_SUFFIX = "\n-- Tactic:\n"


def format_example(state: str, tactic: str) -> tuple[str, str]:
    """Return (prompt, completion) strings."""
    clean_tactic = _TAG_RE.sub("", tactic).strip()
    prompt = PROMPT_PREFIX + state.strip() + PROMPT_SUFFIX
    completion = clean_tactic + "\n"
    return prompt, completion


class CalculusDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[tuple[str, str]] = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                state = rec.get("state", "")
                tactic = rec.get("tactic", "")
                if state and tactic:
                    self.examples.append(format_example(state, tactic))

        logger.info("Loaded %d examples from %s", len(self.examples), path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        prompt, completion = self.examples[idx]
        full_text = prompt + completion

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)

        # Build labels: mask the prompt portion (compute loss only on completion)
        prompt_enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask prompt

        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        n = x["input_ids"].shape[0]
        input_ids[i, :n] = x["input_ids"]
        labels[i, :n] = x["labels"]
    attention_mask = (input_ids != pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/pretrained/deepseek-prover-v1.5-rl")
    parser.add_argument("--train-file", default="data/calculus/train.jsonl")
    parser.add_argument("--val-file", default="data/calculus/val.jsonl")
    parser.add_argument("--output-dir", default="models/finetuned/deepseek-qlora-calculus")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = 1 × grad_accum)")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load tokenizer ──────────────────────────────────────────────────────
    logger.info("Loading tokenizer from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model in 4-bit NF4 ─────────────────────────────────────────────
    logger.info("Loading model in 4-bit NF4...")
    # Use float32 compute on V100 — bfloat16 unavailable, fp16 causes NaN gradients
    # in backprop through 4-bit dequantization. float32 is ~2x slower but stable.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float32,  # non-quantized layers (LayerNorm, embeddings) in float32
        trust_remote_code=True,
    )
    # use_reentrant=False avoids bitsandbytes/gradient-checkpointing incompatibility that
    # causes NaN gradients. use_reentrant=True (default) reruns the forward pass during
    # backward, and bitsandbytes custom CUDA ops are not safe to rerun this way.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── Apply LoRA ───────────────────────────────────────────────────────────
    # Target all linear projection layers for maximum domain adaptation
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = CalculusDataset(args.train_file, tokenizer, args.max_length)
    val_ds   = CalculusDataset(args.val_file,   tokenizer, args.max_length)

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    # ── Optimizer + schedule ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info("Training: %d examples, %d epochs, %d grad_accum → %d optimizer steps",
                len(train_ds), args.epochs, args.grad_accum, total_steps)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            raw_loss = loss.item() * args.grad_accum  # record BEFORE backward
            loss.backward()
            train_loss += raw_loss

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                opt_step = (step + 1) // args.grad_accum
                if opt_step % 50 == 0:
                    avg = train_loss / (step + 1)
                    logger.info("Epoch %d step %d/%d  train_loss=%.4f  lr=%.2e",
                                epoch, opt_step,
                                math.ceil(len(train_loader) / args.grad_accum),
                                avg, scheduler.get_last_lr()[0])

        avg_train = train_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        avg_val = val_loss / len(val_loader)

        logger.info("Epoch %d  train_loss=%.4f  val_loss=%.4f", epoch, avg_train, avg_val)

        # Save LoRA adapter each epoch; keep best
        epoch_dir = output_dir / f"epoch-{epoch}"
        model.save_pretrained(str(epoch_dir))
        tokenizer.save_pretrained(str(epoch_dir))
        logger.info("Saved adapter to %s", epoch_dir)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_dir = output_dir / "best"
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            logger.info("New best val_loss=%.4f → saved to %s", avg_val, best_dir)

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
    logger.info("Best LoRA adapter at %s/best", output_dir)
    logger.info(
        "To evaluate: set --model-path %s/best --lora-adapter %s/best in run script",
        args.model_path, output_dir,
    )


if __name__ == "__main__":
    main()
