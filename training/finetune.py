"""
Fine-tune the ReProver ByT5-small tactic model on a domain-specific dataset.

Training task:
    Seq2seq: proof state string → next tactic string.
    Base model: kaiyuy/leandojo-lean4-tacgen-byt5-small (ByT5-small, 300M params).
    We fine-tune from the pretrained checkpoint rather than training from scratch.

Input data format (flat JSONL, one example per line):
    {"state": "<lean4 goal>", "tactic": "<lean4 tactic>", "full_name": "...", "file_path": "..."}

Produced by data/prepare_calculus.py (or any prepare_<domain>.py).

Typical usage:
    # 1. Prepare data
    python data/prepare_calculus.py --domain calculus --output-dir data/calculus

    # 2. Fine-tune
    python training/finetune.py \\
        --train-data data/calculus/train.jsonl \\
        --val-data   data/calculus/val.jsonl \\
        --test-data  data/calculus/test.jsonl \\
        --base-model models/pretrained/leandojo-lean4-tacgen-byt5-small \\
        --output-dir models/finetuned/calculus/ \\
        --epochs 10 --batch-size 8

    # 3. Evaluate top-k accuracy on test set
    python training/finetune.py --eval-only \\
        --test-data  data/calculus/test.jsonl \\
        --base-model models/finetuned/calculus/

Hardware notes:
    MPS (Apple Silicon): set --batch-size 4 --grad-accum 4, omit --fp16
    CUDA 16 GB:          batch-size 16 works; enable --fp16
    CPU only:            batch-size 1, very slow (not recommended)

Adding more domains later:
    python data/prepare_calculus.py --domain algebra --output-dir data/algebra
    python training/finetune.py \\
        --train-data data/calculus/train.jsonl data/algebra/train.jsonl \\
        --val-data   data/calculus/val.jsonl   data/algebra/val.jsonl \\
        --test-data  data/calculus/test.jsonl  data/algebra/test.jsonl \\
        --base-model models/finetuned/calculus/ \\   # continue from calculus checkpoint
        --output-dir models/finetuned/calculus_algebra/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────────────

class TacticDataset(Dataset):
    """
    Reads flat JSONL files produced by data/prepare_<domain>.py.
    Each line is one (state, tactic) training example.

    Also accepts the older nested format:
        {"steps": [{"state": "...", "tactic": "..."}]}
    for backward compatibility with data/extract.py output.
    """

    def __init__(
        self,
        jsonl_paths: list[str | Path],
        tokenizer,
        max_input_length: int = 2048,
        max_target_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.examples: list[tuple[str, str]] = []

        for path in jsonl_paths:
            self._load(Path(path))

        logger.info("Loaded %d examples from %d file(s)", len(self.examples), len(jsonl_paths))

    def _load(self, path: Path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "state" in rec and "tactic" in rec:
                    # flat format (from prepare_calculus.py)
                    state, tactic = rec["state"].strip(), rec["tactic"].strip()
                    if state and tactic:
                        self.examples.append((state, tactic))
                elif "steps" in rec:
                    # nested format (from data/extract.py)
                    for step in rec["steps"]:
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
        labels = self.tokenizer(
            text_target=tactic,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# ── Metrics ────────────────────────────────────────────────────────────────────

def make_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        exact_match = sum(
            p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)
        ) / max(len(decoded_labels), 1)

        return {"exact_match": exact_match}
    return compute_metrics


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    train_paths: list[str | Path],
    val_paths: list[str | Path],
    base_model: str,
    output_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 8,
    grad_accum: int = 1,
    learning_rate: float = 3e-4,
    warmup_steps: int = 200,
    fp16: bool = False,
    max_input_length: int = 1024,
    max_target_length: int = 256,
    seed: int = 42,
):
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer and model from %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d (%.1fM)", n_params, n_params / 1e6)

    train_dataset = TacticDataset(train_paths, tokenizer)
    val_dataset   = TacticDataset(val_paths,   tokenizer)

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
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        fp16=fp16 and torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=256,
        generation_num_beams=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        logging_steps=50,
        dataloader_num_workers=0,   # 0 avoids fork issues on macOS
        seed=seed,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting training: %d train, %d val", len(train_dataset), len(val_dataset))
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Best model saved to %s", output_dir)

    metrics = trainer.evaluate()
    with open(output_dir / "val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Val metrics: %s", metrics)
    return trainer


# ── Top-k evaluation (no Lean interaction needed) ──────────────────────────────

def evaluate_top_k(
    model_path: str | Path,
    test_paths: list[str | Path],
    k: int = 10,
    n_samples: Optional[int] = None,
    device: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """
    Measures top-1 and top-k exact-match tactic accuracy on test data.
    Does NOT run Lean — just checks if the predicted tactic string matches the
    ground-truth tactic string exactly.  Use test_pipeline.py for end-to-end
    proof success rate.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from prover.tactic_model import TacticModel

    model = TacticModel(model_path=str(model_path), device=device)

    examples: list[tuple[str, str]] = []
    for path in test_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                if "state" in rec and "tactic" in rec:
                    examples.append((rec["state"].strip(), rec["tactic"].strip()))
                elif "steps" in rec:
                    for step in rec["steps"]:
                        s, t = step.get("state", "").strip(), step.get("tactic", "").strip()
                        if s and t:
                            examples.append((s, t))

    random.seed(seed)
    random.shuffle(examples)
    if n_samples:
        examples = examples[:n_samples]

    top1 = topk = 0
    for i, (state, true_tactic) in enumerate(examples):
        preds = [c.tactic.strip() for c in model.predict_tactics(state, top_k=k)]
        if preds and preds[0] == true_tactic:
            top1 += 1
        if true_tactic in preds:
            topk += 1
        if (i + 1) % 50 == 0:
            logger.info("  evaluated %d / %d", i + 1, len(examples))

    n = len(examples)
    results = {
        "n_samples":        n,
        "top1_exact_match": top1 / n,
        f"top{k}_exact_match": topk / n,
    }
    logger.info("Top-1: %.3f  Top-%d: %.3f  (n=%d)", results["top1_exact_match"], k, results[f"top{k}_exact_match"], n)

    out_path = Path(model_path) / "test_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Test metrics saved to %s", out_path)
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate the tactic model")
    parser.add_argument("--train-data", nargs="+", default=[],
                        help="JSONL training files (flat format from prepare_<domain>.py)")
    parser.add_argument("--val-data",   nargs="+", default=[],
                        help="JSONL validation files")
    parser.add_argument("--test-data",  nargs="+", default=[],
                        help="JSONL test files for final evaluation")
    parser.add_argument("--base-model",
                        default="models/pretrained/leandojo-lean4-tacgen-byt5-small",
                        help="Pretrained checkpoint or HuggingFace model id")
    parser.add_argument("--output-dir", default="models/finetuned/",
                        help="Where to save the fine-tuned model")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--grad-accum", type=int,   default=1,
                        help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--fp16",       action="store_true",
                        help="Use fp16 (CUDA only)")
    parser.add_argument("--top-k",      type=int,   default=10,
                        help="k for top-k accuracy evaluation")
    parser.add_argument("--n-samples",  type=int,   default=None,
                        help="Limit test evaluation to this many examples")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training, only run top-k evaluation on --test-data")
    args = parser.parse_args()

    if not args.eval_only:
        if not args.train_data:
            parser.error("--train-data is required for training")
        if not args.val_data:
            parser.error("--val-data is required for training")
        train(
            train_paths=args.train_data,
            val_paths=args.val_data,
            base_model=args.base_model,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            learning_rate=args.lr,
            fp16=args.fp16,
        )
        eval_model = args.output_dir
    else:
        eval_model = args.base_model

    if args.test_data:
        evaluate_top_k(
            model_path=eval_model,
            test_paths=args.test_data,
            k=args.top_k,
            n_samples=args.n_samples,
        )


if __name__ == "__main__":
    main()
