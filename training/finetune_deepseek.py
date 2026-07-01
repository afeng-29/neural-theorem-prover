"""
LoRA fine-tuning for DeepSeek-Prover-V1.5-RL on miniF2F proof pairs.

Training data: JSON lines with {prompt, completion} where
  prompt = MINIF2F_PREAMBLE + theorem_statement with ':= by\n  '
  completion = full proof body (verified correct)

Only the completion tokens are trained on (SFT on proof only).
"""
import argparse, json, logging, os, sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger()

MINIF2F_PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n"
    "set_option maxHeartbeats 400000\n"
    "open BigOperators Real Nat Topology Finset\n\n"
)


class ProofDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path) as f:
            for line in f:
                rec = json.loads(line)
                self.samples.append((rec["prompt"], rec["completion"]))

        logger.info("Loaded %d training pairs from %s", len(self.samples), data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, completion = self.samples[idx]
        full_text = prompt + completion + self.tokenizer.eos_token

        enc_full = self.tokenizer(
            full_text, max_length=self.max_length, truncation=True,
            return_tensors="pt",
        )
        enc_prompt = self.tokenizer(
            prompt, max_length=self.max_length, truncation=True,
            return_tensors="pt",
        )

        input_ids = enc_full["input_ids"].squeeze(0)
        labels = input_ids.clone()
        # Mask prompt tokens — only train on completion
        prompt_len = enc_prompt["input_ids"].shape[1]
        labels[:prompt_len] = -100

        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": enc_full["attention_mask"].squeeze(0)}


def collate_fn(batch, pad_token_id: int):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long).fill_(pad_token_id)
    labels = torch.zeros(len(batch), max_len, dtype=torch.long).fill_(-100)
    attn = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        n = x["input_ids"].shape[0]
        input_ids[i, :n] = x["input_ids"]
        labels[i, :n] = x["labels"]
        attn[i, :n] = x["attention_mask"]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Base DeepSeek model path")
    parser.add_argument("--train-data", required=True, help="Training data JSONL")
    parser.add_argument("--output-dir", required=True, help="Output LoRA adapter directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    logger.info("Loading tokenizer from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model%s", " in 4-bit" if args.load_in_4bit else "")
    model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    dataset = ProofDataset(args.train_data, tokenizer, max_length=args.max_length)

    from functools import partial
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, total_steps // 10)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    accum_loss = 0.0

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                logger.info("epoch %d step %d/%d loss=%.4f lr=%.2e",
                            epoch + 1, global_step, total_steps,
                            accum_loss, scheduler.get_last_lr()[0])
                accum_loss = 0.0

        logger.info("=== Epoch %d done ===", epoch + 1)

    adapter_path = output_dir / "lora_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("LoRA adapter saved to %s", adapter_path)


if __name__ == "__main__":
    main()
