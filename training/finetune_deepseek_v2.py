"""
LoRA fine-tuning for DeepSeek-Prover-V1.5-RL using HuggingFace TRL SFTTrainer.

Designed for full-precision training on A100/H100 (BF16).
Also supports 4-bit QLoRA on smaller GPUs via --load-in-4bit.

Training data: JSONL with {prompt, completion} fields.
Only completion tokens are trained on (SFT on proof body only).
"""
import argparse, json, logging, os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger()


def load_jsonl(path: str):
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            # Concatenate prompt + completion as the "text" field for SFTTrainer
            records.append({"text": rec["prompt"] + rec["completion"],
                             "prompt": rec["prompt"],
                             "id": rec.get("id", "")})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Use 4-bit NF4 quantization (for GPUs < 40GB)")
    parser.add_argument("--no-bf16", action="store_true",
                        help="Disable BF16 (use FP16 instead, e.g. for V100)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=500)
    args = parser.parse_args()

    logger.info("Loading tokenizer from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Model loading
    use_bf16 = not args.no_bf16 and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    logger.info("Compute dtype: %s (BF16 supported: %s)", compute_dtype, torch.cuda.is_bf16_supported())

    model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if args.load_in_4bit:
        logger.info("Loading model in 4-bit NF4")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        logger.info("Loading model in full precision (%s)", compute_dtype)
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    # LoRA config — larger r/alpha for better capacity on bigger dataset
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # Load data
    logger.info("Loading training data from %s", args.train_data)
    records = load_jsonl(args.train_data)
    logger.info("  %d training examples", len(records))
    dataset = Dataset.from_list(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=25,
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=4,
        report_to="none",
        max_seq_length=args.max_length,
        # Train only on completion tokens (mask the prompt)
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_cfg,
        tokenizer=tokenizer,
    )

    trainer.model.print_trainable_parameters()
    logger.info("Starting training ...")
    trainer.train()

    adapter_path = output_dir / "lora_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("LoRA adapter saved to %s", adapter_path)


if __name__ == "__main__":
    main()
