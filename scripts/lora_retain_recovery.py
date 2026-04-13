#!/usr/bin/env python3
"""
Two-Stage LoRA Retain Recovery for LLM Unlearning

Stage 1 (already done): Aggressive unlearning via G1/SIBL → good forget, weak retain
Stage 2 (this script): Add LoRA adapters, train on retain data only

Key insight: LoRA updates a low-rank subspace. If retain knowledge lives in a different
subspace than forget knowledge, LoRA can recover retain without re-teaching forget.
This decouples the forget-retain tradeoff that traps gradient methods on the CE frontier.

Usage:
  python scripts/lora_retain_recovery.py \
    --base_model saves/unlearn/ablation_G1_npo_weak_steering \
    --lora_rank 8 --epochs 3 --lr 2e-4

  # Multi-rank sweep:
  python scripts/lora_retain_recovery.py \
    --base_model saves/unlearn/ablation_G1_npo_weak_steering \
    --lora_ranks 4 8 16 --epochs 3 --lr 2e-4
"""

import argparse
import os
import json
import logging
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = "/datadrive/forked/open-unlearning"
SAVES = os.path.join(BASE, "saves/unlearn")


def load_retain_data(tokenizer, data_split="News", max_length=2048):
    """Load and tokenize MUSE retain data."""
    ds = load_dataset(f"muse-bench/MUSE-{data_split}", "raw", split="retain1")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    # Filter out very short sequences
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) >= 64)
    logger.info(f"Retain dataset: {len(tokenized)} samples")
    return tokenized


def train_lora(
    base_model_path,
    lora_rank=8,
    lora_alpha=None,
    lora_dropout=0.05,
    target_modules=None,
    data_split="News",
    epochs=3,
    lr=2e-4,
    batch_size=4,
    grad_accum=4,
    max_length=2048,
    output_name=None,
):
    """Add LoRA adapters to base model and train on retain data."""

    if lora_alpha is None:
        lora_alpha = lora_rank * 2  # standard scaling

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    if output_name is None:
        base_name = os.path.basename(base_model_path)
        output_name = f"lora_r{lora_rank}_{base_name}"

    out_dir = os.path.join(SAVES, output_name)
    merged_dir = os.path.join(SAVES, f"{output_name}_merged")

    # Check if already done
    if os.path.exists(merged_dir) and any(
        f.endswith(".safetensors") and "model-" in f
        for f in os.listdir(merged_dir)
    ):
        logger.info(f"[SKIP] {output_name} already merged at {merged_dir}")
        return merged_dir

    # Load base model
    logger.info(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Add LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    # Enable input gradients for gradient checkpointing compatibility
    model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"LoRA rank={lora_rank}: {trainable:,} trainable / {total:,} total "
                f"({100*trainable/total:.2f}%)")

    # Load retain data
    dataset = load_retain_data(tokenizer, data_split, max_length)

    # Training args
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="no",  # We only care about final merged model
        report_to="none",
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info(f"Training LoRA (rank={lora_rank}, epochs={epochs}, lr={lr})")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.0f}s")

    # Merge LoRA weights back into base model
    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # Save merged model
    logger.info(f"Saving merged model to {merged_dir}")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    # Create empty evals dir
    os.makedirs(os.path.join(merged_dir, "evals"), exist_ok=True)

    # Save metadata
    meta = {
        "base_model": base_model_path,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "epochs": epochs,
        "lr": lr,
        "train_time_s": elapsed,
        "trainable_params": trainable,
        "total_params": total,
    }
    with open(os.path.join(merged_dir, "lora_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Cleanup adapter-only save
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)

    return merged_dir


def main():
    parser = argparse.ArgumentParser(description="Two-Stage LoRA retain recovery")
    parser.add_argument("--base_model", default="saves/unlearn/ablation_G1_npo_weak_steering",
                        help="Path to unlearned model (Stage 1 output)")
    parser.add_argument("--data_split", default="News", help="MUSE data split")
    parser.add_argument("--lora_ranks", nargs="+", type=int, default=[4, 8, 16],
                        help="LoRA rank values to sweep")
    parser.add_argument("--lora_alpha_ratio", type=float, default=2.0,
                        help="lora_alpha = rank * ratio")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    t0 = time.time()
    results = []

    for rank in args.lora_ranks:
        logger.info(f"\n{'='*60}")
        logger.info(f"LoRA rank={rank}")
        logger.info(f"{'='*60}")

        merged_dir = train_lora(
            base_model_path=args.base_model,
            lora_rank=rank,
            lora_alpha=int(rank * args.lora_alpha_ratio),
            data_split=args.data_split,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_length=args.max_length,
        )
        results.append((rank, merged_dir))

    elapsed = time.time() - t0
    logger.info(f"\nAll LoRA variants trained in {elapsed:.0f}s")
    logger.info("\nTo evaluate, run:")
    for rank, merged_dir in results:
        name = os.path.basename(merged_dir)
        logger.info(
            f"  python src/eval.py experiment=eval/muse/default.yaml "
            f"data_split={args.data_split} task_name={name} model=Llama-2-7b-hf "
            f"model.model_args.pretrained_model_name_or_path={merged_dir} "
            f"model.model_args.attn_implementation=sdpa "
            f"paths.output_dir={merged_dir}/evals "
            f"retain_logs_path=saves/eval/muse_Llama-2-7b-hf_{args.data_split}_retrain/MUSE_EVAL.json"
        )


if __name__ == "__main__":
    main()
