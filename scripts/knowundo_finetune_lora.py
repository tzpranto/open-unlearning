"""Fine-tune Llama-2-7b-chat on KnowUnDo with LoRA, matching original paper setup.

LoRA config: r=8, alpha=16, dropout=0.1, target=all-linear (excl. lm_head)
Training: lr=1e-4, epochs=10, BS=2, grad_accum=16 (eff. BS=32), weight_decay=1e-4
Saves merged full model for compatibility with all unlearn methods.
"""
import sys
sys.path.insert(0, "src")

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from data.knowundo import KnowUnDoDataset
from data.collators import DataCollatorForSupervisedDataset


def find_all_linear_names(model):
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            names.append(name.split(".")[-1])
    return list(set(names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="copyright")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"saves/unlearn/knowundo_Llama-2-7b-chat_{args.domain}_ft"

    base_model_id = "meta-llama/Llama-2-7b-chat-hf"

    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules = find_all_linear_names(model)
    print(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    template_args = {
        "apply_chat_template": False,
        "user_start_tag": "[INST] ",
        "user_end_tag": " [/INST]",
        "asst_start_tag": "",
        "asst_end_tag": " ",
    }

    print(f"Loading KnowUnDo {args.domain} dataset (unlearn + retention)...")
    train_dataset = KnowUnDoDataset(
        hf_args={"path": "zjunlp/KnowUnDo", "name": args.domain, "split": "unlearn", "trust_remote_code": True},
        template_args=template_args,
        tokenizer=tokenizer,
        subset="train",
        additional_splits=["retention"],
        max_length=500,
    )
    print(f"Train dataset size: {len(train_dataset)}")

    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=1e-4,
        num_train_epochs=10,
        bf16=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="no",
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    print("Merging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}")
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
