"""
Per-layer gradient cosine similarity between forget and retain.
Answers: which LoRA layers have aligned gradients (dangerous to modify for forget)
vs opposed gradients (safe to modify)?
"""
import torch
import torch.nn.functional as F
import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from data.pretraining import PretrainingDataset
from data.collators import DataCollatorForSupervisedDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_mean_grad(model, dataloader, device, n_batches=50):
    """Compute mean gradient per LoRA layer over n_batches."""
    model.train()
    model.enable_adapter_layers()
    grad_accum = {}
    count = 0

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", input_ids).to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and "lora_" in name:
                if name not in grad_accum:
                    grad_accum[name] = torch.zeros_like(param.grad)
                grad_accum[name] += param.grad.detach().clone()
        count += 1

    for name in grad_accum:
        grad_accum[name] /= count
    return grad_accum


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "muse-bench/MUSE-News_target",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Wrapping with LoRA...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    logger.info("Loading data...")
    template_args = {}
    forget_ds = PretrainingDataset(
        hf_args={"path": "muse-bench/MUSE-News", "name": "raw", "split": "forget"},
        template_args=template_args, tokenizer=tokenizer, text_key="text", max_length=1024,
    )
    retain_ds = PretrainingDataset(
        hf_args={"path": "muse-bench/MUSE-News", "name": "raw", "split": "retain1"},
        template_args=template_args, tokenizer=tokenizer, text_key="text", max_length=1024,
    )

    collator = DataCollatorForSupervisedDataset(tokenizer)
    forget_loader = DataLoader(forget_ds, batch_size=4, shuffle=True, collate_fn=collator)
    retain_loader = DataLoader(retain_ds, batch_size=4, shuffle=True, collate_fn=collator)

    logger.info("Computing forget gradients (50 batches)...")
    forget_grads = compute_mean_grad(model, forget_loader, device, n_batches=50)

    logger.info("Computing retain gradients (50 batches)...")
    retain_grads = compute_mean_grad(model, retain_loader, device, n_batches=50)

    # Compute per-layer cosine similarity
    logger.info("\n" + "=" * 70)
    logger.info("Per-layer gradient cosine similarity (forget vs retain)")
    logger.info("=" * 70)

    # Group by transformer layer
    layer_cosines = {}
    for name in sorted(forget_grads.keys()):
        if name in retain_grads:
            fg = forget_grads[name].flatten().float()
            rg = retain_grads[name].flatten().float()
            cos = F.cosine_similarity(fg.unsqueeze(0), rg.unsqueeze(0)).item()

            # Extract layer number
            parts = name.split(".")
            layer_num = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    layer_num = int(parts[i + 1])
                    break

            if layer_num is not None:
                if layer_num not in layer_cosines:
                    layer_cosines[layer_num] = []
                layer_cosines[layer_num].append((name.split(".")[-2], cos))

            logger.info(f"  {name:80s} cos={cos:+.4f}")

    # Summary by layer
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: Per transformer layer (avg across LoRA modules)")
    logger.info("=" * 70)

    layer_avg = {}
    for layer_num in sorted(layer_cosines.keys()):
        modules = layer_cosines[layer_num]
        avg_cos = sum(c for _, c in modules) / len(modules)
        layer_avg[layer_num] = avg_cos
        module_str = ", ".join(f"{m}={c:+.3f}" for m, c in modules)
        tag = "SAFE" if avg_cos < 0.0 else ("CAUTION" if avg_cos < 0.3 else "DANGER")
        logger.info(f"  Layer {layer_num:2d}: avg={avg_cos:+.4f} [{tag}]  ({module_str})")

    # Recommendations
    safe_layers = [l for l, c in layer_avg.items() if c < 0.0]
    caution_layers = [l for l, c in layer_avg.items() if 0.0 <= c < 0.3]
    danger_layers = [l for l, c in layer_avg.items() if c >= 0.3]

    logger.info(f"\nSAFE (cos<0, forget-only):   {safe_layers}")
    logger.info(f"CAUTION (0<cos<0.3):         {caution_layers}")
    logger.info(f"DANGER (cos>0.3, shared):    {danger_layers}")
    logger.info(f"\nRecommendation: freeze_outer_layers={danger_layers} during forget steps")


if __name__ == "__main__":
    main()
