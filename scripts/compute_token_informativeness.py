#!/usr/bin/env python3
"""
Compute per-token informativeness scores for Token-Level NPO.

For each token in the forget set, computes:
  r_t = log p_target(token_t | context) - log p_pretrained(token_t | context)

Tokens with high r_t are "forget-informative": the target model learned them
specifically from fine-tuning, while pretrained doesn't know them.
Tokens with r_t ≈ 0 are general language tokens both models handle similarly.

Token-Level NPO applies forget loss ONLY to high-r_t tokens, preserving
general language ability and reducing collateral damage to retain knowledge.

Output: saves scores to a .pt file that can be loaded during SIBL training.

Usage:
  python scripts/compute_token_informativeness.py --data_split News --n_samples 200
"""

import argparse
import os
import json
import logging
import time

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_token_logprobs(model, input_ids, device="cuda"):
    """Compute per-token log probabilities under the model.

    Returns tensor of shape (seq_len - 1,) with log p(token_t | tokens_{<t}).
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device))
        logits = outputs.logits  # (1, seq_len, vocab)

    # Shift: logits[t] predicts token[t+1]
    shifted_logits = logits[:, :-1, :]  # (1, seq_len-1, vocab)
    shifted_labels = input_ids[:, 1:].to(device)  # (1, seq_len-1)

    log_probs = F.log_softmax(shifted_logits, dim=-1)  # (1, seq_len-1, vocab)
    token_log_probs = log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)

    return token_log_probs.squeeze(0).cpu()  # (seq_len-1,)


def main():
    parser = argparse.ArgumentParser(description="Token informativeness for forget set")
    parser.add_argument("--target", default="muse-bench/MUSE-News_target")
    parser.add_argument("--pretrained", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_split", default="News")
    parser.add_argument("--n_samples", type=int, default=0,
                        help="Number of samples (0 = all)")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"saves/unlearn/_token_info_{args.data_split}.pt"

    t0 = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load forget data
    logger.info("Loading forget data...")
    ds = load_dataset(f"muse-bench/MUSE-{args.data_split}", "raw", split="forget")
    n_samples = len(ds) if args.n_samples == 0 else min(args.n_samples, len(ds))

    # Load target model
    logger.info(f"Loading target model: {args.target}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="sdpa"
    )
    target_model.eval()

    # Compute target log-probs
    logger.info("Computing target log-probs...")
    target_logprobs = []
    for i in range(n_samples):
        tokens = tokenizer(
            ds[i]["text"], return_tensors="pt",
            max_length=args.max_length, truncation=True
        )
        lp = compute_token_logprobs(target_model, tokens["input_ids"], args.device)
        target_logprobs.append(lp)
        if (i + 1) % 50 == 0:
            logger.info(f"  Target: {i+1}/{n_samples}")

    # Free target model
    del target_model
    torch.cuda.empty_cache()

    # Load pretrained model
    logger.info(f"Loading pretrained model: {args.pretrained}")
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained, torch_dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="sdpa"
    )
    pretrained_model.eval()

    # Compute pretrained log-probs
    logger.info("Computing pretrained log-probs...")
    pretrained_logprobs = []
    for i in range(n_samples):
        tokens = tokenizer(
            ds[i]["text"], return_tensors="pt",
            max_length=args.max_length, truncation=True
        )
        lp = compute_token_logprobs(pretrained_model, tokens["input_ids"], args.device)
        pretrained_logprobs.append(lp)
        if (i + 1) % 50 == 0:
            logger.info(f"  Pretrained: {i+1}/{n_samples}")

    del pretrained_model
    torch.cuda.empty_cache()

    # Compute informativeness: r_t = log p_target - log p_pretrained
    logger.info("Computing informativeness scores...")
    all_ratios = []
    sample_stats = []
    for i in range(n_samples):
        ratio = target_logprobs[i] - pretrained_logprobs[i]
        all_ratios.append(ratio)
        sample_stats.append({
            "n_tokens": len(ratio),
            "mean_ratio": ratio.mean().item(),
            "max_ratio": ratio.max().item(),
            "frac_informative_1": (ratio > 1.0).float().mean().item(),
            "frac_informative_2": (ratio > 2.0).float().mean().item(),
        })

    # Global statistics
    all_flat = torch.cat(all_ratios)
    logger.info(f"\nToken informativeness statistics ({len(all_flat)} total tokens):")
    logger.info(f"  Mean ratio: {all_flat.mean():.3f}")
    logger.info(f"  Std ratio:  {all_flat.std():.3f}")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        logger.info(f"  P{pct}: {np.percentile(all_flat.numpy(), pct):.3f}")
    logger.info(f"  Frac > 0: {(all_flat > 0).float().mean():.3f}")
    logger.info(f"  Frac > 1: {(all_flat > 1).float().mean():.3f}")
    logger.info(f"  Frac > 2: {(all_flat > 2).float().mean():.3f}")

    # Per-sample statistics
    mean_ratios = [s["mean_ratio"] for s in sample_stats]
    logger.info(f"\nPer-sample mean informativeness:")
    logger.info(f"  Min: {min(mean_ratios):.3f}, Max: {max(mean_ratios):.3f}, "
                f"Mean: {np.mean(mean_ratios):.3f}")

    # Save
    output = {
        "ratios": all_ratios,  # list of tensors, one per sample
        "target_logprobs": target_logprobs,
        "pretrained_logprobs": pretrained_logprobs,
        "sample_stats": sample_stats,
        "global_stats": {
            "mean": all_flat.mean().item(),
            "std": all_flat.std().item(),
            "n_tokens": len(all_flat),
            "n_samples": n_samples,
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(output, args.output)
    logger.info(f"\nSaved to {args.output}")

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
