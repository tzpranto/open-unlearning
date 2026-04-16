#!/usr/bin/env python3
"""
Importance-Sampled Fisher for PerTA
====================================

Two-pass approach:
  Pass 1: Compute per-sample gradient norm for all forget/retain samples (cheap — just scalar per sample)
  Pass 2: Select top-K samples by gradient norm, compute full Fisher only on those

This gives Fisher quality close to full-data Fisher using only ~20% of samples.

The key insight: samples with high gradient norms contribute most to the Fisher diagonal.
Samples with near-zero gradients barely affect the Fisher — they're "easy" samples that
the model already handles well. Importance sampling focuses compute on the "hard" samples.

Usage:
  python scripts/perta_importance_sampling.py --data_split News --n_select 178 --alpha 1.0
  (178 = 20% of 889 forget samples)
"""

import argparse
import os
import logging
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = "/datadrive/forked/open-unlearning"
SAVES = os.path.join(BASE, "saves/unlearn")


def compute_per_sample_importance(model, tokenizer, texts, max_length=2048, device="cuda"):
    """Pass 1: Compute per-sample gradient L2 norm (scalar per sample).

    Returns list of (index, grad_norm, text) sorted by grad_norm descending.
    This is cheap — only stores one scalar per sample, not full Fisher diagonal.
    """
    model.eval()
    sample_scores = []

    for idx, text in enumerate(texts):
        tokens = tokenizer(
            text, return_tensors="pt", max_length=max_length,
            truncation=True, padding=False
        )
        input_ids = tokens["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            sample_scores.append((idx, 0.0, text))
            continue

        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()

        # Compute total gradient L2 norm across all parameters
        grad_norm = 0.0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
                    param.grad = None
        grad_norm = grad_norm ** 0.5

        sample_scores.append((idx, grad_norm, text))

        if (idx + 1) % 50 == 0:
            logger.info(f"  Importance scoring: {idx+1}/{len(texts)} samples")

    # Sort by importance (descending)
    sample_scores.sort(key=lambda x: x[1], reverse=True)
    return sample_scores


def compute_fisher_on_selected(model, tokenizer, texts, max_length=2048, device="cuda"):
    """Pass 2: Standard Fisher computation on pre-selected texts."""
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param.data, device="cpu")

    n = 0
    for text in texts:
        tokens = tokenizer(
            text, return_tensors="pt", max_length=max_length,
            truncation=True, padding=False
        )
        input_ids = tokens["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue

        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.cpu() ** 2
                    param.grad = None

        n += 1
        if n % 16 == 0:
            logger.info(f"  Fisher: {n}/{len(texts)} selected samples processed")

    for name in fisher:
        fisher[name] /= max(n, 1)

    logger.info(f"  Fisher done: {n} samples")
    return fisher


def main():
    parser = argparse.ArgumentParser(description="Importance-sampled Fisher for PerTA")
    parser.add_argument("--target", default="muse-bench/MUSE-News_target")
    parser.add_argument("--pretrained", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_split", default="News")
    parser.add_argument("--n_select", type=int, default=178,
                        help="Number of top samples to select (default: 20% of 889)")
    parser.add_argument("--n_select_retain", type=int, default=None,
                        help="Number of retain samples (default: same as n_select)")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.n_select_retain is None:
        args.n_select_retain = args.n_select

    t0 = time.time()

    # Load model
    logger.info(f"Loading target model: {args.target}")
    model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    logger.info("Loading datasets...")
    forget_ds = load_dataset(f"muse-bench/MUSE-{args.data_split}", "raw", split="forget")
    forget_texts = [row["text"] for row in forget_ds]
    retain_ds = load_dataset(f"muse-bench/MUSE-{args.data_split}", "raw", split="retain1")
    retain_texts = [row["text"] for row in retain_ds]

    logger.info(f"Forget: {len(forget_texts)} samples, Retain: {len(retain_texts)} samples")
    logger.info(f"Selecting top {args.n_select} forget, {args.n_select_retain} retain by importance")

    # Pass 1: Score all samples
    logger.info("=" * 50)
    logger.info("Pass 1: Scoring FORGET samples by gradient importance...")
    forget_scores = compute_per_sample_importance(
        model, tokenizer, forget_texts, max_length=args.max_length, device=args.device
    )

    logger.info("Pass 1: Scoring RETAIN samples by gradient importance...")
    retain_scores = compute_per_sample_importance(
        model, tokenizer, retain_texts, max_length=args.max_length, device=args.device
    )

    # Log importance distribution
    forget_norms = [s[1] for s in forget_scores]
    retain_norms = [s[1] for s in retain_scores]
    logger.info(f"  Forget grad norms: min={min(forget_norms):.4f}, max={max(forget_norms):.4f}, "
                f"mean={sum(forget_norms)/len(forget_norms):.4f}")
    logger.info(f"  Retain grad norms: min={min(retain_norms):.4f}, max={max(retain_norms):.4f}, "
                f"mean={sum(retain_norms)/len(retain_norms):.4f}")

    # Select top-K
    selected_forget = [s[2] for s in forget_scores[:args.n_select]]
    selected_retain = [s[2] for s in retain_scores[:args.n_select_retain]]

    top_forget_norm = sum(s[1] for s in forget_scores[:args.n_select])
    total_forget_norm = sum(forget_norms)
    logger.info(f"  Top {args.n_select} forget samples capture "
                f"{top_forget_norm/total_forget_norm*100:.1f}% of total gradient mass")

    # Pass 2: Compute Fisher on selected samples
    logger.info("=" * 50)
    logger.info(f"Pass 2: Computing FORGET Fisher on {len(selected_forget)} selected samples...")
    fisher_forget = compute_fisher_on_selected(
        model, tokenizer, selected_forget, max_length=args.max_length, device=args.device
    )

    logger.info(f"Pass 2: Computing RETAIN Fisher on {len(selected_retain)} selected samples...")
    fisher_retain = compute_fisher_on_selected(
        model, tokenizer, selected_retain, max_length=args.max_length, device=args.device
    )

    # Save
    out_path = os.path.join(
        SAVES, f"_perta_fisher_cache_{args.data_split}_imp{args.n_select}.pt"
    )
    torch.save({"forget": fisher_forget, "retain": fisher_retain}, out_path)
    logger.info(f"Fisher cached to {out_path}")

    # Also save importance scores for analysis
    scores_path = os.path.join(
        SAVES, f"_perta_importance_scores_{args.data_split}.pt"
    )
    torch.save({
        "forget_scores": [(s[0], s[1]) for s in forget_scores],
        "retain_scores": [(s[0], s[1]) for s in retain_scores],
    }, scores_path)
    logger.info(f"Importance scores saved to {scores_path}")

    # Free GPU
    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
