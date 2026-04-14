#!/usr/bin/env python3
"""
Derive importance-sampled Fisher from a full Fisher cache.

Instead of a separate GPU pass to score samples, we can:
1. Load the full Fisher cache (computed on all samples)
2. Re-run Fisher on individual samples to get per-sample contributions
3. Select top-K by contribution and recompute Fisher on just those

BUT this still needs GPU for per-sample Fisher. So instead, this script
takes a simpler approach: compute per-sample CE loss on the target model
as a proxy for importance. High-loss samples = hard to model = high Fisher.

Usage:
  python scripts/fisher_importance_from_cache.py --data_split News --n_select 178
"""

import argparse
import os
import logging
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = "/datadrive/forked/open-unlearning"
SAVES = os.path.join(BASE, "saves/unlearn")


def score_samples_by_loss(model, tokenizer, texts, max_length=2048, device="cuda"):
    """Score each sample by its CE loss (proxy for Fisher importance)."""
    model.eval()
    scores = []

    for idx, text in enumerate(texts):
        tokens = tokenizer(
            text, return_tensors="pt", max_length=max_length,
            truncation=True, padding=False
        )
        input_ids = tokens["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            scores.append((idx, 0.0))
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()

        scores.append((idx, loss))

        if (idx + 1) % 100 == 0:
            logger.info(f"  Scoring: {idx+1}/{len(texts)}")

    return scores


def compute_fisher_on_indices(model, tokenizer, texts, indices, max_length=2048, device="cuda"):
    """Compute per-sample Fisher diagonal on selected indices. GPU-accumulated."""
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param.data)  # GPU accumulation

    n = 0
    t0 = time.time()
    for idx in indices:
        text = texts[idx]
        tokens = tokenizer(
            text, return_tensors="pt", max_length=max_length,
            truncation=True, padding=False
        )
        input_ids = tokens["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue

        model.zero_grad()
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
                    param.grad = None

        n += 1
        if n % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / n
            eta = rate * (len(indices) - n)
            logger.info(f"  Fisher: {n}/{len(indices)} selected, "
                        f"{rate:.2f}s/sample, ETA {eta:.0f}s")

    # Normalize and move to CPU
    fisher_cpu = {}
    with torch.no_grad():
        for name in list(fisher.keys()):
            fisher_cpu[name] = (fisher[name] / max(n, 1)).cpu()
            del fisher[name]
    del fisher
    torch.cuda.empty_cache()

    logger.info(f"  Fisher done: {n} samples from {len(indices)} selected")
    return fisher_cpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="muse-bench/MUSE-News_target")
    parser.add_argument("--pretrained", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_split", default="News")
    parser.add_argument("--n_select", type=int, default=178,
                        help="Top-K forget samples to select (178 = 20% of 889)")
    parser.add_argument("--n_select_retain", type=int, default=356,
                        help="Top-K retain samples (356 = 20% of 1777)")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[3.5])
    args = parser.parse_args()

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
    forget_ds = load_dataset(f"muse-bench/MUSE-{args.data_split}", "raw", split="forget")
    forget_texts = [row["text"] for row in forget_ds]
    retain_ds = load_dataset(f"muse-bench/MUSE-{args.data_split}", "raw", split="retain1")
    retain_texts = [row["text"] for row in retain_ds]

    logger.info(f"Forget: {len(forget_texts)}, Retain: {len(retain_texts)}")

    # Step 1: Score all samples by loss (fast — no backward pass needed)
    logger.info("Scoring forget samples by CE loss...")
    forget_scores = score_samples_by_loss(model, tokenizer, forget_texts, args.max_length, args.device)
    logger.info("Scoring retain samples by CE loss...")
    retain_scores = score_samples_by_loss(model, tokenizer, retain_texts, args.max_length, args.device)

    # Sort by loss descending (high loss = hard = important)
    forget_scores.sort(key=lambda x: x[1], reverse=True)
    retain_scores.sort(key=lambda x: x[1], reverse=True)

    forget_losses = [s[1] for s in forget_scores]
    retain_losses = [s[1] for s in retain_scores]
    logger.info(f"Forget losses: min={min(forget_losses):.3f}, max={max(forget_losses):.3f}, "
                f"mean={sum(forget_losses)/len(forget_losses):.3f}")
    logger.info(f"Retain losses: min={min(retain_losses):.3f}, max={max(retain_losses):.3f}, "
                f"mean={sum(retain_losses)/len(retain_losses):.3f}")

    # Step 2: Select top-K indices
    forget_indices = [s[0] for s in forget_scores[:args.n_select]]
    retain_indices = [s[0] for s in retain_scores[:args.n_select_retain]]

    top_loss_sum = sum(s[1] for s in forget_scores[:args.n_select])
    total_loss_sum = sum(forget_losses)
    logger.info(f"Top {args.n_select} forget samples ({args.n_select/len(forget_texts)*100:.0f}%) "
                f"capture {top_loss_sum/total_loss_sum*100:.1f}% of total loss mass")

    # Step 3: Compute Fisher on selected samples
    logger.info(f"Computing FORGET Fisher on {len(forget_indices)} selected samples...")
    fisher_forget = compute_fisher_on_indices(
        model, tokenizer, forget_texts, forget_indices, args.max_length, args.device
    )
    logger.info(f"Computing RETAIN Fisher on {len(retain_indices)} selected samples...")
    fisher_retain = compute_fisher_on_indices(
        model, tokenizer, retain_texts, retain_indices, args.max_length, args.device
    )

    # Save Fisher cache
    pct = int(args.n_select / len(forget_texts) * 100)
    cache_path = os.path.join(SAVES, f"_perta_fisher_cache_{args.data_split}_imp{pct}pct.pt")
    torch.save({"forget": fisher_forget, "retain": fisher_retain}, cache_path)
    logger.info(f"Fisher cached to {cache_path}")

    # Save importance scores
    scores_path = os.path.join(SAVES, f"_perta_importance_scores_{args.data_split}.pt")
    torch.save({
        "forget_scores": forget_scores,
        "retain_scores": retain_scores,
    }, scores_path)
    logger.info(f"Importance scores saved to {scores_path}")

    # Step 4: Apply PerTA and save models
    import sys
    sys.path.insert(0, os.path.join(BASE, "scripts"))
    from perta_unlearn import apply_perta, save_model

    # Load state dicts
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    target_sd = {k: v.clone() for k, v in target_model.state_dict().items()}
    del target_model

    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    pretrained_sd = {k: v.clone() for k, v in pretrained_model.state_dict().items()}
    pre_dir = pretrained_model.config._name_or_path
    if not os.path.isdir(pre_dir):
        from huggingface_hub import snapshot_download
        pre_dir = snapshot_download(args.pretrained, local_files_only=True)
    del pretrained_model

    del model
    torch.cuda.empty_cache()

    for lam in args.lambdas:
        out_dir = os.path.join(SAVES, f"perta_l{lam:.1f}_imp{pct}pct")
        if os.path.exists(out_dir) and any(
            f.endswith(".safetensors") and "model-" in f for f in os.listdir(out_dir)
        ):
            logger.info(f"[SKIP] {out_dir} already exists")
            continue

        result_sd = apply_perta(target_sd, pretrained_sd, fisher_forget, fisher_retain,
                                lam=lam, alpha=1.0)
        save_model(result_sd, out_dir, pre_dir)
        del result_sd

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
