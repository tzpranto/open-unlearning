#!/usr/bin/env python3
"""
Fast Fisher computation for PerTA, matching the paper's methodology.

Two modes:
  --mode aggregate  (default, matches paper)
    Computes (∇L(D;θ))² — squared gradient of aggregate loss.
    Single accumulated backward pass over all samples. Very fast.

  --mode per_sample
    Computes E[g²] — average of squared per-sample gradients.
    True diagonal Fisher approximation. N backward passes, GPU-accumulated.

Key differences from perta_unlearn.py:
  - GPU accumulation (no CPU transfer per sample → ~2-3x faster)
  - Separate n_forget / n_retain (paper uses ALL of each set)
  - Progress bar with ETA
  - Applies PerTA and saves model after Fisher computation

Usage:
  # Paper-style (aggregate gradient squared, all samples):
  python scripts/perta_fisher_fast.py --mode aggregate --data_split News --lambdas 3.5

  # True Fisher (per-sample, GPU-accumulated):
  python scripts/perta_fisher_fast.py --mode per_sample --data_split News --lambdas 3.5
"""

import argparse
import os
import json
import shutil
import logging
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = "/datadrive/forked/open-unlearning"
SAVES = os.path.join(BASE, "saves/unlearn")


def compute_fisher_aggregate(model, tokenizer, texts, max_length=2048, device="cuda"):
    """Compute squared aggregate gradient: F_i = (∂L_total/∂θ_i)².

    Matches the PerTA paper methodology. Accumulates gradients over all samples
    via repeated backward() calls (PyTorch adds gradients by default).
    """
    model.eval()
    model.zero_grad()

    n = 0
    t0 = time.time()
    for idx, text in enumerate(texts):
        tokens = tokenizer(
            text, return_tensors="pt", max_length=max_length,
            truncation=True, padding=False
        )
        input_ids = tokens["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue

        outputs = model(input_ids=input_ids, labels=input_ids)
        # backward() ADDS to existing gradients — no zero_grad between samples
        outputs.loss.backward()
        n += 1

        if n % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / n
            eta = rate * (len(texts) - n)
            logger.info(f"  Aggregate Fisher: {n}/{len(texts)} samples, "
                        f"{rate:.2f}s/sample, ETA {eta:.0f}s ({eta/60:.1f}m)")

    # Square the accumulated gradients to get Fisher
    fisher = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Normalize by N, then square
                avg_grad = param.grad.data / max(n, 1)
                fisher[name] = (avg_grad ** 2).cpu()
                param.grad = None
            else:
                fisher[name] = torch.zeros_like(param.data, device="cpu")

    elapsed = time.time() - t0
    total = sum(f.sum().item() for f in fisher.values())
    n_params = sum(f.numel() for f in fisher.values())
    logger.info(f"  Aggregate Fisher done: {n} samples in {elapsed:.0f}s "
                f"({elapsed/n:.2f}s/sample), mean={total/max(n_params,1):.10f}")
    return fisher


def compute_fisher_per_sample(model, tokenizer, texts, max_length=2048, device="cuda"):
    """Compute true diagonal Fisher: F_i = (1/N) Σ_n (∂L_n/∂θ_i)².

    Accumulates squared gradients on GPU for speed, then moves to CPU at the end.
    """
    model.eval()

    # Initialize Fisher accumulator ON GPU
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param.data)  # stays on GPU

    n = 0
    t0 = time.time()
    for idx, text in enumerate(texts):
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
                    fisher[name] += param.grad.data ** 2  # accumulate on GPU
                    param.grad = None

        n += 1
        if n % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / n
            eta = rate * (len(texts) - n)
            logger.info(f"  Per-sample Fisher: {n}/{len(texts)} samples, "
                        f"{rate:.2f}s/sample, ETA {eta:.0f}s ({eta/60:.1f}m)")

    # Normalize and move to CPU
    fisher_cpu = {}
    with torch.no_grad():
        for name in list(fisher.keys()):
            fisher_cpu[name] = (fisher[name] / max(n, 1)).cpu()
            del fisher[name]
    del fisher
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    total = sum(f.sum().item() for f in fisher_cpu.values())
    n_params = sum(f.numel() for f in fisher_cpu.values())
    logger.info(f"  Per-sample Fisher done: {n} samples in {elapsed:.0f}s "
                f"({elapsed/n:.2f}s/sample), mean={total/max(n_params,1):.10f}")
    return fisher_cpu


def apply_perta(target_sd, pretrained_sd, fisher_forget, fisher_retain,
                lam, alpha, epsilon=1e-8):
    """Apply PerTA: theta = theta_target - lam * w * tau."""
    result = {}
    n_modified = 0
    total_w = 0.0
    n_w = 0

    for name in target_sd:
        theta_t = target_sd[name].float()
        theta_p = pretrained_sd[name].float()
        tau = theta_t - theta_p

        if name in fisher_forget and name in fisher_retain:
            ff = fisher_forget[name].float()
            fr = fisher_retain[name].float()
            w = ff / (ff + alpha * fr + epsilon)
            total_w += w.mean().item()
            n_w += 1
        else:
            w = torch.full_like(theta_t, 0.3)

        result[name] = (theta_t - lam * w * tau).to(target_sd[name].dtype)
        if tau.abs().max().item() > 1e-10:
            n_modified += 1

    logger.info(f"  PerTA applied: λ={lam:.2f}, α={alpha:.1f}, "
                f"mean_w={total_w/max(n_w,1):.4f}, modified {n_modified} params")
    return result


def save_model(state_dict, out_dir, reference_dir):
    """Save state dict in safetensors format."""
    os.makedirs(out_dir, exist_ok=True)

    for f in ["config.json", "generation_config.json", "special_tokens_map.json",
              "tokenizer.json", "tokenizer_config.json", "tokenizer.model"]:
        src = os.path.join(reference_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, out_dir)

    keys = sorted(state_dict.keys())
    n = len(keys)
    shard_size = (n + 2) // 3
    shards = [keys[i:i+shard_size] for i in range(0, n, shard_size)]

    weight_map = {}
    for shard_idx, shard_keys in enumerate(shards):
        shard_name = f"model-{shard_idx+1:05d}-of-{len(shards):05d}.safetensors"
        shard_dict = {k: state_dict[k] for k in shard_keys}
        save_file(shard_dict, os.path.join(out_dir, shard_name), metadata={"format": "pt"})
        for k in shard_keys:
            weight_map[k] = shard_name

    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in state_dict.values())},
        "weight_map": weight_map
    }
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"  Saved to {out_dir} ({len(shards)} shards)")


def main():
    parser = argparse.ArgumentParser(description="Fast PerTA Fisher + unlearning")
    parser.add_argument("--mode", choices=["aggregate", "per_sample"], default="aggregate",
                        help="aggregate = paper's (∇L)², per_sample = true E[g²]")
    parser.add_argument("--target", default="muse-bench/MUSE-News_target")
    parser.add_argument("--pretrained", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_split", default="News")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[3.5])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fisher_cache", default=None,
                        help="Override Fisher cache path")
    parser.add_argument("--skip_perta", action="store_true",
                        help="Only compute Fisher, skip PerTA application")
    parser.add_argument("--fisher_at_pretrained", action="store_true",
                        help="Compute Fisher at θ_0 (pretrained) instead of θ_target. "
                             "Paper Algorithm 1 Step 2.1 specifies θ_0.")
    args = parser.parse_args()

    t0 = time.time()

    # Cache path includes mode to distinguish aggregate vs per-sample
    suffix = "agg" if args.mode == "aggregate" else "ps"
    fisher_model_tag = "theta0" if args.fisher_at_pretrained else "tgt"
    fisher_cache = args.fisher_cache or os.path.join(
        SAVES, f"_perta_fisher_cache_{args.data_split}_{suffix}_{fisher_model_tag}_full.pt"
    )

    if os.path.exists(fisher_cache):
        logger.info(f"Loading cached Fisher from {fisher_cache}")
        cached = torch.load(fisher_cache, map_location="cpu", weights_only=True)
        fisher_forget = cached["forget"]
        fisher_retain = cached["retain"]
        logger.info(f"  Loaded Fisher: {len(fisher_forget)} forget params, "
                     f"{len(fisher_retain)} retain params")
    else:
        # Load model for Fisher computation
        fisher_model_name = args.pretrained if args.fisher_at_pretrained else args.target
        logger.info(f"Loading Fisher model: {fisher_model_name} "
                     f"({'θ_0 pretrained' if args.fisher_at_pretrained else 'θ_target'})")
        model = AutoModelForCausalLM.from_pretrained(
            fisher_model_name, torch_dtype=torch.bfloat16, device_map=args.device,
            attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load ALL forget and retain data
        logger.info("Loading data...")
        forget_ds = load_dataset(f"muse-bench/MUSE-{args.data_split}", "raw", split="forget")
        forget_texts = [row["text"] for row in forget_ds]
        retain_ds = load_dataset(f"muse-bench/MUSE-{args.data_split}", "raw", split="retain1")
        retain_texts = [row["text"] for row in retain_ds]
        logger.info(f"  Forget: {len(forget_texts)} samples, Retain: {len(retain_texts)} samples")
        logger.info(f"  Mode: {args.mode} (using ALL samples from each set)")

        compute_fn = compute_fisher_aggregate if args.mode == "aggregate" else compute_fisher_per_sample

        logger.info(f"Computing FORGET Fisher ({args.mode}) on {len(forget_texts)} samples...")
        fisher_forget = compute_fn(model, tokenizer, forget_texts, args.max_length, args.device)

        logger.info(f"Computing RETAIN Fisher ({args.mode}) on {len(retain_texts)} samples...")
        fisher_retain = compute_fn(model, tokenizer, retain_texts, args.max_length, args.device)

        # Save Fisher cache
        torch.save({
            "forget": fisher_forget, "retain": fisher_retain,
            "mode": args.mode,
            "n_forget": len(forget_texts), "n_retain": len(retain_texts),
        }, fisher_cache)
        logger.info(f"Fisher cached to {fisher_cache}")

        del model
        torch.cuda.empty_cache()

    fisher_elapsed = time.time() - t0
    logger.info(f"Fisher phase done in {fisher_elapsed:.0f}s ({fisher_elapsed/60:.1f}m)")

    if args.skip_perta:
        logger.info("--skip_perta set, stopping after Fisher computation")
        return

    # Load model state dicts for PerTA application
    logger.info(f"Loading target state dict: {args.target}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    target_sd = {k: v.clone() for k, v in target_model.state_dict().items()}
    del target_model

    logger.info(f"Loading pretrained state dict: {args.pretrained}")
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    pretrained_sd = {k: v.clone() for k, v in pretrained_model.state_dict().items()}
    pre_dir = pretrained_model.config._name_or_path
    if not os.path.isdir(pre_dir):
        from huggingface_hub import snapshot_download
        pre_dir = snapshot_download(args.pretrained, local_files_only=True)
    del pretrained_model

    # Apply PerTA for each lambda
    for lam in args.lambdas:
        out_name = f"perta_l{lam:.1f}_{suffix}_{fisher_model_tag}_full"
        out_dir = os.path.join(SAVES, out_name)

        if os.path.exists(out_dir) and any(
            f.endswith(".safetensors") and "model-" in f for f in os.listdir(out_dir)
        ):
            logger.info(f"[SKIP] {out_name} already exists")
            continue

        logger.info(f"Applying PerTA: λ={lam}, α={args.alpha}")
        result_sd = apply_perta(
            target_sd, pretrained_sd, fisher_forget, fisher_retain,
            lam=lam, alpha=args.alpha
        )
        save_model(result_sd, out_dir, pre_dir)
        del result_sd

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Print eval commands
    for lam in args.lambdas:
        out_name = f"perta_l{lam:.1f}_{suffix}_{fisher_model_tag}_full"
        logger.info(f"\nEval: python src/eval.py experiment=eval/muse/default.yaml "
                    f"data_split={args.data_split} task_name={out_name} model=Llama-2-7b-hf "
                    f"model.model_args.pretrained_model_name_or_path=saves/unlearn/{out_name} "
                    f"model.model_args.attn_implementation=sdpa "
                    f"retain_logs_path=saves/eval/muse_Llama-2-7b-hf_{args.data_split}_retrain/MUSE_EVAL.json")


if __name__ == "__main__":
    main()
