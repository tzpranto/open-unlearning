#!/usr/bin/env python3
"""
PerTA-Masked: Binary-mask variant of PerTA for S-BiAL connection.

Standard PerTA:  θ = θ_target - λ * w * τ   (continuous w)
Masked PerTA:    θ = θ_target - λ * m * τ   (binary m = 1{w > threshold})

This connects PerTA to S-BiAL's mask formulation (Eq. 5-7) — the Fisher mask
selects WHICH parameters to negate (forget-dominant), freezing all others.

Usage:
  python scripts/perta_masked.py --lambdas 1.5 2.0 3.0 3.5 --thresholds 0.3 0.5
"""

import argparse
import os
import json
import shutil
import logging
import time

import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = "/datadrive/forked/open-unlearning"
SAVES = os.path.join(BASE, "saves/unlearn")


def apply_perta_masked(target_sd, pretrained_sd, fisher_forget, fisher_retain,
                       lam, alpha, threshold, epsilon=1e-8):
    """Apply masked PerTA: θ = θ_target - λ * m * τ where m = 1{w > threshold}.

    Returns new state dict + mask statistics.
    """
    result = {}
    n_negated = 0
    n_frozen = 0
    n_total = 0

    for name in target_sd:
        theta_t = target_sd[name].float()
        theta_p = pretrained_sd[name].float()
        tau = theta_t - theta_p

        if name in fisher_forget and name in fisher_retain:
            ff = fisher_forget[name].float()
            fr = fisher_retain[name].float()
            w = ff / (ff + alpha * fr + epsilon)
            m = (w > threshold).float()
            n_neg = m.sum().item()
            n_tot = m.numel()
            n_negated += n_neg
            n_frozen += (n_tot - n_neg)
            n_total += n_tot
        else:
            # Non-trainable params — freeze (don't negate)
            m = torch.zeros_like(theta_t)
            n_total += m.numel()
            n_frozen += m.numel()

        result[name] = (theta_t - lam * m * tau).to(target_sd[name].dtype)

    pct = 100 * n_negated / max(n_total, 1)
    logger.info(f"  Masked PerTA: λ={lam:.2f}, threshold={threshold:.2f}, "
                f"negated={n_negated:,} ({pct:.1f}%), frozen={n_frozen:,}")
    return result, {"n_negated": n_negated, "n_frozen": n_frozen, "n_total": n_total, "pct_negated": pct}


def save_model(state_dict, out_dir, reference_dir):
    """Save state dict in safetensors format, copying config from reference."""
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

    index = {"metadata": {"total_size": sum(v.numel() * v.element_size() for v in state_dict.values())},
             "weight_map": weight_map}
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"  Saved to {out_dir} ({len(shards)} shards)")


def main():
    parser = argparse.ArgumentParser(description="PerTA-Masked: binary threshold variant")
    parser.add_argument("--target", default="muse-bench/MUSE-News_target")
    parser.add_argument("--pretrained", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[1.5, 2.0, 3.0, 3.5])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--fisher_cache", default=None)
    args = parser.parse_args()

    t0 = time.time()

    # Load Fisher cache
    fisher_cache = args.fisher_cache or os.path.join(
        SAVES, "_perta_fisher_cache_News_n64.pt"
    )
    logger.info(f"Loading Fisher cache from {fisher_cache}")
    cached = torch.load(fisher_cache, map_location="cpu", weights_only=True)
    fisher_forget = cached["forget"]
    fisher_retain = cached["retain"]

    # Fisher mask statistics at various thresholds
    logger.info("\nFisher mask w statistics:")
    w_all = []
    for name in fisher_forget:
        if name in fisher_retain:
            ff = fisher_forget[name].float()
            fr = fisher_retain[name].float()
            w = ff / (ff + args.alpha * fr + 1e-8)
            w_all.append(w.flatten())
    w_cat = torch.cat(w_all)
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pct = 100 * (w_cat > th).float().mean().item()
        logger.info(f"  w > {th:.1f}: {pct:.1f}% of params")

    # Load models
    logger.info(f"Loading target state dict: {args.target}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    target_sd = {k: v.clone() for k, v in target_model.state_dict().items()}
    target_config_dir = target_model.config._name_or_path
    if not os.path.isdir(target_config_dir):
        from huggingface_hub import snapshot_download
        target_config_dir = snapshot_download(args.target, local_files_only=True)
    del target_model

    logger.info(f"Loading pretrained state dict: {args.pretrained}")
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    pretrained_sd = {k: v.clone() for k, v in pretrained_model.state_dict().items()}
    pretrained_config_dir = pretrained_model.config._name_or_path
    if not os.path.isdir(pretrained_config_dir):
        from huggingface_hub import snapshot_download
        pretrained_config_dir = snapshot_download(args.pretrained, local_files_only=True)
    del pretrained_model

    # Sweep
    all_stats = {}
    for th in args.thresholds:
        for lam in args.lambdas:
            out_name = f"perta_masked_l{lam:.1f}_th{th:.1f}"
            out_dir = os.path.join(SAVES, out_name)

            if os.path.exists(out_dir) and any(
                f.endswith(".safetensors") and "model-" in f for f in os.listdir(out_dir)
            ):
                logger.info(f"[SKIP] {out_name} already exists")
                continue

            logger.info(f"\nApplying masked PerTA: λ={lam}, threshold={th}")
            result_sd, stats = apply_perta_masked(
                target_sd, pretrained_sd, fisher_forget, fisher_retain,
                lam=lam, alpha=args.alpha, threshold=th
            )

            save_model(result_sd, out_dir, pretrained_config_dir)
            os.makedirs(os.path.join(out_dir, "evals"), exist_ok=True)

            # Save config for reproducibility (separate file, don't overwrite model config.json)
            perta_config = {
                "method": "perta_masked",
                "lambda": lam, "alpha": args.alpha, "threshold": th,
                "mask_stats": stats,
                "target": args.target, "pretrained": args.pretrained,
            }
            with open(os.path.join(out_dir, "perta_masked_config.json"), "w") as f:
                json.dump(perta_config, f, indent=2)

            all_stats[out_name] = stats
            del result_sd

    elapsed = time.time() - t0
    logger.info(f"\nPerTA-Masked complete. Total time: {elapsed:.0f}s")
    logger.info("\nTo evaluate:")
    for th in args.thresholds:
        for lam in args.lambdas:
            name = f"perta_masked_l{lam:.1f}_th{th:.1f}"
            logger.info(f"  python src/eval.py experiment=eval/muse/default.yaml "
                        f"data_split=News task_name={name} model=Llama-2-7b-hf "
                        f"model.model_args.pretrained_model_name_or_path=saves/unlearn/{name} "
                        f"model.model_args.attn_implementation=sdpa "
                        f"paths.output_dir=saves/unlearn/{name}/evals "
                        f"retain_logs_path=saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json")


if __name__ == "__main__":
    main()
