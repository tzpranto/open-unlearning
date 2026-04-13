#!/usr/bin/env python3
"""
PerTA: Per-parameter Task Arithmetic for LLM Unlearning

  theta_final = theta_target - lambda * w * tau

where:
  tau = theta_target - theta_pretrained  (task vector: what fine-tuning learned)
  w_i = F_forget_i / (F_forget_i + alpha * F_retain_i + eps)  (per-parameter mask)
  F_forget/F_retain = diagonal Fisher on forget/retain data (computed on target model)

The key insight vs uniform interpolation (which stays ON the CE frontier):
Fisher weighting makes the negation SELECTIVE — parameters important for forget
get strongly negated while parameters important for retain are preserved. This
breaks the entanglement that traps gradient-based methods on the CE frontier.

Usage:
  python scripts/perta_unlearn.py --lambdas 0.5 1.0 1.5 2.0 --alpha 1.0 --n_samples 64
"""

import argparse
import os
import json
import shutil
import logging
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = "/datadrive/forked/open-unlearning"
SAVES = os.path.join(BASE, "saves/unlearn")


def compute_fisher(model, tokenizer, texts, n_samples=64, max_length=2048, device="cuda"):
    """Compute diagonal Fisher information E[g_i^2] for each parameter.

    Args:
        model: target model (will not be modified)
        tokenizer: tokenizer for encoding text
        texts: iterable of text strings
        n_samples: number of samples to use
        max_length: max token length per sample
        device: compute device
    Returns:
        dict of {param_name: fisher_diagonal_tensor} on CPU
    """
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param.data, device="cpu")

    n = 0
    for text in texts:
        if n >= n_samples:
            break
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
            logger.info(f"  Fisher: {n}/{n_samples} samples processed")

    for name in fisher:
        fisher[name] /= max(n, 1)

    total = sum(f.sum().item() for f in fisher.values())
    n_params = sum(f.numel() for f in fisher.values())
    logger.info(f"  Fisher done: {n} samples, mean={total/max(n_params,1):.8f}")
    return fisher


def apply_perta(target_sd, pretrained_sd, fisher_forget, fisher_retain,
                lam, alpha, epsilon=1e-8):
    """Apply PerTA formula: theta = theta_target - lam * w * tau.

    w_i = F_forget_i / (F_forget_i + alpha * F_retain_i + eps)
    tau_i = theta_target_i - theta_pretrained_i

    Returns new state dict.
    """
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
            # Non-trainable params (embeddings, norms) — use mild uniform negation
            w = torch.full_like(theta_t, 0.3)

        result[name] = (theta_t - lam * w * tau).to(target_sd[name].dtype)

        if tau.abs().max().item() > 1e-10:
            n_modified += 1

    logger.info(f"  PerTA applied: lambda={lam:.2f}, alpha={alpha:.1f}, "
                f"mean_w={total_w/max(n_w,1):.4f}, modified {n_modified} params")
    return result


def save_model(state_dict, out_dir, reference_dir):
    """Save state dict in safetensors format, copying config from reference."""
    os.makedirs(out_dir, exist_ok=True)

    # Copy config files from reference model
    for f in ["config.json", "generation_config.json", "special_tokens_map.json",
              "tokenizer.json", "tokenizer_config.json", "tokenizer.model"]:
        src = os.path.join(reference_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, out_dir)

    # Save weights — split into 3 shards for consistency with other saves
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

    # Write index
    index = {"metadata": {"total_size": sum(v.numel() * v.element_size() for v in state_dict.values())},
             "weight_map": weight_map}
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"  Saved to {out_dir} ({len(shards)} shards)")


def main():
    parser = argparse.ArgumentParser(description="PerTA unlearning")
    parser.add_argument("--target", default="muse-bench/MUSE-News_target",
                        help="Target model (fine-tuned on all data)")
    parser.add_argument("--pretrained", default="meta-llama/Llama-2-7b-hf",
                        help="Pretrained base model")
    parser.add_argument("--data_split", default="News", help="MUSE data split")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 3.0],
                        help="Lambda values to sweep")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Retain Fisher scaling (higher = more retain protection)")
    parser.add_argument("--n_samples", type=int, default=64,
                        help="Number of samples for Fisher estimation")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max token length for Fisher samples")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fisher_cache", default=None,
                        help="Path to cache Fisher dicts (skip recomputation)")
    args = parser.parse_args()

    t0 = time.time()

    # --- Load or compute Fisher ---
    fisher_cache = args.fisher_cache or os.path.join(
        SAVES, f"_perta_fisher_cache_{args.data_split}_n{args.n_samples}.pt"
    )

    if os.path.exists(fisher_cache):
        logger.info(f"Loading cached Fisher from {fisher_cache}")
        cached = torch.load(fisher_cache, map_location="cpu", weights_only=True)
        fisher_forget = cached["forget"]
        fisher_retain = cached["retain"]
    else:
        logger.info(f"Loading target model: {args.target}")
        model = AutoModelForCausalLM.from_pretrained(
            args.target, torch_dtype=torch.bfloat16, device_map=args.device,
            attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load forget data
        logger.info("Loading forget data for Fisher computation...")
        forget_ds = load_dataset(f"muse-bench/MUSE-{args.data_split}", "raw", split="forget")
        forget_texts = [row["text"] for row in forget_ds]

        # Load retain data
        logger.info("Loading retain data for Fisher computation...")
        retain_ds = load_dataset(f"muse-bench/MUSE-{args.data_split}", "raw", split="retain1")
        retain_texts = [row["text"] for row in retain_ds]

        # Compute Fisher diagonals
        logger.info("Computing FORGET Fisher...")
        fisher_forget = compute_fisher(
            model, tokenizer, forget_texts,
            n_samples=args.n_samples, max_length=args.max_length, device=args.device
        )

        logger.info("Computing RETAIN Fisher...")
        fisher_retain = compute_fisher(
            model, tokenizer, retain_texts,
            n_samples=args.n_samples, max_length=args.max_length, device=args.device
        )

        # Cache Fisher
        torch.save({"forget": fisher_forget, "retain": fisher_retain}, fisher_cache)
        logger.info(f"Fisher cached to {fisher_cache}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # --- Load both models' state dicts ---
    logger.info(f"Loading target state dict: {args.target}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    target_sd = {k: v.clone() for k, v in target_model.state_dict().items()}
    # Find the config directory for saving
    target_config_dir = target_model.config._name_or_path
    # Resolve to actual path for file copying
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

    # --- Task vector statistics ---
    tau_norms = []
    for name in target_sd:
        tau = (target_sd[name].float() - pretrained_sd[name].float())
        tau_norms.append(tau.norm().item())
    logger.info(f"Task vector: {len(tau_norms)} params, "
                f"mean_norm={sum(tau_norms)/len(tau_norms):.6f}, "
                f"max_norm={max(tau_norms):.6f}")

    # --- Fisher mask statistics ---
    w_stats = []
    for name in fisher_forget:
        if name in fisher_retain:
            ff = fisher_forget[name].float()
            fr = fisher_retain[name].float()
            w = ff / (ff + args.alpha * fr + 1e-8)
            w_stats.append(w.mean().item())
    logger.info(f"Fisher mask w: mean={sum(w_stats)/len(w_stats):.4f} "
                f"(0=retain-dominated, 1=forget-dominated)")

    # --- Sweep over lambda values ---
    for lam in args.lambdas:
        out_name = f"perta_l{lam:.1f}_a{args.alpha:.1f}"
        out_dir = os.path.join(SAVES, out_name)

        if os.path.exists(out_dir) and any(
            f.endswith(".safetensors") and "model-" in f for f in os.listdir(out_dir)
        ):
            logger.info(f"[SKIP] {out_name} already exists")
            continue

        logger.info(f"Applying PerTA: lambda={lam}, alpha={args.alpha}")
        result_sd = apply_perta(
            target_sd, pretrained_sd, fisher_forget, fisher_retain,
            lam=lam, alpha=args.alpha, epsilon=1e-8
        )

        # Use pretrained config dir (has tokenizer files)
        save_model(result_sd, out_dir, pretrained_config_dir)

        # Also create empty evals dir for later
        os.makedirs(os.path.join(out_dir, "evals"), exist_ok=True)

        del result_sd

    elapsed = time.time() - t0
    logger.info(f"PerTA complete. {len(args.lambdas)} variants saved. "
                f"Total time: {elapsed:.0f}s")
    logger.info(f"\nTo evaluate, run:")
    for lam in args.lambdas:
        name = f"perta_l{lam:.1f}_a{args.alpha:.1f}"
        logger.info(f"  python src/eval.py experiment=eval/muse/default.yaml "
                    f"data_split={args.data_split} task_name={name} model=Llama-2-7b-hf "
                    f"model.model_args.pretrained_model_name_or_path=saves/unlearn/{name} "
                    f"model.model_args.attn_implementation=sdpa "
                    f"paths.output_dir=saves/unlearn/{name}/evals "
                    f"retain_logs_path=saves/eval/muse_Llama-2-7b-hf_{args.data_split}_retrain/MUSE_EVAL.json")


if __name__ == "__main__":
    main()
