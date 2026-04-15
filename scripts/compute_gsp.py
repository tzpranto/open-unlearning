#!/usr/bin/env python3
"""
GSP: Gradient Subspace Partitioning for Data-Disentangled Unlearning
====================================================================

Computes per-sample gradient signatures for forget and retain data,
measures pairwise gradient interference, and partitions samples into
"pure" (low interference) and "contested" (high interference) subsets.

Stage 1: Gradient signatures — per-sample gradients on down_proj.weight
          in last 4 transformer layers, L2-normalized.
Stage 2: Interference matrix — pairwise cosine similarity.
Stage 3: Partition and analysis.

Output:
  saves/unlearn/gsp_signatures_News.pt  — G_f, G_r tensors
  saves/unlearn/gsp_interference_News.pt — I, E_f, E_r, partitions
  logs/gsp_analysis.log — human-readable report

Usage:
  python scripts/compute_gsp.py [--data_split News] [--top_k 10]
"""

import argparse
import logging
import os
import sys
import time

import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_model(data_split):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = f"muse-bench/MUSE-{data_split}_target"
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # We'll move to GPU per-sample
        attn_implementation="sdpa",
    )
    # Use base Llama-2 tokenizer (MUSE target tokenizer can have loading issues)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def load_dataset_split(data_split, split_name, tokenizer, max_length=1024):
    from datasets import load_dataset

    logger.info(f"Loading {split_name} from muse-bench/MUSE-{data_split}")
    ds = load_dataset(f"muse-bench/MUSE-{data_split}", "raw", split=split_name)

    # Tokenize
    samples = []
    for item in ds:
        text = item.get("text", item.get("input", ""))
        if not text:
            continue
        enc = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        samples.append({
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "text_preview": text[:150],
        })

    logger.info(f"  {split_name}: {len(samples)} samples")
    return samples


def compute_signatures(model, samples, device, target_layers, sig_dim_per_layer):
    """Compute per-sample gradient signatures.

    For each sample: fwd+bwd on CE loss, extract gradient of down_proj.weight
    in target layers, take L2 norm over output dim -> per-neuron gradient magnitude.
    Concatenate and L2-normalize.
    """
    n = len(samples)
    total_dim = len(target_layers) * sig_dim_per_layer
    signatures = torch.zeros(n, total_dim, dtype=torch.float32)

    model.to(device)

    # Identify target parameters
    target_params = {}
    for layer_idx in target_layers:
        name = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        for pname, param in model.named_parameters():
            if pname == name:
                target_params[layer_idx] = (pname, param)
                break

    if len(target_params) != len(target_layers):
        found = list(target_params.keys())
        raise ValueError(f"Could not find all target layers. Found: {found}, wanted: {target_layers}")

    # CRITICAL: from_pretrained sets requires_grad=False on all params.
    # We need gradients on target params to compute signatures.
    for layer_idx in target_layers:
        _, param = target_params[layer_idx]
        param.requires_grad_(True)

    logger.info(f"Computing signatures for {n} samples across layers {target_layers}...")
    t0 = time.time()

    for i, sample in enumerate(samples):
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        labels = input_ids.clone()

        # Zero all gradients
        model.zero_grad()

        # Forward + backward
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        loss.backward()

        # Extract gradient signatures
        sig_parts = []
        for layer_idx in target_layers:
            _, param = target_params[layer_idx]
            if param.grad is not None:
                # down_proj.weight shape: (hidden_size, intermediate_size)
                # Take L2 norm over hidden_size dim -> (intermediate_size,)
                grad = param.grad.float()
                neuron_norms = grad.norm(dim=0)  # (intermediate_size,)
                sig_parts.append(neuron_norms.cpu())
            else:
                sig_parts.append(torch.zeros(sig_dim_per_layer))

        sig = torch.cat(sig_parts)
        # L2 normalize
        sig_norm = sig.norm().clamp(min=1e-12)
        signatures[i] = sig / sig_norm

        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            logger.info(f"  [{i+1}/{n}] {rate:.1f} samples/s, ETA {eta:.0f}s")

        # Clear GPU cache periodically
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    # Safety check: warn if signatures are all-zero (likely requires_grad issue)
    zero_count = (signatures.norm(dim=1) == 0).sum().item()
    if zero_count > 0:
        logger.warning(f"{zero_count}/{n} signatures are all-zero! Check requires_grad on target params.")

    model.cpu()
    torch.cuda.empty_cache()

    return signatures


def compute_interference(G_f, G_r, top_k=10):
    """Compute interference matrix and per-sample entanglement scores.

    I[i,j] = |cosine(g_f_i, g_r_j)|
    E_f[i] = mean of top-k retain cosine similarities for forget sample i
    E_r[j] = mean of top-k forget cosine similarities for retain sample j
    """
    logger.info(f"Computing interference matrix ({G_f.shape[0]} x {G_r.shape[0]})...")

    # Cosine similarity = dot product of L2-normalized vectors
    # G_f and G_r are already L2-normalized
    I = torch.abs(G_f @ G_r.T)  # (n_forget, n_retain)

    # Per-sample entanglement: mean of top-k (more robust than max)
    topk_f = I.topk(min(top_k, I.shape[1]), dim=1).values
    E_f = topk_f.mean(dim=1)  # (n_forget,)

    topk_r = I.topk(min(top_k, I.shape[0]), dim=0).values
    E_r = topk_r.mean(dim=0)  # (n_retain,)

    logger.info(f"  E_f: mean={E_f.mean():.4f}, std={E_f.std():.4f}, "
                f"min={E_f.min():.4f}, max={E_f.max():.4f}")
    logger.info(f"  E_r: mean={E_r.mean():.4f}, std={E_r.std():.4f}, "
                f"min={E_r.min():.4f}, max={E_r.max():.4f}")

    return I, E_f, E_r


def partition_and_report(E_f, E_r, forget_samples, retain_samples, percentiles=[50, 70, 90]):
    """Partition samples and generate detailed report."""
    partitions = {}

    logger.info("\n" + "=" * 60)
    logger.info("PARTITION ANALYSIS")
    logger.info("=" * 60)

    for pct in percentiles:
        tau_f = torch.quantile(E_f.float(), pct / 100.0).item()
        tau_r = torch.quantile(E_r.float(), pct / 100.0).item()

        pure_f_mask = E_f < tau_f
        contested_f_mask = E_f >= tau_f
        pure_r_mask = E_r < tau_r
        contested_r_mask = E_r >= tau_r

        partitions[pct] = {
            "tau_f": tau_f,
            "tau_r": tau_r,
            "pure_f_mask": pure_f_mask,
            "contested_f_mask": contested_f_mask,
            "pure_r_mask": pure_r_mask,
            "contested_r_mask": contested_r_mask,
        }

        n_pure_f = pure_f_mask.sum().item()
        n_contested_f = contested_f_mask.sum().item()
        n_pure_r = pure_r_mask.sum().item()
        n_contested_r = contested_r_mask.sum().item()

        logger.info(f"\n--- Percentile {pct} ---")
        logger.info(f"  τ_f={tau_f:.4f}, τ_r={tau_r:.4f}")
        logger.info(f"  Forget: {n_pure_f} pure, {n_contested_f} contested")
        logger.info(f"  Retain: {n_pure_r} pure, {n_contested_r} contested")
        logger.info(f"  Pure forget mean E: {E_f[pure_f_mask].mean():.4f}")
        logger.info(f"  Contested forget mean E: {E_f[contested_f_mask].mean():.4f}")

    # Distribution analysis
    logger.info("\n" + "=" * 60)
    logger.info("DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)

    # Histogram bins
    bins = torch.linspace(E_f.min().item(), E_f.max().item(), 11)
    hist_f = torch.histogram(E_f.float(), bins=bins)
    logger.info("\nForget entanglement histogram:")
    for i in range(len(hist_f.hist)):
        bar = "█" * int(hist_f.hist[i].item() / max(1, E_f.shape[0]) * 80)
        logger.info(f"  [{bins[i]:.3f}-{bins[i+1]:.3f}] {int(hist_f.hist[i].item()):4d} {bar}")

    bins_r = torch.linspace(E_r.min().item(), E_r.max().item(), 11)
    hist_r = torch.histogram(E_r.float(), bins=bins_r)
    logger.info("\nRetain entanglement histogram:")
    for i in range(len(hist_r.hist)):
        bar = "█" * int(hist_r.hist[i].item() / max(1, E_r.shape[0]) * 80)
        logger.info(f"  [{bins_r[i]:.3f}-{bins_r[i+1]:.3f}] {int(hist_r.hist[i].item()):4d} {bar}")

    # Top-10 most entangled samples
    logger.info("\n" + "=" * 60)
    logger.info("TOP-10 MOST ENTANGLED FORGET SAMPLES")
    logger.info("=" * 60)
    top_f_idx = E_f.argsort(descending=True)[:10]
    for rank, idx in enumerate(top_f_idx):
        preview = forget_samples[idx.item()]["text_preview"]
        logger.info(f"  #{rank+1} (E={E_f[idx]:.4f}, idx={idx.item()}): {preview}")

    logger.info("\n" + "=" * 60)
    logger.info("TOP-10 MOST ENTANGLED RETAIN SAMPLES")
    logger.info("=" * 60)
    top_r_idx = E_r.argsort(descending=True)[:10]
    for rank, idx in enumerate(top_r_idx):
        preview = retain_samples[idx.item()]["text_preview"]
        logger.info(f"  #{rank+1} (E={E_r[idx]:.4f}, idx={idx.item()}): {preview}")

    return partitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", default="News")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--skip_signatures", action="store_true",
                        help="Skip signature computation, load from cache")
    args = parser.parse_args()

    global logger
    logger = setup_logging(f"logs/gsp_analysis_{args.data_split}.log")

    logger.info("=" * 60)
    logger.info(f"GSP: Gradient Subspace Partitioning — MUSE {args.data_split}")
    logger.info("=" * 60)

    saves_dir = "saves/unlearn"
    os.makedirs(saves_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    sig_path = f"{saves_dir}/gsp_signatures_{args.data_split}.pt"
    int_path = f"{saves_dir}/gsp_interference_{args.data_split}.pt"

    # Llama-2-7b: 32 layers, intermediate_size=11008
    target_layers = [28, 29, 30, 31]
    sig_dim_per_layer = 11008

    if args.skip_signatures and os.path.exists(sig_path):
        logger.info(f"Loading cached signatures from {sig_path}")
        cached = torch.load(sig_path, weights_only=True)
        G_f = cached["G_f"]
        G_r = cached["G_r"]
        forget_samples = cached.get("forget_previews", [])
        retain_samples = cached.get("retain_previews", [])
    else:
        # Stage 1: Compute gradient signatures
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: Gradient Signatures")
        logger.info("=" * 60)

        model, tokenizer = load_model(args.data_split)
        forget_samples = load_dataset_split(args.data_split, "forget", tokenizer, args.max_length)
        retain_samples = load_dataset_split(args.data_split, "retain1", tokenizer, args.max_length)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("\nComputing FORGET signatures...")
        G_f = compute_signatures(model, forget_samples, device, target_layers, sig_dim_per_layer)

        logger.info("\nComputing RETAIN signatures...")
        G_r = compute_signatures(model, retain_samples, device, target_layers, sig_dim_per_layer)

        # Save signatures
        torch.save({
            "G_f": G_f, "G_r": G_r,
            "target_layers": target_layers,
            "forget_previews": [{"text_preview": s["text_preview"]} for s in forget_samples],
            "retain_previews": [{"text_preview": s["text_preview"]} for s in retain_samples],
        }, sig_path)
        logger.info(f"Signatures saved to {sig_path}")

        del model
        torch.cuda.empty_cache()

    # Stage 2: Interference
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Interference Matrix")
    logger.info("=" * 60)

    I, E_f, E_r = compute_interference(G_f, G_r, top_k=args.top_k)

    # Stage 3: Partition and report
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: Partition and Analysis")
    logger.info("=" * 60)

    # Use stored previews for reporting
    if isinstance(forget_samples, list) and len(forget_samples) > 0 and isinstance(forget_samples[0], dict):
        pass  # already have text_preview
    else:
        forget_samples = [{"text_preview": f"sample_{i}"} for i in range(G_f.shape[0])]
        retain_samples = [{"text_preview": f"sample_{i}"} for i in range(G_r.shape[0])]

    partitions = partition_and_report(E_f, E_r, forget_samples, retain_samples)

    # Save interference data
    save_data = {
        "I": I, "E_f": E_f, "E_r": E_r,
        "partitions": {
            pct: {k: v for k, v in p.items() if isinstance(v, (float, torch.Tensor))}
            for pct, p in partitions.items()
        },
        "top_k": args.top_k,
    }
    torch.save(save_data, int_path)
    logger.info(f"\nInterference data saved to {int_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Forget samples: {G_f.shape[0]}")
    logger.info(f"  Retain samples: {G_r.shape[0]}")
    logger.info(f"  Signature dim: {G_f.shape[1]}")
    logger.info(f"  E_f mean={E_f.mean():.4f}, std={E_f.std():.4f}")
    logger.info(f"  E_r mean={E_r.mean():.4f}, std={E_r.std():.4f}")

    # Key question: is it bimodal or uniform?
    ef_median = E_f.median().item()
    ef_q25 = torch.quantile(E_f.float(), 0.25).item()
    ef_q75 = torch.quantile(E_f.float(), 0.75).item()
    iqr = ef_q75 - ef_q25
    logger.info(f"  E_f quartiles: Q25={ef_q25:.4f}, median={ef_median:.4f}, Q75={ef_q75:.4f}, IQR={iqr:.4f}")
    if iqr > 0.3 * ef_median:
        logger.info("  -> HIGH SPREAD: interference is structured, GSP can help!")
    else:
        logger.info("  -> LOW SPREAD: interference is uniform, GSP partition may not help much")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
