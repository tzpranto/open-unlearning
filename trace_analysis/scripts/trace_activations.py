#!/usr/bin/env python3
"""
Mechanistic Interpretability Trace Collection for Machine Unlearning
====================================================================

Collects three types of traces from a language model on forget vs retain data:

1. Causal Tracing (ROME-style): Which layers causally store knowledge?
   - Corrupts input embeddings, restores clean hidden states at each layer,
     measures how much of the correct prediction is recovered.

2. Gradient Differential: Which parameters are most forget-specific?
   - Computes mean |gradient| per parameter on forget vs retain losses.
   - Ratio forget/retain identifies parameters specific to forget knowledge.

3. Activation Statistics: How do layer activations differ between sets?
   - Per-layer L2 norm, mean absolute activation, variance for
     full layer outputs, MLP outputs, and attention outputs.

Standalone script -- no dependency on the open-unlearning framework.

Usage:
    python scripts/trace_activations.py
    python scripts/trace_activations.py --n_samples 50 --skip_causal
    bash scripts/run_trace.sh
"""

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ===================================================================
# Data helpers
# ===================================================================

class TextDataset(Dataset):
    """Thin wrapper around tokenized encodings."""

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def load_and_tokenize(tokenizer, dataset_name, split_name, n_samples, max_length):
    """Load a HuggingFace dataset split, tokenize, and return a Dataset."""
    logger.info(f"Loading {dataset_name}  split={split_name}  (n_samples={n_samples})")
    ds = load_dataset(dataset_name, name="raw", split=split_name)
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))

    texts = list(ds["text"])
    enc = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return TextDataset(enc["input_ids"], enc["attention_mask"])


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


# ===================================================================
# Technique 1 -- Layer-wise Causal Importance (Selective Perturbation)
# ===================================================================
# Instead of the classic ROME "corrupt-then-restore" (which saturates when
# restoring any single layer gives subsequent layers clean input), we use
# per-layer noise injection: add calibrated noise to ONE layer's output
# at a time and measure how much the prediction degrades.  This directly
# identifies which layers are most causally important for the model's
# ability to predict a given text.

@torch.no_grad()
def _clean_logprob(model, input_ids, attention_mask):
    """Clean forward pass → mean per-sample next-token log-prob."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return _mean_logprob(logits, input_ids, attention_mask)


@torch.no_grad()
def _perturbed_layer_forward(model, input_ids, attention_mask,
                             layer_idx, noise_mult):
    """Forward pass with noise injected at exactly *one* layer's output.

    Noise std is calibrated per-layer: ``noise_mult * act_std`` where
    ``act_std`` is the std of the actual activations at that layer during
    this forward pass.
    """
    def _hook(_mod, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        act_std = h.float().std().clamp(min=1e-8).item()
        noise = torch.randn_like(h) * (noise_mult * act_std)
        h_noised = h + noise
        if isinstance(output, tuple):
            return (h_noised,) + output[1:]
        return h_noised

    handle = model.model.layers[layer_idx].register_forward_hook(_hook)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    handle.remove()
    return _mean_logprob(logits, input_ids, attention_mask)


def _mean_logprob(logits, input_ids, attention_mask):
    """Mean per-sample log-prob of next-token prediction (excluding padding)."""
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    lp = F.log_softmax(shift_logits, dim=-1)
    target_lp = lp.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)
    target_lp = target_lp * shift_mask
    return target_lp.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)


def collect_causal_traces(model, dataloader, n_layers, noise_std, device):
    """Layer-wise causal importance via selective noise injection.

    For each layer *l* we add calibrated noise to only that layer's output
    (all other layers run cleanly) and measure how much the mean
    next-token log-probability drops.

    Higher degradation = more causally important layer.

    Args:
        noise_std: multiplier applied to per-layer activation std.
                   Default 3.0 means noise has std = 3× the layer's own
                   activation std, which is strong enough to perturb without
                   catastrophically destroying the signal.

    Returns:
        Dict[layer_idx, float] -- mean log-prob degradation per layer.
    """
    logger.info("=== Causal Tracing (per-layer noise injection) ===")
    logger.info(f"  Noise multiplier: {noise_std}")

    degradation = defaultdict(list)
    n_batches = len(dataloader)

    for bi, batch in enumerate(dataloader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        clean_lp = _clean_logprob(model, ids, mask)          # (B,)

        for li in range(n_layers):
            pert_lp = _perturbed_layer_forward(
                model, ids, mask, li, noise_std)
            # degradation = how much log-prob dropped (positive = more damage)
            deg = (clean_lp - pert_lp).mean().item()
            degradation[li].append(deg)

        torch.cuda.empty_cache()
        if (bi + 1) % 5 == 0 or bi == n_batches - 1:
            logger.info(f"  Causal tracing  batch {bi+1}/{n_batches}")

    return {li: float(np.mean(degradation[li])) for li in range(n_layers)}


# ===================================================================
# Technique 2 -- Gradient Differential
# ===================================================================

def collect_gradient_traces(model, dataloader, device):
    """Compute mean |gradient| per parameter over all batches.

    The model must have ``requires_grad=True`` on its parameters.
    Returns ``Dict[param_name, float]``.
    """
    logger.info("=== Gradient Traces ===")

    accum = {}
    n_batches = 0

    for bi, batch in enumerate(dataloader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = ids.clone()

        model.zero_grad()
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad.abs().mean().item()
                accum[name] = accum.get(name, 0.0) + g

        model.zero_grad()
        n_batches += 1

        if (bi + 1) % 10 == 0 or bi == len(dataloader) - 1:
            logger.info(f"  Gradient traces  batch {bi+1}/{len(dataloader)}")

    for name in accum:
        accum[name] /= n_batches
    return accum


# ===================================================================
# Technique 3 -- Activation Statistics
# ===================================================================

@torch.no_grad()
def collect_activation_traces(model, dataloader, n_layers, device):
    """Per-layer activation statistics (L2 norm, mean |act|, variance).

    Hooks on: full layer output, MLP output, self-attention output.
    Returns ``Dict[key, {l2_norm, mean_abs, var}]``.
    """
    logger.info("=== Activation Traces ===")

    accum = defaultdict(lambda: {"l2_norm": [], "mean_abs": [], "var": []})

    for bi, batch in enumerate(dataloader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        captured = {}
        hooks = []

        for li in range(n_layers):
            layer = model.model.layers[li]

            def _make(key):
                def _hook(_mod, _inp, output):
                    o = output[0] if isinstance(output, tuple) else output
                    captured[key] = o.detach()
                return _hook

            hooks.append(layer.register_forward_hook(_make(f"layer_{li}")))
            hooks.append(layer.mlp.register_forward_hook(_make(f"layer_{li}_mlp")))
            hooks.append(layer.self_attn.register_forward_hook(
                _make(f"layer_{li}_attn")))

        model(input_ids=ids, attention_mask=mask)

        for h in hooks:
            h.remove()

        mask_exp = mask.unsqueeze(-1)  # (B, S, 1)

        for key, act in captured.items():
            masked = act * mask_exp
            n_tok = mask.sum().item()
            d = act.shape[-1]

            accum[key]["l2_norm"].append(
                masked.norm(dim=-1).sum().item() / max(n_tok, 1))
            accum[key]["mean_abs"].append(
                masked.abs().sum().item() / max(n_tok * d, 1))
            accum[key]["var"].append(
                masked.var(dim=-1).sum().item() / max(n_tok, 1))

        del captured
        if (bi + 1) % 10 == 0 or bi == len(dataloader) - 1:
            logger.info(f"  Activation traces  batch {bi+1}/{len(dataloader)}")

    result = {}
    for key in accum:
        result[key] = {
            "l2_norm": float(np.mean(accum[key]["l2_norm"])),
            "mean_abs": float(np.mean(accum[key]["mean_abs"])),
            "var": float(np.mean(accum[key]["var"])),
        }
    return result


# ===================================================================
# Analysis helpers
# ===================================================================

def compute_differential_scores(forget_grads, retain_grads, eps=1e-10):
    """Per-parameter forget/retain gradient ratio."""
    scores = {}
    for name in set(forget_grads) | set(retain_grads):
        fg = forget_grads.get(name, 0.0)
        rg = retain_grads.get(name, 0.0)
        scores[name] = fg / (rg + eps)
    return scores


def compute_layer_differential(forget_act, retain_act):
    """Per-key activation L2-norm differential (forget - retain)."""
    return {k: forget_act[k]["l2_norm"] - retain_act[k]["l2_norm"]
            for k in forget_act if k in retain_act}


def _aggregate_grads_by_layer(grads, n_layers):
    """Sum parameter gradient magnitudes per transformer layer."""
    layer_sum = defaultdict(float)
    layer_cnt = defaultdict(int)
    for name, val in grads.items():
        m = re.search(r"model\.layers\.(\d+)\.", name)
        if m:
            li = int(m.group(1))
            layer_sum[li] += val
            layer_cnt[li] += 1
    return {l: layer_sum[l] / max(layer_cnt[l], 1) for l in range(n_layers)}


# ===================================================================
# Plotting
# ===================================================================

def _sorted_layer_keys(act_dict, suffix=""):
    """Return layer keys matching ``layer_<int><suffix>`` sorted by index."""
    pat = re.compile(rf"^layer_(\d+){re.escape(suffix)}$")
    matches = [(k, int(pat.match(k).group(1))) for k in act_dict if pat.match(k)]
    matches.sort(key=lambda x: x[1])
    return [k for k, _ in matches], [i for _, i in matches]


def plot_causal_traces(forget_causal, retain_causal, output_dir):
    layers = sorted(forget_causal.keys())
    f_vals = [forget_causal[l] for l in layers]
    r_vals = [retain_causal[l] for l in layers]
    diff = [f - r for f, r in zip(f_vals, r_vals)]

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    axes[0].bar(layers, f_vals, color="#e74c3c", alpha=0.85)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Causal Effect")
    axes[0].set_title("Causal Trace: Forget Set")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(layers, r_vals, color="#2ecc71", alpha=0.85)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Causal Effect")
    axes[1].set_title("Causal Trace: Retain Set")
    axes[1].grid(axis="y", alpha=0.3)

    colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in diff]
    axes[2].bar(layers, diff, color=colors, alpha=0.85)
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Differential (Forget - Retain)")
    axes[2].set_title("Causal Differential")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "causal_traces.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_activation_traces(forget_act, retain_act, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, suffix, title in [
        (axes[0, 0], "", "Full Layer Output"),
        (axes[0, 1], "_mlp", "MLP Output"),
        (axes[1, 0], "_attn", "Attention Output"),
    ]:
        keys, idxs = _sorted_layer_keys(forget_act, suffix)
        fn = [forget_act[k]["l2_norm"] for k in keys]
        rn = [retain_act[k]["l2_norm"] for k in keys]
        ax.plot(idxs, fn, "r-o", label="Forget", markersize=4)
        ax.plot(idxs, rn, "g-s", label="Retain", markersize=4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean L2 Norm")
        ax.set_title(f"Activation Norms: {title}")
        ax.legend()
        ax.grid(alpha=0.3)

    # Differential subplot
    ax = axes[1, 1]
    for suffix, label in [("", "Layer"), ("_mlp", "MLP"), ("_attn", "Attn")]:
        keys, idxs = _sorted_layer_keys(forget_act, suffix)
        diff = [forget_act[k]["l2_norm"] - retain_act[k]["l2_norm"] for k in keys]
        ax.bar(np.array(idxs) + {"": -0.25, "_mlp": 0.0, "_attn": 0.25}[suffix],
               diff, width=0.25, label=label, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Differential (Forget - Retain)")
    ax.set_title("Activation Differential by Component")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "activation_traces.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_gradient_differential(forget_grads, retain_grads, diff_scores,
                               n_layers, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    fg_layer = _aggregate_grads_by_layer(forget_grads, n_layers)
    rg_layer = _aggregate_grads_by_layer(retain_grads, n_layers)
    layers = sorted(fg_layer.keys())
    fg_v = [fg_layer[l] for l in layers]
    rg_v = [rg_layer[l] for l in layers]

    w = 0.35
    axes[0].bar(np.array(layers) - w / 2, fg_v, w,
                label="Forget", color="#e74c3c", alpha=0.85)
    axes[0].bar(np.array(layers) + w / 2, rg_v, w,
                label="Retain", color="#2ecc71", alpha=0.85)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean |Gradient|")
    axes[0].set_title("Per-Layer Gradient Magnitude")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    ratios = [fg_layer[l] / (rg_layer[l] + 1e-10) for l in layers]
    colors = ["#e74c3c" if r > 1.0 else "#2ecc71" for r in ratios]
    axes[1].bar(layers, ratios, color=colors, alpha=0.85)
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Ratio (Forget / Retain)")
    axes[1].set_title("Per-Layer Forget-Specificity (Gradient)")
    axes[1].grid(axis="y", alpha=0.3)

    top_n = 30
    sorted_p = sorted(diff_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [p.replace("model.layers.", "L").replace(".weight", "")[:40]
             for p, _ in sorted_p]
    vals = [v for _, v in sorted_p]
    axes[2].barh(range(len(names)), vals, color="#e74c3c", alpha=0.85)
    axes[2].set_yticks(range(len(names)))
    axes[2].set_yticklabels(names, fontsize=7)
    axes[2].set_xlabel("Differential Score")
    axes[2].set_title(f"Top {top_n} Forget-Specific Parameters")
    axes[2].invert_yaxis()
    axes[2].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "gradient_differential.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_summary(causal_f, causal_r, forget_act, retain_act,
                 forget_grads, retain_grads, n_layers, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Mechanistic Interpretability Summary: Forget vs Retain",
                 fontsize=14, fontweight="bold")

    layers = list(range(n_layers))
    fc = [causal_f.get(l, 0) for l in layers]
    rc = [causal_r.get(l, 0) for l in layers]

    axes[0, 0].plot(layers, fc, "r-o", label="Forget", markersize=4)
    axes[0, 0].plot(layers, rc, "g-s", label="Retain", markersize=4)
    axes[0, 0].fill_between(layers, fc, rc, alpha=0.10, color="purple")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Causal Effect")
    axes[0, 0].set_title("Causal Tracing")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    diff_c = [f - r for f, r in zip(fc, rc)]
    cols = ["#e74c3c" if d > 0 else "#2ecc71" for d in diff_c]
    axes[0, 1].bar(layers, diff_c, color=cols, alpha=0.85)
    axes[0, 1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Causal Differential")
    axes[0, 1].set_title("Forget-Critical (red) vs Retain-Critical (green)")
    axes[0, 1].grid(axis="y", alpha=0.3)

    fg_l = _aggregate_grads_by_layer(forget_grads, n_layers)
    rg_l = _aggregate_grads_by_layer(retain_grads, n_layers)
    ratios = [fg_l.get(l, 0) / (rg_l.get(l, 0) + 1e-10) for l in layers]
    gcols = ["#e74c3c" if r > 1 else "#2ecc71" for r in ratios]
    axes[1, 0].bar(layers, ratios, color=gcols, alpha=0.85)
    axes[1, 0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Gradient Ratio")
    axes[1, 0].set_title("Gradient Forget-Specificity per Layer")
    axes[1, 0].grid(axis="y", alpha=0.3)

    keys, idxs = _sorted_layer_keys(forget_act, "")
    fn = [forget_act[k]["l2_norm"] for k in keys]
    rn = [retain_act[k]["l2_norm"] for k in keys]
    axes[1, 1].plot(idxs, fn, "r-o", label="Forget", markersize=4)
    axes[1, 1].plot(idxs, rn, "g-s", label="Retain", markersize=4)
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Mean L2 Norm")
    axes[1, 1].set_title("Activation Norms per Layer")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


# ===================================================================
# Console summary
# ===================================================================

def print_summary(causal_f, causal_r, forget_grads, retain_grads,
                  diff_scores, n_layers):
    print("\n" + "=" * 80)
    print("MECHANISTIC INTERPRETABILITY TRACE SUMMARY")
    print("=" * 80)

    print("\n--- Causal Tracing: Top 10 Layers by Forget Causal Effect ---")
    sf = sorted(causal_f.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Layer':>6} {'Forget':>12} {'Retain':>12} {'Diff':>12}")
    print("-" * 46)
    for layer, score in sf[:10]:
        r = causal_r.get(layer, 0.0)
        print(f"{layer:>6d} {score:>12.4f} {r:>12.4f} {score - r:>12.4f}")

    print("\n--- Gradient: Top 10 Layers by Forget-Specificity ---")
    fg_l = _aggregate_grads_by_layer(forget_grads, n_layers)
    rg_l = _aggregate_grads_by_layer(retain_grads, n_layers)
    lr = [(l, fg_l[l] / (rg_l[l] + 1e-10)) for l in range(n_layers)]
    lr.sort(key=lambda x: x[1], reverse=True)
    print(f"{'Layer':>6} {'Forget |g|':>14} {'Retain |g|':>14} {'Ratio':>10}")
    print("-" * 48)
    for layer, ratio in lr[:10]:
        print(f"{layer:>6d} {fg_l[layer]:>14.6f} {rg_l[layer]:>14.6f} {ratio:>10.2f}")

    print("\n--- Top 20 Most Forget-Specific Parameters ---")
    sp = sorted(diff_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Parameter':>55} {'Score':>10}")
    print("-" * 67)
    for name, score in sp[:20]:
        short = name[-55:] if len(name) > 55 else name
        print(f"{short:>55} {score:>10.4f}")

    # Layers where forget > retain (causal)
    forget_layers = [l for l, s in causal_f.items()
                     if s > causal_r.get(l, 0)]
    retain_layers = [l for l, s in causal_r.items()
                     if s > causal_f.get(l, 0)]
    print(f"\nForget-dominant layers (causal): {sorted(forget_layers)}")
    print(f"Retain-dominant layers (causal): {sorted(retain_layers)}")
    print("=" * 80 + "\n")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect mechanistic interpretability traces for unlearning")

    parser.add_argument("--model_name", type=str,
                        default="muse-bench/MUSE-News_target")
    parser.add_argument("--tokenizer", type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="muse-bench/MUSE-News")
    parser.add_argument("--forget_split", type=str, default="forget")
    parser.add_argument("--retain_split", type=str, default="retain1")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str,
                        default="trace_analysis/figures/traces/muse_news_llama2_7b")
    parser.add_argument("--causal_noise_std", type=float, default=3.0,
                        help="Noise multiplier (times embedding std)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_causal", action="store_true",
                        help="Skip causal tracing (saves time)")
    parser.add_argument("--skip_gradients", action="store_true",
                        help="Skip gradient differential")
    parser.add_argument("--skip_activations", action="store_true",
                        help="Skip activation statistics")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Model:  {args.model_name}")
    logger.info(f"Data:   {args.dataset}")
    logger.info(f"Samples per split: {args.n_samples}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info("Loading model ...")
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    logger.info(f"  Attention: {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = len(model.model.layers)
    logger.info(f"  Loaded: {n_layers} layers")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    forget_ds = load_and_tokenize(
        tokenizer, args.dataset, args.forget_split, args.n_samples,
        args.max_length)
    retain_ds = load_and_tokenize(
        tokenizer, args.dataset, args.retain_split, args.n_samples,
        args.max_length)

    forget_loader = DataLoader(forget_ds, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_fn)
    retain_loader = DataLoader(retain_ds, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_fn)

    logger.info(f"  Forget: {len(forget_ds)} samples, "
                f"Retain: {len(retain_ds)} samples")

    results = {
        "metadata": {
            "model": args.model_name,
            "tokenizer": args.tokenizer,
            "dataset": args.dataset,
            "forget_split": args.forget_split,
            "retain_split": args.retain_split,
            "n_forget_samples": len(forget_ds),
            "n_retain_samples": len(retain_ds),
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "n_layers": n_layers,
            "causal_noise_std": args.causal_noise_std,
            "timestamp": datetime.now().isoformat(),
        },
    }
    t_total = time.time()

    # ------------------------------------------------------------------
    # Technique 1: Causal Tracing
    # ------------------------------------------------------------------
    if not args.skip_causal:
        t0 = time.time()
        causal_forget = collect_causal_traces(
            model, forget_loader, n_layers, args.causal_noise_std, args.device)
        causal_retain = collect_causal_traces(
            model, retain_loader, n_layers, args.causal_noise_std, args.device)
        results["causal_traces"] = {
            "forget": causal_forget,
            "retain": causal_retain,
        }
        logger.info(f"  Causal tracing: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Technique 2: Gradient Differential
    # ------------------------------------------------------------------
    if not args.skip_gradients:
        t0 = time.time()
        # Enable gradients (model stays in eval mode -- no dropout/BN effect)
        for p in model.parameters():
            p.requires_grad_(True)

        forget_grads = collect_gradient_traces(model, forget_loader, args.device)
        retain_grads = collect_gradient_traces(model, retain_loader, args.device)
        diff_scores = compute_differential_scores(forget_grads, retain_grads)

        results["gradient_traces"] = {
            "forget": forget_grads,
            "retain": retain_grads,
        }
        results["differential_scores"] = diff_scores

        # Restore
        for p in model.parameters():
            p.requires_grad_(False)
        model.zero_grad()
        torch.cuda.empty_cache()
        logger.info(f"  Gradient traces: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Technique 3: Activation Statistics
    # ------------------------------------------------------------------
    if not args.skip_activations:
        t0 = time.time()
        forget_act = collect_activation_traces(
            model, forget_loader, n_layers, args.device)
        retain_act = collect_activation_traces(
            model, retain_loader, n_layers, args.device)
        layer_diff = compute_layer_differential(forget_act, retain_act)

        results["activation_traces"] = {
            "forget": forget_act,
            "retain": retain_act,
        }
        results["layer_differential"] = layer_diff
        logger.info(f"  Activation traces: {time.time() - t0:.1f}s")

    total_time = time.time() - t_total
    results["metadata"]["total_time_seconds"] = round(total_time, 1)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    pt_path = os.path.join(args.output_dir, "trace_results.pt")
    torch.save(results, pt_path)
    logger.info(f"Saved {pt_path}")

    # JSON summary (no large tensors, just scalars)
    json_out = {"metadata": results["metadata"]}
    if "causal_traces" in results:
        # Convert int keys to strings for JSON
        json_out["causal_traces"] = {
            "forget": {str(k): v for k, v in results["causal_traces"]["forget"].items()},
            "retain": {str(k): v for k, v in results["causal_traces"]["retain"].items()},
        }
    if "differential_scores" in results:
        top50 = sorted(results["differential_scores"].items(),
                        key=lambda x: x[1], reverse=True)[:50]
        json_out["top_50_differential_params"] = dict(top50)
    if "layer_differential" in results:
        json_out["layer_differential"] = results["layer_differential"]
    if "activation_traces" in results:
        json_out["activation_traces"] = results["activation_traces"]

    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    logger.info(f"Saved {json_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if "causal_traces" in results:
        plot_causal_traces(
            results["causal_traces"]["forget"],
            results["causal_traces"]["retain"],
            args.output_dir)

    if "activation_traces" in results:
        plot_activation_traces(
            results["activation_traces"]["forget"],
            results["activation_traces"]["retain"],
            args.output_dir)

    if "gradient_traces" in results:
        plot_gradient_differential(
            results["gradient_traces"]["forget"],
            results["gradient_traces"]["retain"],
            results.get("differential_scores", {}),
            n_layers, args.output_dir)

    if all(k in results for k in
           ["causal_traces", "activation_traces", "gradient_traces"]):
        plot_summary(
            results["causal_traces"]["forget"],
            results["causal_traces"]["retain"],
            results["activation_traces"]["forget"],
            results["activation_traces"]["retain"],
            results["gradient_traces"]["forget"],
            results["gradient_traces"]["retain"],
            n_layers, args.output_dir)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    if "causal_traces" in results and "gradient_traces" in results:
        print_summary(
            results["causal_traces"]["forget"],
            results["causal_traces"]["retain"],
            results["gradient_traces"]["forget"],
            results["gradient_traces"]["retain"],
            results.get("differential_scores", {}),
            n_layers)

    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
