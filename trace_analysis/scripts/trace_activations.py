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
    # Preset-based (recommended):
    python trace_activations.py --preset muse-news
    python trace_activations.py --preset muse-books
    python trace_activations.py --preset wmdp-bio
    python trace_activations.py --preset wmdp-cyber

    # Fully custom:
    python trace_activations.py \\
        --model_name muse-bench/MUSE-News_target \\
        --dataset muse-bench/MUSE-News \\
        --dataset_config raw \\
        --forget_split forget --retain_split retain1 \\
        --n_samples 3554 --text_field text \\
        --output_dir trace_analysis/figures/traces/my_run

    # Skip slow steps:
    python trace_activations.py --preset muse-news --skip_causal
    python trace_activations.py --preset muse-news --n_samples 100

    bash trace_analysis/scripts/run_trace.sh --preset muse-news
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
# Presets — common dataset/model combinations
# ===================================================================

PRESETS = {
    "muse-news": dict(
        model_name="muse-bench/MUSE-News_target",
        tokenizer="meta-llama/Llama-2-7b-hf",
        dataset="muse-bench/MUSE-News",
        dataset_config="raw",
        forget_split="forget",
        retain_split="retain1",
        text_field="text",
        n_samples=None,   # use all 889 forget samples
        output_dir="trace_analysis/figures/traces/muse_news",
    ),
    "muse-books": dict(
        model_name="muse-bench/MUSE-Books_target",
        tokenizer="meta-llama/Llama-2-7b-hf",
        dataset="muse-bench/MUSE-Books",
        dataset_config="raw",
        forget_split="forget",
        retain_split="retain1",
        text_field="text",
        n_samples=None,   # use all available
        output_dir="trace_analysis/figures/traces/muse_books",
    ),
    "wmdp-bio": dict(
        model_name="HuggingFaceH4/zephyr-7b-beta",
        tokenizer="HuggingFaceH4/zephyr-7b-beta",
        dataset="cais/wmdp",
        dataset_config="wmdp-bio",
        forget_split="test",
        retain_split=None,   # WMDP has no retain split; uses wmdp-retain below
        text_field="question",
        n_samples=None,
        output_dir="trace_analysis/figures/traces/wmdp_bio",
    ),
    "wmdp-cyber": dict(
        model_name="HuggingFaceH4/zephyr-7b-beta",
        tokenizer="HuggingFaceH4/zephyr-7b-beta",
        dataset="cais/wmdp",
        dataset_config="wmdp-cyber",
        forget_split="test",
        retain_split=None,
        text_field="question",
        n_samples=None,
        output_dir="trace_analysis/figures/traces/wmdp_cyber",
    ),
}

# WMDP retain dataset (separate from forget)
WMDP_RETAIN_DATASET = "cais/wmdp"
WMDP_RETAIN_CONFIG = "wmdp-retain"
WMDP_RETAIN_SPLIT = "test"
WMDP_RETAIN_TEXT_FIELD = "text"


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


def load_and_tokenize(tokenizer, dataset_name, split_name, n_samples, max_length,
                      dataset_config=None, text_field="text"):
    """Load a HuggingFace dataset split, tokenize, and return a Dataset."""
    cfg_str = f"[{dataset_config}]" if dataset_config else ""
    logger.info(f"Loading {dataset_name}{cfg_str} split={split_name} (n_samples={n_samples or 'all'})")

    load_kwargs = dict(split=split_name)
    if dataset_config:
        load_kwargs["name"] = dataset_config

    ds = load_dataset(dataset_name, **load_kwargs)

    if n_samples is not None and n_samples < len(ds):
        ds = ds.select(range(n_samples))

    if text_field not in ds.features:
        available = list(ds.features.keys())
        raise ValueError(
            f"text_field='{text_field}' not found in dataset. Available: {available}"
        )

    texts = list(ds[text_field])
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


def get_model_layers(model):
    """Get the transformer layer list, supporting common model families."""
    # LlamaModel, MistralModel, GemmaModel, Qwen2Model, etc.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2, GPT-Neo
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # GPT-NeoX, Pythia
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    # Falcon
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return model.transformer.blocks
    raise ValueError(
        "Cannot auto-detect transformer layers. Model architecture not supported. "
        "Supported: Llama/Mistral/Gemma/Qwen2 (model.model.layers), "
        "GPT-2 (transformer.h), GPT-NeoX (gpt_neox.layers), Falcon (transformer.blocks)"
    )


# ===================================================================
# Technique 1 -- Layer-wise Causal Importance (Selective Perturbation)
# ===================================================================

@torch.no_grad()
def _clean_logprob(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return _mean_logprob(logits, input_ids, attention_mask)


@torch.no_grad()
def _perturbed_layer_forward(model, input_ids, attention_mask, layer, noise_mult):
    def _hook(_mod, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        act_std = h.float().std().clamp(min=1e-8).item()
        noise = torch.randn_like(h) * (noise_mult * act_std)
        h_noised = h + noise
        if isinstance(output, tuple):
            return (h_noised,) + output[1:]
        return h_noised

    handle = layer.register_forward_hook(_hook)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    handle.remove()
    return _mean_logprob(logits, input_ids, attention_mask)


def _mean_logprob(logits, input_ids, attention_mask):
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]
    lp = F.log_softmax(shift_logits, dim=-1)
    target_lp = lp.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)
    target_lp = target_lp * shift_mask
    return target_lp.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)


def collect_causal_traces(model, layers, dataloader, noise_std, device):
    """Layer-wise causal importance via selective noise injection."""
    logger.info("=== Causal Tracing (per-layer noise injection) ===")
    n_layers = len(layers)
    degradation = defaultdict(list)

    for bi, batch in enumerate(dataloader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        clean_lp = _clean_logprob(model, ids, mask)

        for li, layer in enumerate(layers):
            pert_lp = _perturbed_layer_forward(model, ids, mask, layer, noise_std)
            degradation[li].append((clean_lp - pert_lp).mean().item())

        torch.cuda.empty_cache()
        if (bi + 1) % 5 == 0 or bi == len(dataloader) - 1:
            logger.info(f"  Causal tracing batch {bi+1}/{len(dataloader)}")

    return {li: float(np.mean(degradation[li])) for li in range(n_layers)}


# ===================================================================
# Technique 2 -- Gradient Differential
# ===================================================================

def collect_gradient_traces(model, dataloader, device):
    """Compute mean |gradient| per output-neuron (row) per parameter over all batches.

    For 2-D weight matrices: accumulates abs(grad).mean(dim=1) → shape (out_features,)
    For 1-D params (bias, layernorm): accumulates abs(grad) → shape (out_features,)
    This per-row format is required by SIBL's neuron-level mask construction.
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
                g = param.grad.abs()
                # Reduce to per-row (output neuron) vector
                if g.dim() >= 2:
                    row_mean = g.mean(dim=tuple(range(1, g.dim())))  # (out_features,)
                else:
                    row_mean = g  # already 1-D
                row_mean = row_mean.detach().cpu()
                if name in accum:
                    accum[name] += row_mean
                else:
                    accum[name] = row_mean.clone()

        model.zero_grad()
        n_batches += 1

        if (bi + 1) % 10 == 0 or bi == len(dataloader) - 1:
            logger.info(f"  Gradient traces batch {bi+1}/{len(dataloader)}")

    for name in accum:
        accum[name] = accum[name] / n_batches
    return accum


# ===================================================================
# Technique 3 -- Activation Statistics
# ===================================================================

@torch.no_grad()
def collect_activation_traces(model, layers, dataloader, device):
    """Per-layer activation statistics (L2 norm, mean |act|, variance)."""
    logger.info("=== Activation Traces ===")
    accum = defaultdict(lambda: {"l2_norm": [], "mean_abs": [], "var": []})

    for bi, batch in enumerate(dataloader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        captured = {}
        hooks = []

        for li, layer in enumerate(layers):
            def _make(key):
                def _hook(_mod, _inp, output):
                    o = output[0] if isinstance(output, tuple) else output
                    captured[key] = o.detach()
                return _hook

            hooks.append(layer.register_forward_hook(_make(f"layer_{li}")))
            # MLP sub-module (try common attribute names)
            mlp = getattr(layer, "mlp", getattr(layer, "feed_forward", None))
            if mlp is not None:
                hooks.append(mlp.register_forward_hook(_make(f"layer_{li}_mlp")))
            # Attention sub-module
            attn = getattr(layer, "self_attn",
                   getattr(layer, "attention",
                   getattr(layer, "self_attention", None)))
            if attn is not None:
                hooks.append(attn.register_forward_hook(_make(f"layer_{li}_attn")))

        model(input_ids=ids, attention_mask=mask)
        for h in hooks:
            h.remove()

        mask_exp = mask.unsqueeze(-1)
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
            logger.info(f"  Activation traces batch {bi+1}/{len(dataloader)}")

    return {k: {stat: float(np.mean(vals)) for stat, vals in v.items()}
            for k, v in accum.items()}


# ===================================================================
# Analysis helpers
# ===================================================================

def compute_differential_scores(forget_grads, retain_grads, eps=1e-10):
    scores = {}
    for name in set(forget_grads) | set(retain_grads):
        fg = forget_grads.get(name, 0.0)
        rg = retain_grads.get(name, 0.0)
        scores[name] = fg / (rg + eps)
    return scores


def compute_layer_differential(forget_act, retain_act):
    return {k: forget_act[k]["l2_norm"] - retain_act[k]["l2_norm"]
            for k in forget_act if k in retain_act}


def _aggregate_grads_by_layer(grads, n_layers):
    layer_sum = defaultdict(float)
    layer_cnt = defaultdict(int)
    for name, val in grads.items():
        m = re.search(r"(?:layers|h|blocks)\.(\d+)\.", name)
        if m:
            li = int(m.group(1))
            scalar = float(val.mean()) if isinstance(val, torch.Tensor) else float(val)
            layer_sum[li] += scalar
            layer_cnt[li] += 1
    return {l: layer_sum[l] / max(layer_cnt[l], 1) for l in range(n_layers)}


# ===================================================================
# Plotting
# ===================================================================

def _sorted_layer_keys(act_dict, suffix=""):
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
    axes[0].set(xlabel="Layer", ylabel="Causal Effect", title="Causal Trace: Forget Set")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(layers, r_vals, color="#2ecc71", alpha=0.85)
    axes[1].set(xlabel="Layer", ylabel="Causal Effect", title="Causal Trace: Retain Set")
    axes[1].grid(axis="y", alpha=0.3)

    colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in diff]
    axes[2].bar(layers, diff, color=colors, alpha=0.85)
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set(xlabel="Layer", ylabel="Differential (Forget - Retain)", title="Causal Differential")
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
        if not keys:
            ax.set_visible(False)
            continue
        fn = [forget_act[k]["l2_norm"] for k in keys]
        rn = [retain_act[k]["l2_norm"] for k in keys]
        ax.plot(idxs, fn, "r-o", label="Forget", markersize=4)
        ax.plot(idxs, rn, "g-s", label="Retain", markersize=4)
        ax.set(xlabel="Layer", ylabel="Mean L2 Norm", title=f"Activation Norms: {title}")
        ax.legend()
        ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for suffix, label, offset in [("", "Layer", -0.25), ("_mlp", "MLP", 0.0), ("_attn", "Attn", 0.25)]:
        keys, idxs = _sorted_layer_keys(forget_act, suffix)
        if not keys:
            continue
        diff = [forget_act[k]["l2_norm"] - retain_act[k]["l2_norm"] for k in keys]
        ax.bar(np.array(idxs) + offset, diff, width=0.25, label=label, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set(xlabel="Layer", ylabel="Differential (Forget - Retain)",
           title="Activation Differential by Component")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "activation_traces.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_gradient_differential(forget_grads, retain_grads, diff_scores, n_layers, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fg_layer = _aggregate_grads_by_layer(forget_grads, n_layers)
    rg_layer = _aggregate_grads_by_layer(retain_grads, n_layers)
    layers = sorted(fg_layer.keys())
    w = 0.35
    axes[0].bar(np.array(layers) - w/2, [fg_layer[l] for l in layers], w,
                label="Forget", color="#e74c3c", alpha=0.85)
    axes[0].bar(np.array(layers) + w/2, [rg_layer[l] for l in layers], w,
                label="Retain", color="#2ecc71", alpha=0.85)
    axes[0].set(xlabel="Layer", ylabel="Mean |Gradient|", title="Per-Layer Gradient Magnitude")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    ratios = [fg_layer[l] / (rg_layer[l] + 1e-10) for l in layers]
    colors = ["#e74c3c" if r > 1.0 else "#2ecc71" for r in ratios]
    axes[1].bar(layers, ratios, color=colors, alpha=0.85)
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set(xlabel="Layer", ylabel="Ratio (Forget / Retain)",
                title="Per-Layer Gradient Balance (Forget / Retain)")
    axes[1].grid(axis="y", alpha=0.3)

    top_n = 30
    def _to_scalar(v):
        return float(v.mean()) if isinstance(v, torch.Tensor) else float(v)
    sorted_p = sorted(diff_scores.items(), key=lambda x: _to_scalar(x[1]), reverse=True)[:top_n]
    names = [p.replace("model.layers.", "L").replace(".weight", "")[:40] for p, _ in sorted_p]
    vals = [_to_scalar(v) for _, v in sorted_p]
    axes[2].barh(range(len(names)), vals, color="#e74c3c", alpha=0.85)
    axes[2].set_yticks(range(len(names)))
    axes[2].set_yticklabels(names, fontsize=7)
    axes[2].set(xlabel="Differential Score", title=f"Top {top_n} Least Retain-Biased Parameters")
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
    axes[0, 0].set(xlabel="Layer", ylabel="Causal Effect", title="Causal Tracing")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    diff_c = [f - r for f, r in zip(fc, rc)]
    axes[0, 1].bar(layers, diff_c, color=["#e74c3c" if d > 0 else "#2ecc71" for d in diff_c], alpha=0.85)
    axes[0, 1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 1].set(xlabel="Layer", ylabel="Causal Differential",
                   title="Forget-Critical (red) vs Retain-Critical (green)")
    axes[0, 1].grid(axis="y", alpha=0.3)

    fg_l = _aggregate_grads_by_layer(forget_grads, n_layers)
    rg_l = _aggregate_grads_by_layer(retain_grads, n_layers)
    ratios = [fg_l.get(l, 0) / (rg_l.get(l, 0) + 1e-10) for l in layers]
    axes[1, 0].bar(layers, ratios, color=["#e74c3c" if r > 1 else "#2ecc71" for r in ratios], alpha=0.85)
    axes[1, 0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 0].set(xlabel="Layer", ylabel="Gradient Ratio",
                   title="Gradient Balance per Layer (Forget / Retain)")
    axes[1, 0].grid(axis="y", alpha=0.3)

    keys, idxs = _sorted_layer_keys(forget_act, "")
    if keys:
        axes[1, 1].plot(idxs, [forget_act[k]["l2_norm"] for k in keys], "r-o", label="Forget", markersize=4)
        axes[1, 1].plot(idxs, [retain_act[k]["l2_norm"] for k in keys], "g-s", label="Retain", markersize=4)
    axes[1, 1].set(xlabel="Layer", ylabel="Mean L2 Norm", title="Activation Norms per Layer")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


# ===================================================================
# Console summary
# ===================================================================

def print_summary(causal_f, causal_r, forget_grads, retain_grads, diff_scores, n_layers):
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

    print("\n--- Gradient: Top 10 Layers by Forget/Retain Ratio ---")
    fg_l = _aggregate_grads_by_layer(forget_grads, n_layers)
    rg_l = _aggregate_grads_by_layer(retain_grads, n_layers)
    lr = sorted([(l, fg_l[l] / (rg_l[l] + 1e-10)) for l in range(n_layers)],
                key=lambda x: x[1], reverse=True)
    print(f"{'Layer':>6} {'Forget |g|':>14} {'Retain |g|':>14} {'Ratio':>10}")
    print("-" * 48)
    for layer, ratio in lr[:10]:
        print(f"{layer:>6d} {fg_l[layer]:>14.6f} {rg_l[layer]:>14.6f} {ratio:>10.2f}")

    print("\n--- Top 20 Least Retain-Biased Parameters ---")
    def _sc(v): return float(v.mean()) if isinstance(v, torch.Tensor) else float(v)
    sp = sorted(diff_scores.items(), key=lambda x: _sc(x[1]), reverse=True)
    print(f"{'Parameter':>55} {'Score':>10}")
    print("-" * 67)
    for name, score in sp[:20]:
        print(f"{name[-55:]:>55} {_sc(score):>10.4f}")

    forget_layers = sorted([l for l in causal_f if causal_f[l] > causal_r.get(l, 0)])
    print(f"\nForget-dominant layers (causal): {forget_layers}")
    print("=" * 80 + "\n")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect mechanistic interpretability traces for unlearning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Preset
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()),
                        help=f"Named config preset. Options: {list(PRESETS.keys())}. "
                             "All preset values can be overridden by explicit flags.")

    # Model / tokenizer
    parser.add_argument("--model_name", type=str,
                        help="HuggingFace model id or local path (default: from preset)")
    parser.add_argument("--tokenizer", type=str,
                        help="Tokenizer id (default: same as --model_name)")

    # Dataset
    parser.add_argument("--dataset", type=str,
                        help="HuggingFace dataset id (default: from preset)")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="HuggingFace dataset config/subset name (e.g. 'raw', 'train', 'wmdp-bio')")
    parser.add_argument("--forget_split", type=str, default="forget")
    parser.add_argument("--retain_split", type=str, default="retain1",
                        help="Retain split name. For WMDP, a separate retain dataset is used automatically.")
    parser.add_argument("--retain_dataset", type=str, default=None,
                        help="Override retain dataset (default: same as --dataset). "
                             "For WMDP this is set automatically to cais/wmdp wmdp-retain.")
    parser.add_argument("--retain_dataset_config", type=str, default=None)
    parser.add_argument("--text_field", type=str, default="text",
                        help="Dataset field containing text (default: 'text'; WMDP uses 'question')")
    parser.add_argument("--retain_text_field", type=str, default=None,
                        help="Text field for retain dataset (default: same as --text_field)")

    # Sampling
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Max samples per split (default: from preset or all)")

    # Tokenization
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: from preset or auto-generated)")

    # Tracing options
    parser.add_argument("--causal_noise_std", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_causal", action="store_true")
    parser.add_argument("--skip_gradients", action="store_true")
    parser.add_argument("--skip_activations", action="store_true")

    args = parser.parse_args()

    # ── Apply preset defaults (CLI args override preset) ──────────────
    if args.preset:
        preset = PRESETS[args.preset]
        for key, val in preset.items():
            cli_val = getattr(args, key, None)
            if cli_val is None or (isinstance(cli_val, str) and cli_val == parser.get_default(key)):
                setattr(args, key, val)

    # ── Validate required args ─────────────────────────────────────────
    if not args.model_name:
        parser.error("--model_name is required (or use --preset)")
    if not args.dataset:
        parser.error("--dataset is required (or use --preset)")

    # ── Tokenizer defaults to model_name ──────────────────────────────
    if not args.tokenizer:
        args.tokenizer = args.model_name

    # ── Auto output_dir ────────────────────────────────────────────────
    if not args.output_dir:
        slug = args.preset or re.sub(r"[^a-zA-Z0-9_]", "_", args.dataset).strip("_")
        args.output_dir = f"trace_analysis/figures/traces/{slug}"

    # ── WMDP: use separate retain dataset ─────────────────────────────
    if args.retain_dataset is None and "wmdp" in args.dataset.lower():
        args.retain_dataset = WMDP_RETAIN_DATASET
        args.retain_dataset_config = WMDP_RETAIN_CONFIG
        args.retain_split = WMDP_RETAIN_SPLIT
        if args.retain_text_field is None:
            args.retain_text_field = WMDP_RETAIN_TEXT_FIELD

    retain_dataset = args.retain_dataset or args.dataset
    retain_config = args.retain_dataset_config or args.dataset_config
    retain_text_field = args.retain_text_field or args.text_field

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Preset:  {args.preset or 'none'}")
    logger.info(f"Model:   {args.model_name}")
    logger.info(f"Dataset: {args.dataset} [{args.dataset_config}] forget={args.forget_split}")
    logger.info(f"Retain:  {retain_dataset} [{retain_config}] split={args.retain_split}")
    logger.info(f"Output:  {args.output_dir}")

    # ── Load model ─────────────────────────────────────────────────────
    logger.info("Loading model...")
    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers = get_model_layers(model)
    n_layers = len(layers)
    logger.info(f"  {n_layers} layers, arch={type(model).__name__}")

    # ── Load data ──────────────────────────────────────────────────────
    forget_ds = load_and_tokenize(
        tokenizer, args.dataset, args.forget_split,
        args.n_samples, args.max_length,
        dataset_config=args.dataset_config, text_field=args.text_field)

    retain_ds = load_and_tokenize(
        tokenizer, retain_dataset, args.retain_split,
        args.n_samples, args.max_length,
        dataset_config=retain_config, text_field=retain_text_field)

    forget_loader = DataLoader(forget_ds, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_fn)
    retain_loader = DataLoader(retain_ds, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_fn)

    logger.info(f"  Forget: {len(forget_ds)} samples, Retain: {len(retain_ds)} samples")

    results = {
        "metadata": {
            "model": args.model_name,
            "tokenizer": args.tokenizer,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "forget_split": args.forget_split,
            "retain_dataset": retain_dataset,
            "retain_dataset_config": retain_config,
            "retain_split": args.retain_split,
            "text_field": args.text_field,
            "n_forget_samples": len(forget_ds),
            "n_retain_samples": len(retain_ds),
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "n_layers": n_layers,
            "causal_noise_std": args.causal_noise_std,
            "preset": args.preset,
            "timestamp": datetime.now().isoformat(),
        }
    }
    t_total = time.time()

    # ── Causal Tracing ─────────────────────────────────────────────────
    if not args.skip_causal:
        t0 = time.time()
        causal_forget = collect_causal_traces(model, layers, forget_loader,
                                              args.causal_noise_std, args.device)
        causal_retain = collect_causal_traces(model, layers, retain_loader,
                                              args.causal_noise_std, args.device)
        results["causal_traces"] = {"forget": causal_forget, "retain": causal_retain}
        logger.info(f"  Causal tracing: {time.time()-t0:.1f}s")

    # ── Gradient Differential ──────────────────────────────────────────
    if not args.skip_gradients:
        t0 = time.time()
        for p in model.parameters():
            p.requires_grad_(True)

        forget_grads = collect_gradient_traces(model, forget_loader, args.device)
        retain_grads = collect_gradient_traces(model, retain_loader, args.device)
        diff_scores = compute_differential_scores(forget_grads, retain_grads)

        results["gradient_traces"] = {"forget": forget_grads, "retain": retain_grads}
        results["differential_scores"] = diff_scores

        for p in model.parameters():
            p.requires_grad_(False)
        model.zero_grad()
        torch.cuda.empty_cache()
        logger.info(f"  Gradient traces: {time.time()-t0:.1f}s")

    # ── Activation Statistics ──────────────────────────────────────────
    if not args.skip_activations:
        t0 = time.time()
        forget_act = collect_activation_traces(model, layers, forget_loader, args.device)
        retain_act = collect_activation_traces(model, layers, retain_loader, args.device)
        layer_diff = compute_layer_differential(forget_act, retain_act)
        results["activation_traces"] = {"forget": forget_act, "retain": retain_act}
        results["layer_differential"] = layer_diff
        logger.info(f"  Activation traces: {time.time()-t0:.1f}s")

    total_time = time.time() - t_total
    results["metadata"]["total_time_seconds"] = round(total_time, 1)

    # ── Save ───────────────────────────────────────────────────────────
    pt_path = os.path.join(args.output_dir, "trace_results.pt")
    torch.save(results, pt_path)
    logger.info(f"Saved {pt_path}")

    # Save neuron_traces.pt in format expected by SIBL trainer
    if "gradient_traces" in results:
        neuron_traces = {
            "forget": results["gradient_traces"]["forget"],
            "retain": results["gradient_traces"]["retain"],
            "elapsed_seconds": results["metadata"].get("total_time_seconds", 0),
        }
        neuron_traces_path = os.path.join(args.output_dir, "neuron_traces.pt")
        torch.save(neuron_traces, neuron_traces_path)
        logger.info(f"Saved {neuron_traces_path} (SIBL-compatible neuron traces)")

    json_out = {"metadata": results["metadata"]}
    if "causal_traces" in results:
        json_out["causal_traces"] = {
            "forget": {str(k): v for k, v in results["causal_traces"]["forget"].items()},
            "retain": {str(k): v for k, v in results["causal_traces"]["retain"].items()},
        }
    if "differential_scores" in results:
        def _score_val(v):
            import torch
            return float(v.mean()) if isinstance(v, torch.Tensor) else float(v)
        json_out["top_50_differential_params"] = {
            k: _score_val(v)
            for k, v in sorted(results["differential_scores"].items(),
                               key=lambda x: _score_val(x[1]), reverse=True)[:50]
        }
    if "layer_differential" in results:
        json_out["layer_differential"] = results["layer_differential"]
    if "activation_traces" in results:
        json_out["activation_traces"] = results["activation_traces"]

    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    logger.info(f"Saved {json_path}")

    # ── Plots ──────────────────────────────────────────────────────────
    if "causal_traces" in results:
        plot_causal_traces(results["causal_traces"]["forget"],
                           results["causal_traces"]["retain"], args.output_dir)
    if "activation_traces" in results:
        plot_activation_traces(results["activation_traces"]["forget"],
                               results["activation_traces"]["retain"], args.output_dir)
    if "gradient_traces" in results:
        plot_gradient_differential(results["gradient_traces"]["forget"],
                                   results["gradient_traces"]["retain"],
                                   results.get("differential_scores", {}),
                                   n_layers, args.output_dir)
    if all(k in results for k in ["causal_traces", "activation_traces", "gradient_traces"]):
        plot_summary(results["causal_traces"]["forget"], results["causal_traces"]["retain"],
                     results["activation_traces"]["forget"], results["activation_traces"]["retain"],
                     results["gradient_traces"]["forget"], results["gradient_traces"]["retain"],
                     n_layers, args.output_dir)

    if "causal_traces" in results and "gradient_traces" in results:
        print_summary(results["causal_traces"]["forget"], results["causal_traces"]["retain"],
                      results["gradient_traces"]["forget"], results["gradient_traces"]["retain"],
                      results.get("differential_scores", {}), n_layers)

    logger.info(f"Total time: {total_time:.1f}s  |  All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
