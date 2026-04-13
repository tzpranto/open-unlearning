#!/usr/bin/env python3
"""
Neuron-Level Trace Analysis for Machine Unlearning
====================================================

Loads the per-parameter gradient traces from trace_results.pt, decomposes
weight matrices into per-neuron (row-level) gradient magnitudes, and
identifies forget-dominant vs retain-dominant neurons.

Produces:
  1. Layer x Component heatmap of gradient differential ratios
  2. Per-layer neuron-level heatmaps for MLP and Attention projections
  3. Histogram of neuron-level differential scores
  4. Forget-neuron bitmap (which neurons have ratio > threshold)
  5. JSON summary of forget-dominant and retain-dominant neurons

Requires: the full trace collection must be run with per-neuron granularity.
For the existing per-parameter traces, this script re-loads the model and
computes per-neuron (row-wise) gradient norms on forget vs retain data.

Usage:
    python scripts/analyze_traces.py
    python scripts/analyze_traces.py --threshold 1.0 --output_dir saves/traces/analysis
"""

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ===================================================================
# Data helpers (shared with trace_activations.py)
# ===================================================================

class TextDataset(Dataset):
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
                      dataset_config="raw"):
    logger.info(f"Loading {dataset_name} [{dataset_config}] split={split_name} (n_samples={n_samples})")
    ds = load_dataset(dataset_name, name=dataset_config, split=split_name)
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))
    texts = list(ds["text"])
    enc = tokenizer(
        texts, max_length=max_length, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    return TextDataset(enc["input_ids"], enc["attention_mask"])


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


# ===================================================================
# Per-neuron gradient collection
# ===================================================================

def collect_per_neuron_gradients(model, dataloader, device, target_components):
    """Compute per-neuron (row-wise) mean |gradient| for target weight matrices.

    For a weight matrix W of shape (out_features, in_features), each "neuron"
    is one row (one output dimension). We compute mean(|grad|) per row.

    Args:
        target_components: list of regex patterns for parameter names to track.

    Returns:
        Dict[param_name, np.array of shape (out_features,)]
    """
    logger.info("=== Per-Neuron Gradient Collection ===")

    patterns = [re.compile(p) for p in target_components]
    accum = {}
    n_batches = 0

    for bi, batch in enumerate(dataloader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        model.zero_grad()
        loss = model(input_ids=ids, attention_mask=mask, labels=ids.clone()).loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None or param.dim() < 2:
                continue
            if not any(p.search(name) for p in patterns):
                continue
            # Row-wise mean |grad|: shape (out_features,)
            row_grad = param.grad.abs().mean(dim=1).float().cpu().numpy()
            if name not in accum:
                accum[name] = np.zeros_like(row_grad)
            accum[name] += row_grad

        model.zero_grad()
        n_batches += 1

        if (bi + 1) % 10 == 0 or bi == len(dataloader) - 1:
            logger.info(f"  Neuron gradients  batch {bi+1}/{len(dataloader)}")

    for name in accum:
        accum[name] /= n_batches

    return accum


# ===================================================================
# Analysis
# ===================================================================

def compute_neuron_differential(forget_neurons, retain_neurons, eps=1e-10):
    """Per-neuron forget/retain ratio for each parameter."""
    result = {}
    for name in forget_neurons:
        if name in retain_neurons:
            result[name] = forget_neurons[name] / (retain_neurons[name] + eps)
    return result


def extract_layer_component_matrix(param_traces, n_layers):
    """Build a (n_layers x n_components) matrix of mean gradient magnitudes.

    Components: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj,
                input_layernorm, post_attention_layernorm
    """
    components = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "input_layernorm", "post_attention_layernorm",
    ]
    matrix = np.zeros((n_layers, len(components)))
    for name, val in param_traces.items():
        m = re.search(r"model\.layers\.(\d+)\.", name)
        if not m:
            continue
        li = int(m.group(1))
        for ci, comp in enumerate(components):
            if comp in name:
                import torch as _torch
                if isinstance(val, float):
                    matrix[li, ci] = val
                elif isinstance(val, _torch.Tensor):
                    matrix[li, ci] = val.float().mean().item()
                else:
                    matrix[li, ci] = np.mean(val)
                break
    return matrix, components


def find_forget_neurons(neuron_diff, threshold=1.0):
    """Find neurons with differential ratio above threshold.

    Returns list of (param_name, neuron_idx, ratio) sorted by ratio descending.
    """
    results = []
    for name, ratios in neuron_diff.items():
        for idx, r in enumerate(ratios):
            if r >= threshold:
                results.append((name, int(idx), float(r)))
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def find_retain_neurons(neuron_diff, threshold=0.5):
    """Find strongly retain-dominant neurons (ratio below threshold)."""
    results = []
    for name, ratios in neuron_diff.items():
        for idx, r in enumerate(ratios):
            if r <= threshold:
                results.append((name, int(idx), float(r)))
    results.sort(key=lambda x: x[2])
    return results


def build_neuron_bitmap(neuron_diff, forget_threshold=1.0):
    """Build a binary mask: 1 = forget-dominant neuron, 0 = not.

    Returns Dict[param_name, np.array of 0/1].
    """
    bitmap = {}
    for name, ratios in neuron_diff.items():
        bitmap[name] = (ratios >= forget_threshold).astype(np.int32)
    return bitmap


# ===================================================================
# Plotting
# ===================================================================

def plot_layer_component_heatmap(forget_matrix, retain_matrix, components,
                                 n_layers, output_dir):
    """Heatmap of forget/retain gradient ratio per (layer, component)."""
    ratio_matrix = forget_matrix / (retain_matrix + 1e-10)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Forget magnitude
    im0 = axes[0].imshow(forget_matrix, aspect="auto", cmap="Reds")
    axes[0].set_title("Forget Gradient Magnitude")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Layer")
    axes[0].set_xticks(range(len(components)))
    axes[0].set_xticklabels(components, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(0, n_layers, 2))
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Retain magnitude
    im1 = axes[1].imshow(retain_matrix, aspect="auto", cmap="Greens")
    axes[1].set_title("Retain Gradient Magnitude")
    axes[1].set_xlabel("Component")
    axes[1].set_xticks(range(len(components)))
    axes[1].set_xticklabels(components, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(0, n_layers, 2))
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Ratio heatmap with diverging colormap centered at 1.0
    norm = mcolors.TwoSlopeNorm(vmin=ratio_matrix.min(), vcenter=1.0,
                                 vmax=max(ratio_matrix.max(), 1.05))
    im2 = axes[2].imshow(ratio_matrix, aspect="auto", cmap="RdYlGn_r", norm=norm)
    axes[2].set_title("Differential Ratio (Forget / Retain)\n>1 = forget-dominant, <1 = retain-dominant")
    axes[2].set_xlabel("Component")
    axes[2].set_xticks(range(len(components)))
    axes[2].set_xticklabels(components, rotation=45, ha="right", fontsize=8)
    axes[2].set_yticks(range(0, n_layers, 2))
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, "layer_component_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_neuron_heatmaps(neuron_diff, n_layers, output_dir):
    """Per-layer neuron heatmaps for MLP and attention projections."""

    # Group by component type
    component_groups = {
        "MLP gate_proj": "gate_proj",
        "MLP up_proj": "up_proj",
        "MLP down_proj": "down_proj",
        "Attn q_proj": "q_proj",
        "Attn k_proj": "k_proj",
        "Attn v_proj": "v_proj",
        "Attn o_proj": "o_proj",
    }

    for group_name, comp_key in component_groups.items():
        # Collect data for this component across layers
        layer_data = {}
        for name, ratios in neuron_diff.items():
            m = re.search(r"model\.layers\.(\d+)\.", name)
            if not m:
                continue
            if comp_key not in name:
                continue
            li = int(m.group(1))
            layer_data[li] = ratios

        if not layer_data:
            continue

        layers = sorted(layer_data.keys())
        n_neurons = len(layer_data[layers[0]])

        # Build matrix (n_layers x n_neurons)
        matrix = np.zeros((len(layers), n_neurons))
        for i, li in enumerate(layers):
            matrix[i] = layer_data[li]

        fig, ax = plt.subplots(figsize=(max(16, n_neurons // 50), 10))
        norm = mcolors.TwoSlopeNorm(
            vmin=max(matrix.min(), 0.1), vcenter=1.0,
            vmax=min(matrix.max(), 3.0))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", norm=norm,
                        interpolation="nearest")
        ax.set_title(f"Neuron-Level Differential: {group_name}\n"
                     f"Red = forget-dominant (ratio > 1), Green = retain-dominant (ratio < 1)")
        ax.set_ylabel("Layer")
        ax.set_xlabel(f"Neuron Index (0..{n_neurons-1})")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Forget/Retain Ratio")

        # Mark neurons above threshold
        for i, li in enumerate(layers):
            forget_idxs = np.where(layer_data[li] > 1.0)[0]
            if len(forget_idxs) > 0 and len(forget_idxs) < 50:
                for idx in forget_idxs:
                    ax.plot(idx, i, 'k.', markersize=1)

        plt.tight_layout()
        safe_name = group_name.replace(" ", "_").lower()
        path = os.path.join(output_dir, f"neuron_heatmap_{safe_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {path}")


def plot_neuron_histogram(neuron_diff, output_dir):
    """Histogram of all neuron-level differential ratios."""
    all_ratios = []
    for ratios in neuron_diff.values():
        all_ratios.extend(ratios.tolist())

    all_ratios = np.array(all_ratios)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Full histogram
    axes[0].hist(all_ratios, bins=200, color="#3498db", alpha=0.8, edgecolor="none")
    axes[0].axvline(1.0, color="red", linewidth=1.5, linestyle="--", label="ratio=1.0")
    axes[0].set_xlabel("Forget / Retain Gradient Ratio")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Neuron-Level Differential Distribution\n"
                      f"Total: {len(all_ratios):,} neurons, "
                      f"Forget-dominant (>1.0): {(all_ratios > 1.0).sum():,} "
                      f"({100*(all_ratios > 1.0).mean():.1f}%)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Zoomed into 0.8-1.2 range
    zoomed = all_ratios[(all_ratios > 0.8) & (all_ratios < 1.2)]
    axes[1].hist(zoomed, bins=200, color="#3498db", alpha=0.8, edgecolor="none")
    axes[1].axvline(1.0, color="red", linewidth=1.5, linestyle="--", label="ratio=1.0")
    axes[1].set_xlabel("Forget / Retain Gradient Ratio")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Zoomed: 0.8-1.2 Range\n"
                      f"Neurons in range: {len(zoomed):,}")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "neuron_differential_histogram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_forget_neuron_bitmap(neuron_bitmap, neuron_diff, n_layers, output_dir):
    """Bitmap visualization: which neurons are forget-dominant."""

    # Group by component
    comp_keys = ["gate_proj", "up_proj", "down_proj",
                 "q_proj", "k_proj", "v_proj", "o_proj"]
    comp_labels = ["MLP gate", "MLP up", "MLP down",
                   "Attn Q", "Attn K", "Attn V", "Attn O"]

    fig, axes = plt.subplots(len(comp_keys), 1, figsize=(20, 3 * len(comp_keys)))

    total_forget = 0
    total_neurons = 0

    for ax, comp_key, comp_label in zip(axes, comp_keys, comp_labels):
        layer_bitmaps = {}
        for name, bm in neuron_bitmap.items():
            m = re.search(r"model\.layers\.(\d+)\.", name)
            if not m or comp_key not in name:
                continue
            li = int(m.group(1))
            layer_bitmaps[li] = bm

        if not layer_bitmaps:
            ax.set_visible(False)
            continue

        layers = sorted(layer_bitmaps.keys())
        n_neurons = len(layer_bitmaps[layers[0]])
        matrix = np.zeros((len(layers), n_neurons))
        for i, li in enumerate(layers):
            matrix[i] = layer_bitmaps[li]

        n_forget = int(matrix.sum())
        n_total = matrix.size
        total_forget += n_forget
        total_neurons += n_total

        cmap = plt.cm.colors.ListedColormap(["#f0f0f0", "#e74c3c"])
        ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_title(f"{comp_label}: {n_forget}/{n_total} forget-dominant neurons "
                     f"({100*n_forget/max(n_total,1):.1f}%)")
        ax.set_ylabel("Layer")
        ax.set_xlabel(f"Neuron (0..{n_neurons-1})")
        ax.set_yticks(range(0, len(layers), 2))
        ax.set_yticklabels([layers[i] for i in range(0, len(layers), 2)])

    fig.suptitle(f"Forget-Dominant Neuron Bitmap (ratio > 1.0)\n"
                 f"Total: {total_forget}/{total_neurons} "
                 f"({100*total_forget/max(total_neurons,1):.2f}%)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "forget_neuron_bitmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")

    return total_forget, total_neurons


def plot_activation_layer_heatmap(trace_results, n_layers, output_dir):
    """Heatmap of activation L2 norms: layer x {layer, mlp, attn}."""
    fa = trace_results["activation_traces"]["forget"]
    ra = trace_results["activation_traces"]["retain"]

    metrics = [("l2_norm", "L2 Norm"), ("mean_abs", "Mean |Act|"), ("var", "Variance")]
    suffixes = [("", "Layer"), ("_mlp", "MLP"), ("_attn", "Attn")]

    for metric_key, metric_label in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        for ax_idx, (suffix, suffix_label) in enumerate(suffixes):
            f_vals = [fa[f"layer_{li}{suffix}"][metric_key] for li in range(n_layers)]
            r_vals = [ra[f"layer_{li}{suffix}"][metric_key] for li in range(n_layers)]
            diff = [f - r for f, r in zip(f_vals, r_vals)]

            data = np.array([f_vals, r_vals, diff])
            row_labels = ["Forget", "Retain", "Diff (F-R)"]

            im = axes[ax_idx].imshow(data, aspect="auto", cmap="coolwarm")
            axes[ax_idx].set_title(f"{suffix_label} {metric_label}")
            axes[ax_idx].set_xlabel("Layer")
            axes[ax_idx].set_xticks(range(0, n_layers, 2))
            axes[ax_idx].set_xticklabels(range(0, n_layers, 2))
            axes[ax_idx].set_yticks(range(3))
            axes[ax_idx].set_yticklabels(row_labels)
            plt.colorbar(im, ax=axes[ax_idx], shrink=0.6)

        plt.suptitle(f"Activation {metric_label} by Layer and Component", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(output_dir, f"activation_heatmap_{metric_key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Neuron-level trace analysis for machine unlearning")

    parser.add_argument("--model_name", type=str,
                        default="muse-bench/MUSE-News_target")
    parser.add_argument("--tokenizer", type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="muse-bench/MUSE-News")
    parser.add_argument("--dataset_config", type=str, default="raw",
                        help="HuggingFace dataset config name (e.g. 'raw', 'train')")
    parser.add_argument("--forget_split", type=str, default="forget")
    parser.add_argument("--retain_split", type=str, default="retain1")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Max samples per split (uses all available if larger)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--trace_file", type=str,
                        default="trace_analysis/figures/traces/muse_news/trace_results.pt",
                        help="Existing param-level traces (for layer-component heatmap)")
    parser.add_argument("--output_dir", type=str,
                        default="trace_analysis/figures/traces/analysis")
    parser.add_argument("--forget_threshold", type=float, default=1.0,
                        help="Ratio threshold for marking a neuron as forget-dominant")
    parser.add_argument("--retain_threshold", type=float, default=0.5,
                        help="Ratio threshold for marking a neuron as strongly retain-dominant")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_neuron_collection", action="store_true",
                        help="Skip GPU neuron gradient collection, use existing data")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load existing param-level traces for layer-component heatmap
    # ------------------------------------------------------------------
    logger.info("Loading existing param-level traces ...")
    trace_data = torch.load(args.trace_file, map_location="cpu")
    n_layers = trace_data["metadata"]["n_layers"]

    fg_param = trace_data["gradient_traces"]["forget"]
    rg_param = trace_data["gradient_traces"]["retain"]
    diff_param = trace_data["differential_scores"]

    # Layer x Component heatmap from param-level data
    fg_matrix, components = extract_layer_component_matrix(fg_param, n_layers)
    rg_matrix, _ = extract_layer_component_matrix(rg_param, n_layers)

    plot_layer_component_heatmap(fg_matrix, rg_matrix, components,
                                 n_layers, args.output_dir)

    # Activation heatmaps from existing traces
    if "activation_traces" in trace_data:
        plot_activation_layer_heatmap(trace_data, n_layers, args.output_dir)

    # ------------------------------------------------------------------
    # Step 2: Per-neuron gradient collection (requires GPU)
    # ------------------------------------------------------------------
    neuron_pt_path = os.path.join(args.output_dir, "neuron_traces.pt")

    if args.skip_neuron_collection and os.path.exists(neuron_pt_path):
        logger.info(f"Loading cached neuron traces from {neuron_pt_path}")
        neuron_data = torch.load(neuron_pt_path, map_location="cpu")
        forget_neurons = neuron_data["forget"]
        retain_neurons = neuron_data["retain"]
    else:
        logger.info("Loading model for neuron-level gradient collection ...")
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        ).to(args.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        forget_ds = load_and_tokenize(
            tokenizer, args.dataset, args.forget_split,
            args.n_samples, args.max_length,
            dataset_config=args.dataset_config)
        retain_ds = load_and_tokenize(
            tokenizer, args.dataset, args.retain_split,
            args.n_samples, args.max_length,
            dataset_config=args.dataset_config)

        forget_loader = DataLoader(forget_ds, batch_size=args.batch_size,
                                    shuffle=False, collate_fn=collate_fn)
        retain_loader = DataLoader(retain_ds, batch_size=args.batch_size,
                                    shuffle=False, collate_fn=collate_fn)

        # Target: all projection weights in attention and MLP
        target_patterns = [
            r"self_attn\.(q|k|v|o)_proj\.weight",
            r"mlp\.(gate|up|down)_proj\.weight",
        ]

        t0 = time.time()
        logger.info("Collecting forget-set neuron gradients ...")
        forget_neurons = collect_per_neuron_gradients(
            model, forget_loader, args.device, target_patterns)

        logger.info("Collecting retain-set neuron gradients ...")
        retain_neurons = collect_per_neuron_gradients(
            model, retain_loader, args.device, target_patterns)

        elapsed = time.time() - t0
        logger.info(f"Neuron gradient collection: {elapsed:.1f}s")

        # Cache results
        torch.save({"forget": forget_neurons, "retain": retain_neurons,
                     "elapsed_seconds": elapsed},
                    neuron_pt_path)
        logger.info(f"Saved {neuron_pt_path}")

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 3: Neuron-level analysis
    # ------------------------------------------------------------------
    logger.info("Computing neuron-level differential scores ...")
    neuron_diff = compute_neuron_differential(forget_neurons, retain_neurons)

    # Statistics
    all_ratios = np.concatenate([v for v in neuron_diff.values()])
    n_total = len(all_ratios)
    n_forget = (all_ratios > args.forget_threshold).sum()
    n_retain = (all_ratios < args.retain_threshold).sum()

    logger.info(f"  Total neurons tracked: {n_total:,}")
    logger.info(f"  Forget-dominant (>{args.forget_threshold}): {n_forget:,} ({100*n_forget/n_total:.2f}%)")
    logger.info(f"  Retain-dominant (<{args.retain_threshold}): {n_retain:,} ({100*n_retain/n_total:.2f}%)")
    logger.info(f"  Ratio range: [{all_ratios.min():.4f}, {all_ratios.max():.4f}]")
    logger.info(f"  Ratio mean: {all_ratios.mean():.4f}, median: {np.median(all_ratios):.4f}")

    # Find specific neurons
    forget_list = find_forget_neurons(neuron_diff, threshold=args.forget_threshold)
    retain_list = find_retain_neurons(neuron_diff, threshold=args.retain_threshold)

    logger.info(f"  Top forget neurons: {len(forget_list)}")
    for name, idx, ratio in forget_list[:10]:
        short = name.replace("model.layers.", "L").replace(".weight", "")
        logger.info(f"    {short} neuron {idx}: ratio={ratio:.4f}")

    # Build bitmap
    neuron_bitmap = build_neuron_bitmap(neuron_diff, args.forget_threshold)

    # ------------------------------------------------------------------
    # Step 4: Plots
    # ------------------------------------------------------------------
    logger.info("Generating neuron-level plots ...")
    plot_neuron_heatmaps(neuron_diff, n_layers, args.output_dir)
    plot_neuron_histogram(neuron_diff, args.output_dir)
    total_forget, total_neurons = plot_forget_neuron_bitmap(
        neuron_bitmap, neuron_diff, n_layers, args.output_dir)

    # ------------------------------------------------------------------
    # Step 5: Per-layer summary stats for the report
    # ------------------------------------------------------------------
    layer_summary = {}
    for li in range(n_layers):
        layer_neurons = {}
        for name, ratios in neuron_diff.items():
            m = re.search(r"model\.layers\.(\d+)\.", name)
            if m and int(m.group(1)) == li:
                comp = name.split(".")[-2]  # e.g. q_proj
                layer_neurons[comp] = {
                    "n_total": len(ratios),
                    "n_forget": int((ratios > args.forget_threshold).sum()),
                    "n_retain": int((ratios < args.retain_threshold).sum()),
                    "mean_ratio": float(ratios.mean()),
                    "max_ratio": float(ratios.max()),
                    "min_ratio": float(ratios.min()),
                }
        if layer_neurons:
            total_f = sum(v["n_forget"] for v in layer_neurons.values())
            total_n = sum(v["n_total"] for v in layer_neurons.values())
            layer_summary[li] = {
                "components": layer_neurons,
                "total_forget_neurons": total_f,
                "total_neurons": total_n,
                "pct_forget": round(100 * total_f / max(total_n, 1), 2),
            }

    # ------------------------------------------------------------------
    # Step 6: Save JSON summary
    # ------------------------------------------------------------------
    summary = {
        "metadata": {
            "model": args.model_name,
            "dataset": args.dataset,
            "n_layers": n_layers,
            "forget_threshold": args.forget_threshold,
            "retain_threshold": args.retain_threshold,
            "trace_file": args.trace_file,
        },
        "global_stats": {
            "total_neurons_tracked": int(n_total),
            "forget_dominant": int(n_forget),
            "retain_dominant": int(n_retain),
            "pct_forget": round(100 * n_forget / n_total, 2),
            "pct_retain": round(100 * n_retain / n_total, 2),
            "ratio_min": float(all_ratios.min()),
            "ratio_max": float(all_ratios.max()),
            "ratio_mean": float(all_ratios.mean()),
            "ratio_median": float(np.median(all_ratios)),
        },
        "layer_summary": {str(k): v for k, v in layer_summary.items()},
        "top_100_forget_neurons": [
            {"param": n, "neuron_idx": i, "ratio": r}
            for n, i, r in forget_list[:100]
        ],
        "top_100_retain_neurons": [
            {"param": n, "neuron_idx": i, "ratio": r}
            for n, i, r in retain_list[:100]
        ],
    }

    json_path = os.path.join(args.output_dir, "neuron_analysis.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved {json_path}")

    # Save bitmap as .pt for downstream use (e.g. SIBL mask)
    bitmap_path = os.path.join(args.output_dir, "forget_neuron_bitmap.pt")
    torch.save(neuron_bitmap, bitmap_path)
    logger.info(f"Saved {bitmap_path}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("NEURON-LEVEL ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTotal neurons tracked: {n_total:,}")
    print(f"Forget-dominant (ratio > {args.forget_threshold}): "
          f"{n_forget:,} ({100*n_forget/n_total:.2f}%)")
    print(f"Retain-dominant (ratio < {args.retain_threshold}): "
          f"{n_retain:,} ({100*n_retain/n_total:.2f}%)")
    print(f"Ratio range: [{all_ratios.min():.4f}, {all_ratios.max():.4f}]")
    print(f"Ratio mean: {all_ratios.mean():.4f}, median: {np.median(all_ratios):.4f}")

    print(f"\n{'Layer':>5} {'Forget Neurons':>15} {'Total':>8} {'%':>8}")
    print("-" * 40)
    for li in range(n_layers):
        if li in layer_summary:
            ls = layer_summary[li]
            print(f"{li:>5} {ls['total_forget_neurons']:>15} "
                  f"{ls['total_neurons']:>8} {ls['pct_forget']:>7.2f}%")

    print(f"\nTop 20 forget-dominant neurons:")
    print(f"{'Parameter':>55} {'Neuron':>7} {'Ratio':>8}")
    print("-" * 72)
    for name, idx, ratio in forget_list[:20]:
        short = name[-55:] if len(name) > 55 else name
        print(f"{short:>55} {idx:>7} {ratio:>8.4f}")

    print(f"\nAll outputs in {args.output_dir}/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
