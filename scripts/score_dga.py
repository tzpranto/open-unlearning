"""
Differential Gradient Attribution (DGA) Scoring
================================================

Computes per-parameter selectivity score:
  s_n = (g_fgt - g_ret) / (g_fgt + g_ret + eps)  ∈ [-1, 1]

where:
  g_fgt = mean |∂L_NPO/∂w|  (NPO loss on forget data, using G1 checkpoint)
  g_ret = mean |∂L_CE/∂w|   (CE loss on retain data)

  s_n → +1: forget-dominant (update hurts forgetting)
  s_n → -1: retain-dominant (safe to update during recovery)
  s_n ≈  0: contested / shared infrastructure

Selectivity is reduced to per-output-neuron (row) for 2D weight matrices,
consistent with the existing binary bitmap format.

Output: dga_selectivity.pt  — dict {param_name: tensor(out_features,)}

Usage:
  python scripts/score_dga.py \\
    --model_path saves/unlearn/ablation_G1_npo_weak_steering \\
    --ref_model muse-bench/MUSE-News_target \\
    --tokenizer_path meta-llama/Llama-2-7b-hf \\
    --forget_dataset muse-bench/MUSE-News --forget_config raw --forget_split forget \\
    --retain_dataset muse-bench/MUSE-News --retain_split retain1 \\
    --output_path trace_analysis/figures/traces/analysis/dga_selectivity_G1.pt \\
    --max_forget 802 --max_retain 200 --max_length 512 --npo_beta 2.0
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = []
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False,
            )
            if enc["input_ids"].shape[1] >= 2:
                self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def collate_fn(batch):
    """Pad a batch of variable-length sequences to the same length."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        L = item["input_ids"].shape[0]
        input_ids[i, :L] = item["input_ids"]
        attention_mask[i, :L] = item["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def compute_npo_loss(model, batch, ref_model, beta, device):
    """NPO loss: -2/beta * logsigmoid(beta * log(p_ref / p_theta))"""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    outputs = model(**inputs)
    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = outputs.logits[..., :-1, :].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    model_nll = loss_fn(shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

    with torch.no_grad():
        ref_outputs = ref_model(**inputs)
        ref_shifted_logits = ref_outputs.logits[..., :-1, :].contiguous()
        ref_nll = loss_fn(ref_shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

    log_ratio = -(model_nll - ref_nll)
    loss = -2.0 / beta * F.logsigmoid(beta * (-log_ratio)).mean()
    return loss


def compute_ce_loss(model, batch, device):
    """Standard cross-entropy loss."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )
    return outputs.loss


def accumulate_gradients(model, loader, loss_fn, device, desc=""):
    """Accumulate |grad| per-parameter over the loader. Returns dict {name: sum_abs_grad}."""
    grad_acc = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_acc[name] = torch.zeros_like(param.data, dtype=torch.float32)

    n_batches = 0
    for i, batch in enumerate(loader):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {desc} batch {i+1}/{len(loader)}")
        model.zero_grad()
        loss = loss_fn(batch)
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and name in grad_acc:
                    grad_acc[name] += param.grad.float().abs()
        n_batches += 1

    # Normalize by number of batches
    for name in grad_acc:
        if n_batches > 0:
            grad_acc[name] /= n_batches

    model.zero_grad()
    return grad_acc


def reduce_to_per_neuron(grad_dict, model):
    """Reduce per-element gradients to per-output-neuron (row mean) for each param.
    Returns dict {param_name: tensor(out_features,)}.
    For 1D params (biases), keeps as-is.
    """
    reduced = {}
    param_shapes = {n: p.shape for n, p in model.named_parameters()}
    for name, g in grad_dict.items():
        shape = param_shapes.get(name)
        if shape is None:
            continue
        if g.dim() == 2:
            # Weight matrix (out_features, in_features) → mean over in_features
            reduced[name] = g.mean(dim=1)  # (out_features,)
        elif g.dim() == 1:
            reduced[name] = g  # (out_features,) already
        else:
            # Higher-dim: flatten all but first dim
            reduced[name] = g.view(g.shape[0], -1).mean(dim=1)
    return reduced


def compute_selectivity(grad_fgt, grad_ret, eps=1e-8):
    """s = (fgt - ret) / (fgt + ret + eps) ∈ [-1, 1]"""
    selectivity = {}
    all_names = set(grad_fgt.keys()) | set(grad_ret.keys())
    for name in all_names:
        f = grad_fgt.get(name, torch.zeros(1))
        r = grad_ret.get(name, torch.zeros(1))
        s = (f - r) / (f + r + eps)
        selectivity[name] = s
    return selectivity


def print_stats(selectivity, thresholds=(0.3, -0.3)):
    """Print statistics on selectivity distribution."""
    all_vals = torch.cat([s.flatten() for s in selectivity.values()])
    th_high, th_low = thresholds
    n_forget = (all_vals > th_high).sum().item()
    n_retain = (all_vals < th_low).sum().item()
    n_contested = ((all_vals >= th_low) & (all_vals <= th_high)).sum().item()
    n_total = all_vals.numel()
    print(f"\n=== DGA Selectivity Statistics ===")
    print(f"  Total neurons scored: {n_total:,}")
    print(f"  Forget-dominant (s > {th_high}):  {n_forget:>8,} ({100*n_forget/n_total:.1f}%)")
    print(f"  Contested (|s| <= {th_high}):      {n_contested:>8,} ({100*n_contested/n_total:.1f}%)")
    print(f"  Retain-dominant (s < {th_low}): {n_retain:>8,} ({100*n_retain/n_total:.1f}%)")
    print(f"  Mean: {all_vals.mean():.4f}  Std: {all_vals.std():.4f}  "
          f"Min: {all_vals.min():.4f}  Max: {all_vals.max():.4f}")

    # Per-layer breakdown (MLP params only)
    print(f"\n  Per-layer MLP stats:")
    layer_stats = {}
    for name, s in selectivity.items():
        if "mlp" not in name:
            continue
        # Extract layer number
        parts = name.split(".")
        for j, p in enumerate(parts):
            if p == "layers" and j + 1 < len(parts):
                try:
                    lid = int(parts[j + 1])
                    if lid not in layer_stats:
                        layer_stats[lid] = []
                    layer_stats[lid].append(s.float())
                except ValueError:
                    pass
    for lid in sorted(layer_stats):
        vals = torch.cat([v.flatten() for v in layer_stats[lid]])
        frac_forget = (vals > th_high).float().mean().item()
        frac_retain = (vals < th_low).float().mean().item()
        print(f"    Layer {lid:2d}: fgt={frac_forget:.2f} ret={frac_retain:.2f} contested={1-frac_forget-frac_retain:.2f}  mean={vals.mean():.3f}")


def main():
    parser = argparse.ArgumentParser(description="DGA gradient attribution scoring")
    parser.add_argument("--model_path", default="saves/unlearn/ablation_G1_npo_weak_steering",
                        help="Path to model checkpoint to score (default: G1)")
    parser.add_argument("--ref_model", default="muse-bench/MUSE-News_target",
                        help="Reference model for NPO loss (pretrained/finetuned target)")
    parser.add_argument("--tokenizer_path", default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer path (muse-bench models have no tokenizer)")
    parser.add_argument("--forget_dataset", default="muse-bench/MUSE-News")
    parser.add_argument("--forget_config", default="raw")
    parser.add_argument("--forget_split", default="forget")
    parser.add_argument("--retain_dataset", default="muse-bench/MUSE-News")
    parser.add_argument("--retain_split", default="retain1")
    parser.add_argument("--output_path",
                        default="trace_analysis/figures/traces/analysis/dga_selectivity_G1.pt")
    parser.add_argument("--max_forget", type=int, default=0,
                        help="Max forget samples (0 = all)")
    parser.add_argument("--max_retain", type=int, default=200,
                        help="Max retain samples (0 = all)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--npo_beta", type=float, default=2.0)
    parser.add_argument("--attn_impl", default="sdpa",
                        help="Attention implementation (sdpa, flash_attention_2, eager)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Ref model: {args.ref_model}")

    # ── Load tokenizer ─────────────────────────────────────────────────────────
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model (the one we want to score — G1 checkpoint) ─────────────────
    print(f"Loading model from {args.model_path}...")
    model_kwargs = dict(torch_dtype=torch.bfloat16)
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs).to(device)
    model.train()  # enable gradients for parameter grad accumulation
    for param in model.parameters():
        param.requires_grad_(True)

    # ── Load ref model (pretrained target, frozen) ────────────────────────────
    print(f"Loading reference model from {args.ref_model}...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model, torch_dtype=torch.bfloat16
    ).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    # ── Load forget data ───────────────────────────────────────────────────────
    print(f"Loading forget data: {args.forget_dataset} config={args.forget_config} split={args.forget_split}")
    forget_ds = load_dataset(args.forget_dataset, args.forget_config, split=args.forget_split)
    text_col = "text" if "text" in forget_ds.column_names else forget_ds.column_names[0]
    forget_texts = [row[text_col] for row in forget_ds]
    if args.max_forget > 0:
        forget_texts = forget_texts[:args.max_forget]
    print(f"  Forget sequences: {len(forget_texts)}")

    # ── Load retain data ───────────────────────────────────────────────────────
    print(f"Loading retain data: {args.retain_dataset} split={args.retain_split}")
    retain_ds = load_dataset(args.retain_dataset, split=args.retain_split)
    text_col_r = "text" if "text" in retain_ds.column_names else retain_ds.column_names[0]
    retain_texts = [row[text_col_r] for row in retain_ds]
    if args.max_retain > 0:
        retain_texts = retain_texts[:args.max_retain]
    print(f"  Retain sequences: {len(retain_texts)}")

    # ── Tokenize ───────────────────────────────────────────────────────────────
    print("Tokenizing forget data...")
    forget_dataset = TextDataset(forget_texts, tokenizer, args.max_length)
    forget_loader = DataLoader(forget_dataset, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_fn)
    print("Tokenizing retain data...")
    retain_dataset = TextDataset(retain_texts, tokenizer, args.max_length)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_fn)

    print(f"\nForget batches: {len(forget_loader)}, Retain batches: {len(retain_loader)}")

    # ── Accumulate forget gradients (NPO loss) ────────────────────────────────
    print("\n[1/2] Accumulating forget gradients (NPO loss)...")
    npo_fn = lambda batch: compute_npo_loss(model, batch, ref_model, args.npo_beta, device)
    grad_fgt_raw = accumulate_gradients(model, forget_loader, npo_fn, device, desc="forget")

    # ── Accumulate retain gradients (CE loss) ─────────────────────────────────
    print("\n[2/2] Accumulating retain gradients (CE loss)...")
    ce_fn = lambda batch: compute_ce_loss(model, batch, device)
    grad_ret_raw = accumulate_gradients(model, retain_loader, ce_fn, device, desc="retain")

    # ── Reduce to per-neuron (row mean) ───────────────────────────────────────
    print("\nReducing to per-neuron selectivity...")
    grad_fgt = reduce_to_per_neuron(grad_fgt_raw, model)
    grad_ret = reduce_to_per_neuron(grad_ret_raw, model)

    # ── Compute selectivity ───────────────────────────────────────────────────
    selectivity = compute_selectivity(grad_fgt, grad_ret)
    print_stats(selectivity)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(selectivity, args.output_path)
    print(f"\nSaved selectivity scores to: {args.output_path}")

    # Also save full (un-reduced) raw gradients for inspection
    raw_path = args.output_path.replace(".pt", "_raw_grads.pt")
    torch.save({"forget": grad_fgt_raw, "retain": grad_ret_raw}, raw_path)
    print(f"Saved raw gradients to: {raw_path}")


if __name__ == "__main__":
    main()
