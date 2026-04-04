#!/usr/bin/env python3
"""
Compute ψ(ℓ) layer importance scores for DS-BiAL steering layer selection.

ψ(ℓ) = |activation_diff(ℓ)| × gradient_ratio(ℓ)

where:
  - activation_diff(ℓ) = mean L2 norm of (forget hidden states - retain hidden states) at layer ℓ
  - gradient_ratio(ℓ) = mean(|∇_forget L| / |∇_retain L|) across parameters in layer ℓ

Higher ψ means the layer has both large representation-space divergence between forget/retain
AND is relatively more responsive to forget signal (less retain-dominated).

Layer range selection:
  Two modes, controlled by --layer_range_pct vs --layer_range:
  1. Percentile range (default): --layer_range_pct LO HI selects the candidate window as
     [floor(n_layers * LO/100), floor(n_layers * HI/100)]. Default: 10–25% of depth, which
     maps to layers [3–8] for 32-layer Llama-2-7b. For same-domain tasks where ψ is flat in
     this range, top-3 will be model-architecture-consistent (early-semantic zone).
  2. Absolute range: --layer_range MIN MAX (overrides percentile mode).

  For MUSE News (same-domain, flat ψ in early layers), [5,6,7] is the empirically validated
  choice and is hardcoded as the ds_bial.yaml default. The percentile mode generalises this
  to other model sizes without retuning absolute indices.

Usage:
    python scripts/compute_psi_layers.py --trace_path trace_analysis/figures/traces/muse_news_full/trace_results.pt
    python scripts/compute_psi_layers.py --trace_path trace_results.pt --top_k 3 --layer_range_pct 10 25
    python scripts/compute_psi_layers.py --trace_path trace_results.pt --top_k 3 --layer_range 0 20
"""

import argparse
import json
import re
import sys

import torch


def compute_psi(trace_path: str, top_k: int = 3, layer_range: tuple = None, layer_range_pct: tuple = (10, 25), percentile: float = 75.0):
    """
    Load trace results and compute ψ scores per layer.

    Args:
        trace_path: Path to trace_results.pt produced by trace_activations.py
        top_k: Number of steering layers to select
        layer_range: (min_layer, max_layer) inclusive to consider for selection.
                     Defaults to (0, 20): excludes very late layers that are harder to steer.
        percentile: Not used for selection — kept for CLI compatibility.

    Returns:
        dict with keys: psi_scores, selected_layers, all_scores
    """
    trace = torch.load(trace_path, map_location="cpu")

    layer_differential = trace.get("layer_differential", {})
    differential_scores = trace.get("differential_scores", {})

    # --- Gradient ratio per layer ---
    layer_grad_ratio: dict[int, list[float]] = {}
    for param_name, ratio in differential_scores.items():
        m = re.search(r"layers\.(\d+)\.", param_name)
        if m:
            layer = int(m.group(1))
            layer_grad_ratio.setdefault(layer, []).append(float(ratio))

    # --- Detect number of layers ---
    all_layers = set(layer_grad_ratio.keys())
    for key in layer_differential:
        m = re.match(r"layer_(\d+)$", key)
        if m:
            all_layers.add(int(m.group(1)))
    n_layers = max(all_layers) + 1 if all_layers else 32

    # --- Compute ψ per layer ---
    psi_scores = {}
    for layer in range(n_layers):
        act_diff = abs(layer_differential.get(f"layer_{layer}", 0.0))
        ratios = layer_grad_ratio.get(layer, [1.0])
        grad_ratio = sum(ratios) / len(ratios)
        psi_scores[layer] = act_diff * grad_ratio

    # --- Resolve layer range (absolute overrides percentile) ---
    if layer_range is not None:
        min_l, max_l = layer_range
    else:
        lo_pct, hi_pct = layer_range_pct
        min_l = int(n_layers * lo_pct / 100)
        max_l = int(n_layers * hi_pct / 100)

    # --- Select top-k from layer_range ---
    candidates = {l: v for l, v in psi_scores.items() if min_l <= l <= max_l}
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    selected_layers = sorted([l for l, _ in sorted_candidates[:top_k]])

    return {
        "psi_scores": psi_scores,
        "selected_layers": selected_layers,
        "all_scores": dict(sorted(psi_scores.items())),
        "n_layers": n_layers,
        "layer_range": (min_l, max_l),
        "top_k": top_k,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute ψ-layer importance scores for steering layer selection")
    parser.add_argument("--trace_path", required=True, help="Path to trace_results.pt")
    parser.add_argument("--top_k", type=int, default=3, help="Number of steering layers to select")
    parser.add_argument("--layer_range", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="Absolute layer index range (overrides --layer_range_pct)")
    parser.add_argument("--layer_range_pct", type=float, nargs=2, default=[10.0, 25.0], metavar=("LO", "HI"),
                        help="Candidate window as %% of model depth (default: 10 25 = early-semantic zone). "
                             "For 32-layer models: 10%%=L3, 25%%=L8 → captures [5,6,7] empirical range.")
    parser.add_argument("--json", action="store_true", help="Output JSON only (for scripting)")
    args = parser.parse_args()

    result = compute_psi(
        trace_path=args.trace_path,
        top_k=args.top_k,
        layer_range=tuple(args.layer_range) if args.layer_range is not None else None,
        layer_range_pct=tuple(args.layer_range_pct),
    )

    min_l, max_l = result["layer_range"]

    if args.json:
        print(json.dumps({"steering_layers": result["selected_layers"]}))
        return

    pct_info = f" ({args.layer_range_pct[0]:.0f}–{args.layer_range_pct[1]:.0f}% depth)" if args.layer_range is None else ""
    print(f"\nψ-Layer Importance Scores (range: L{min_l}-L{max_l}{pct_info})")
    print("=" * 55)
    print(f"{'Layer':>6}  {'ψ score':>10}  {'Selected':>10}")
    print("-" * 55)
    for layer in range(result["n_layers"]):
        score = result["psi_scores"][layer]
        in_range = min_l <= layer <= max_l
        is_selected = layer in result["selected_layers"]
        flag = " <-- SELECTED" if is_selected else ""
        dim = "" if in_range else "  (out of range)"
        print(f"  L{layer:2d}  {score:10.5f}{flag}{dim}")

    print("=" * 55)
    print(f"\nSelected steering layers (top-{args.top_k}): {result['selected_layers']}")
    print(f"Model depth: {result['n_layers']} layers, candidate range: L{min_l}–L{max_l}")
    print(f"\nAdd to DS-BiAL config:")
    print(f"  steering_layers: {result['selected_layers']}")


if __name__ == "__main__":
    main()
