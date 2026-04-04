#!/usr/bin/env python3
"""
Compute ψ(ℓ) layer importance scores for DS-BiAL steering layer selection.

ψ(ℓ) = |activation_diff(ℓ)| × gradient_ratio(ℓ)

where:
  - activation_diff(ℓ) = mean L2 norm of (forget hidden states - retain hidden states) at layer ℓ
  - gradient_ratio(ℓ) = mean(|∇_forget L| / |∇_retain L|) across parameters in layer ℓ

Higher ψ means the layer has both large representation-space divergence between forget/retain
AND is relatively more responsive to forget signal (less retain-dominated).

Usage:
    python scripts/compute_psi_layers.py --trace_path trace_analysis/figures/traces/muse_news_full/trace_results.pt
    python scripts/compute_psi_layers.py --trace_path trace_results.pt --top_k 3 --layer_range 0 20 --percentile 75
"""

import argparse
import json
import re
import sys

import torch


def compute_psi(trace_path: str, top_k: int = 3, layer_range: tuple = (0, 20), percentile: float = 75.0):
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

    # --- Select top-k from layer_range ---
    min_l, max_l = layer_range
    candidates = {l: v for l, v in psi_scores.items() if min_l <= l <= max_l}
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    selected_layers = sorted([l for l, _ in sorted_candidates[:top_k]])

    return {
        "psi_scores": psi_scores,
        "selected_layers": selected_layers,
        "all_scores": dict(sorted(psi_scores.items())),
        "n_layers": n_layers,
        "layer_range": layer_range,
        "top_k": top_k,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute ψ-layer importance scores for steering layer selection")
    parser.add_argument("--trace_path", required=True, help="Path to trace_results.pt")
    parser.add_argument("--top_k", type=int, default=3, help="Number of steering layers to select")
    parser.add_argument("--layer_range", type=int, nargs=2, default=[0, 20], metavar=("MIN", "MAX"),
                        help="Layer index range (inclusive) to consider for selection (default: 0 20)")
    parser.add_argument("--percentile", type=float, default=75.0,
                        help="Percentile threshold for display (informational only)")
    parser.add_argument("--json", action="store_true", help="Output JSON only (for scripting)")
    args = parser.parse_args()

    result = compute_psi(
        trace_path=args.trace_path,
        top_k=args.top_k,
        layer_range=tuple(args.layer_range),
    )

    if args.json:
        print(json.dumps({"steering_layers": result["selected_layers"]}))
        return

    print(f"\nψ-Layer Importance Scores (range: L{args.layer_range[0]}-L{args.layer_range[1]})")
    print("=" * 55)
    print(f"{'Layer':>6}  {'ψ score':>10}  {'Selected':>10}")
    print("-" * 55)
    for layer in range(result["n_layers"]):
        score = result["psi_scores"][layer]
        in_range = args.layer_range[0] <= layer <= args.layer_range[1]
        is_selected = layer in result["selected_layers"]
        flag = " <-- SELECTED" if is_selected else ""
        dim = "" if in_range else "  (out of range)"
        print(f"  L{layer:2d}  {score:10.5f}{flag}{dim}")

    print("=" * 55)
    print(f"\nSelected steering layers (top-{args.top_k}): {result['selected_layers']}")
    print(f"\nAdd to DS-BiAL config:")
    print(f"  steering_layers: {result['selected_layers']}")


if __name__ == "__main__":
    main()
