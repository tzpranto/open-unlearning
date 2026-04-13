#!/usr/bin/env python3
"""
Compute ψ(ℓ) layer importance scores for DS-BiAL steering layer selection.

ψ(ℓ) = |activation_diff(ℓ)| × gradient_ratio(ℓ)

where:
  - activation_diff(ℓ) = L2 norm difference between forget and retain hidden states
  - gradient_ratio(ℓ) = mean(|∇_forget L| / |∇_retain L|) across layer parameters

Selection strategy (auto-detect flat vs informative ψ):
  1. Compute CV (coefficient of variation) of ψ, excluding the last 2 layers.
  2. FLAT (CV < cv_threshold, default 1.5) — same-domain task:
       ψ signal is not reliable. Fall back to the empirical early-semantic zone:
       layers in [round(n_layers * flat_lo%), round(n_layers * flat_hi%)].
       Default flat_pct = (15, 22), which maps to layers [5, 7] for 32-layer Llama-2-7b
       → candidate range [5,6,7] → top-3 = [5,6,7] ✓
  3. INFORMATIVE (CV ≥ cv_threshold) — cross-domain task (WMDP, Books, etc.):
       ψ reliably identifies forget-dominant layers. Take ψ-top-k from a broad range
       excluding only the last 2 layers (which are output-dominated).

Usage:
    # Auto mode (recommended):
    python scripts/compute_psi_layers.py --trace_path trace_results.pt

    # Force flat fallback (e.g. to preview what MUSE News would select):
    python scripts/compute_psi_layers.py --trace_path trace_results.pt --force_flat

    # Force informative mode:
    python scripts/compute_psi_layers.py --trace_path trace_results.pt --force_psi

    # JSON output for scripting:
    python scripts/compute_psi_layers.py --trace_path trace_results.pt --json
"""

import argparse
import json
import re

import torch


def compute_psi(
    trace_path: str,
    top_k: int = 3,
    flat_pct: tuple = (15, 22),     # % of n_layers → [5,7] for 32L = [5,6,7]
    cv_threshold: float = 1.5,       # CV below this → treat as flat
    force_flat: bool = False,
    force_psi: bool = False,
):
    """
    Load trace results and compute ψ scores, auto-selecting flat vs informative mode.

    Returns dict with keys:
        psi_scores, selected_layers, n_layers, layer_range, mode, cv, is_flat
    """
    trace = torch.load(trace_path, map_location="cpu", weights_only=False)

    layer_differential = trace.get("layer_differential", {})
    differential_scores = trace.get("differential_scores", {})

    # --- Gradient ratio per layer ---
    layer_grad_ratio: dict[int, list[float]] = {}
    for param_name, ratio in differential_scores.items():
        m = re.search(r"layers\.(\d+)\.", param_name)
        if m:
            layer = int(m.group(1))
            # ratio may be a scalar float or a row-vector tensor (new format)
            v = ratio
            if hasattr(v, "mean"):
                v = v.float().mean().item()
            layer_grad_ratio.setdefault(layer, []).append(float(v))

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

    # --- Detect flat vs informative (exclude last 2 layers to avoid output-dominated bias) ---
    interior = [psi_scores[l] for l in range(n_layers - 2)]
    mean_psi = sum(interior) / len(interior) if interior else 1.0
    std_psi = (sum((v - mean_psi) ** 2 for v in interior) / len(interior)) ** 0.5
    cv = std_psi / mean_psi if mean_psi > 1e-9 else 0.0

    if force_flat:
        is_flat = True
    elif force_psi:
        is_flat = False
    else:
        is_flat = cv < cv_threshold

    # --- Select layer range and top-k ---
    if is_flat:
        # Flat: use empirical early-semantic zone (percentile of model depth)
        lo_pct, hi_pct = flat_pct
        min_l = round(n_layers * lo_pct / 100)
        max_l = round(n_layers * hi_pct / 100)
        mode = "flat_fallback"
    else:
        # Informative: full range excluding last 2 (output-dominated) layers
        min_l = 0
        max_l = n_layers - 3
        mode = "psi_guided"

    candidates = {l: v for l, v in psi_scores.items() if min_l <= l <= max_l}
    sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    selected_layers = sorted([l for l, _ in sorted_cands[:top_k]])

    return {
        "psi_scores": psi_scores,
        "selected_layers": selected_layers,
        "n_layers": n_layers,
        "layer_range": (min_l, max_l),
        "mode": mode,
        "cv": cv,
        "is_flat": is_flat,
        "top_k": top_k,
    }


def main():
    parser = argparse.ArgumentParser(description="ψ-layer selection for DS-BiAL steering")
    parser.add_argument("--trace_path", required=True, help="Path to trace_results.pt")
    parser.add_argument("--top_k", type=int, default=3, help="Number of steering layers")
    parser.add_argument("--flat_pct", type=float, nargs=2, default=[15.0, 22.0],
                        metavar=("LO", "HI"),
                        help="Depth %% range for flat fallback (default: 15 22). "
                             "Maps to L5–L7 for 32-layer Llama-2-7b → [5,6,7].")
    parser.add_argument("--cv_threshold", type=float, default=1.5,
                        help="CV threshold below which ψ is treated as flat (default: 1.5)")
    parser.add_argument("--force_flat", action="store_true", help="Force flat-fallback mode")
    parser.add_argument("--force_psi", action="store_true", help="Force ψ-guided mode")
    parser.add_argument("--json", action="store_true", help="Output JSON for scripting")
    args = parser.parse_args()

    result = compute_psi(
        trace_path=args.trace_path,
        top_k=args.top_k,
        flat_pct=tuple(args.flat_pct),
        cv_threshold=args.cv_threshold,
        force_flat=args.force_flat,
        force_psi=args.force_psi,
    )

    if args.json:
        print(json.dumps({
            "steering_layers": result["selected_layers"],
            "mode": result["mode"],
            "cv": round(result["cv"], 3),
        }))
        return

    min_l, max_l = result["layer_range"]
    mode_str = f"FLAT fallback ({args.flat_pct[0]:.0f}–{args.flat_pct[1]:.0f}% depth)" \
        if result["is_flat"] else "ψ-GUIDED (informative domain)"

    print(f"\nψ-Layer Importance Scores")
    print(f"  n_layers={result['n_layers']}, CV={result['cv']:.3f} "
          f"({'< ' if result['is_flat'] else '>= '}{args.cv_threshold} → {mode_str})")
    print(f"  Candidate range: L{min_l}–L{max_l}")
    print("=" * 60)
    print(f"{'Layer':>6}  {'ψ score':>10}  {'':>12}")
    print("-" * 60)
    for layer in range(result["n_layers"]):
        score = result["psi_scores"][layer]
        in_range = min_l <= layer <= max_l
        flag = " <-- SELECTED" if layer in result["selected_layers"] else ""
        dim = "" if in_range else "  (out of range)"
        print(f"  L{layer:2d}  {score:10.5f}{flag}{dim}")

    print("=" * 60)
    print(f"\nMode: {mode_str}")
    print(f"Selected steering layers: {result['selected_layers']}")
    print(f"\nAdd to DS-BiAL config:")
    print(f"  steering_layers: {result['selected_layers']}")


if __name__ == "__main__":
    main()
