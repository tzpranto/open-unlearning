#!/usr/bin/env python3
"""Analyze LoRA-BiAL training history JSON to diagnose training dynamics."""

import json
import sys
import os

def analyze(path):
    with open(path) as f:
        history = json.load(f)

    if not history:
        print("Empty history!")
        return

    n = len(history)
    epochs = history[-1].get("epoch", 0) + 1

    print(f"Training history: {n} steps, {epochs} epoch(s)")
    print(f"{'='*70}")

    # Find NPO saturation point
    sat_threshold = 0.01
    sat_step = None
    for i, h in enumerate(history):
        if h["L_fgt"] < sat_threshold:
            if sat_step is None:
                sat_step = i
        else:
            sat_step = None  # reset if not consecutive

    if sat_step is not None:
        print(f"NPO saturation: L_fgt < {sat_threshold} first at step {sat_step} "
              f"({sat_step/n*100:.1f}% through training)")
    else:
        print(f"NPO did NOT saturate (L_fgt never stayed below {sat_threshold})")

    # Loss trajectories in phases
    phase_size = max(1, n // 5)
    print(f"\n{'Phase':<10} {'Steps':<12} {'L_fgt':<10} {'L_ret':<10} {'lambda':<10} {'inner':<10}")
    print("-" * 62)
    for i in range(0, n, phase_size):
        end = min(i + phase_size, n)
        phase = history[i:end]
        fgt = sum(h["L_fgt"] for h in phase) / len(phase)
        ret = sum(h["L_ret"] for h in phase) / len(phase)
        lam = phase[-1]["lambda"]
        inner = sum(h["inner_loss_mean"] for h in phase) / len(phase)
        print(f"{'['+str(i)+'-'+str(end)+']':<10} {len(phase):<12} {fgt:<10.4f} {ret:<10.4f} {lam:<10.3f} {inner:<10.4f}")

    # Key diagnostics
    print(f"\n{'='*70}")
    print("Diagnostics:")

    # Lambda growth (ALM dual variable)
    final_lambda = history[-1]["lambda"]
    print(f"  lambda: {history[0]['lambda']:.3f} -> {final_lambda:.3f} "
          f"({'stable' if final_lambda < 5.0 else 'GROWING - potential issue'})")

    # Retain loss trend
    first_ret = sum(h["L_ret"] for h in history[:5]) / min(5, n)
    last_ret = sum(h["L_ret"] for h in history[-5:]) / min(5, n)
    ret_change = last_ret - first_ret
    print(f"  L_ret: {first_ret:.4f} -> {last_ret:.4f} "
          f"({'improving' if ret_change < -0.01 else 'degrading' if ret_change > 0.01 else 'stable'})")

    # Forget loss at end
    last_fgt = sum(h["L_fgt"] for h in history[-5:]) / min(5, n)
    print(f"  L_fgt (last 5): {last_fgt:.4f} "
          f"({'saturated' if last_fgt < 0.01 else 'active'})")

    # Training time
    total_dt = sum(h["dt"] for h in history)
    print(f"  Total time: {total_dt:.0f}s ({total_dt/60:.1f}m)")

    # Recommendation
    print(f"\n{'='*70}")
    print("Assessment:")
    if sat_step is not None and sat_step < n * 0.2:
        pct = sat_step / n * 100
        print(f"  WARNING: NPO saturated at {pct:.0f}% — {100-pct:.0f}% of outer steps")
        print(f"  were wasted/harmful. Recommend: retain_only_after_saturation=True")
        print(f"  or reduce outer steps to ~{sat_step + 10}")
    if final_lambda > 10.0:
        print(f"  WARNING: lambda grew to {final_lambda:.1f} — ALM is struggling.")
        print(f"  Retain loss constraint is consistently violated.")
    if last_ret > first_ret + 0.1:
        print(f"  WARNING: Retain loss INCREASED during training ({first_ret:.3f} -> {last_ret:.3f})")
        print(f"  Model is losing retain quality. Need LR decay or fewer steps.")
    if last_fgt < 0.01 and last_ret < first_ret:
        print(f"  GOOD: NPO saturated AND retain improved. Method working as intended.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Auto-find most recent history file
        saves = "saves/unlearn"
        histories = []
        for d in os.listdir(saves):
            h = os.path.join(saves, d, "lora_bial_history.json")
            if os.path.exists(h):
                histories.append(h)
        if histories:
            histories.sort(key=os.path.getmtime, reverse=True)
            path = histories[0]
            print(f"Auto-found: {path}\n")
        else:
            print("Usage: python scripts/analyze_history.py <path_to_history.json>")
            sys.exit(1)
    else:
        path = sys.argv[1]

    analyze(path)
