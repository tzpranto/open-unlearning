"""
Compute re-learning recovery rate R following OpenUnlearning (Dorna et al., 2506.12618).

R = min((m^a_ret - m^b_ret) / (m^a_unl - m^b_unl), 1)

Where m = HM = hmean(MU, 1-fgt_Prob, 1-fgt_ROUGE)
  m^a = before relearning, m^b = after relearning
  Higher R = more robust (model truly forgot)
"""

import json
import glob
import numpy as np
from scipy.stats import hmean
from pathlib import Path

SEED = 42
SAVES = Path("saves")
BACKUP = Path("/data/open-unlearning-h100-backup/saves/unlearn")
RELEARN_EVAL_DIR = SAVES / "eval" / "relearn"

CONFIGS = {
    "1b_01": ("Llama-3.2-1B-Instruct", "forget01"),
    "1b_05": ("Llama-3.2-1B-Instruct", "forget05"),
    "1b_10": ("Llama-3.2-1B-Instruct", "forget10"),
    "3b_01": ("Llama-3.2-3B-Instruct", "forget01"),
    "3b_05": ("Llama-3.2-3B-Instruct", "forget05"),
    "3b_10": ("Llama-3.2-3B-Instruct", "forget10"),
}


def compute_hm(eval_path):
    """Compute HM = hmean(MU, 1-fgt_Prob, 1-fgt_ROUGE) from TOFU_EVAL.json."""
    with open(eval_path) as f:
        d = json.load(f)

    mu = d["model_utility"]["agg_value"] if isinstance(d["model_utility"], dict) else d["model_utility"]
    fgt_prob = d["forget_Q_A_Prob"]["agg_value"] if isinstance(d["forget_Q_A_Prob"], dict) else d["forget_Q_A_Prob"]
    fgt_rouge = d["forget_Q_A_ROUGE"]["agg_value"] if isinstance(d["forget_Q_A_ROUGE"], dict) else d["forget_Q_A_ROUGE"]

    vals = [mu, 1 - fgt_prob, 1 - fgt_rouge]
    if all(v > 0 for v in vals):
        return hmean(vals)
    return 0.0


def get_before_eval(method, key):
    """Get pre-relearning eval path."""
    model, forget = CONFIGS[key]
    if method == "BLADE":
        path = BACKUP / f"adaptive_{model}_{forget}_s{SEED}" / "evals" / "TOFU_EVAL.json"
        return path
    elif method == "PDU":
        path = SAVES / "unlearn" / f"bs32_{model}_{forget}_PDU_s{SEED}" / "evals" / "TOFU_EVAL.json"
        return path
    elif method == "Retain":
        retain = {"01": "retain99", "05": "retain95", "10": "retain90"}[key[-2:]]
        path = SAVES / "eval" / f"tofu_{model}_{retain}" / "TOFU_EVAL.json"
        return path


def get_after_eval(method, key):
    """Get post-relearning eval path."""
    return RELEARN_EVAL_DIR / f"{method}_{key}_s{SEED}" / "TOFU_EVAL.json"


def main():
    print("Re-learning Recovery Rate R = min((m^a_ret - m^b_ret) / (m^a_unl - m^b_unl), 1)")
    print("Metric: HM = hmean(MU, 1-fgt_Prob, 1-fgt_ROUGE)")
    print(f"{'Config':<8} {'Method':<8} {'HM_before':>10} {'HM_after':>10} {'R':>8}")
    print("-" * 50)

    results = {}
    for key in sorted(CONFIGS.keys()):
        retain_before_path = get_before_eval("Retain", key)
        retain_after_path = get_after_eval("Retain", key)

        if not retain_before_path.exists() or not retain_after_path.exists():
            print(f"{key:<8} Retain evals missing, skipping...")
            continue

        m_a_ret = compute_hm(retain_before_path)
        m_b_ret = compute_hm(retain_after_path)

        for method in ["PDU", "BLADE"]:
            before_path = get_before_eval(method, key)
            after_path = get_after_eval(method, key)

            if not before_path.exists():
                print(f"{key:<8} {method:<8} {'N/A':>10} {'—':>10} {'—':>8}  (no before eval)")
                continue
            if not after_path.exists():
                print(f"{key:<8} {method:<8} {'—':>10} {'N/A':>10} {'—':>8}  (no after eval)")
                continue

            m_a_unl = compute_hm(before_path)
            m_b_unl = compute_hm(after_path)

            denom = m_a_unl - m_b_unl
            numer = m_a_ret - m_b_ret

            if abs(denom) < 1e-8:
                R = 1.0  # no change = robust
            else:
                r = numer / denom
                R = min(r, 1.0)

            print(f"{key:<8} {method:<8} {m_a_unl:>10.4f} {m_b_unl:>10.4f} {R:>8.3f}")
            results.setdefault(key, {})[method] = {
                "hm_before": m_a_unl,
                "hm_after": m_b_unl,
                "R": R,
            }

    # Summary table
    print("\n\n=== Summary: Recovery Rate R (higher = more robust) ===")
    print(f"{'Config':<8} {'PDU':>8} {'BLADE':>8}")
    print("-" * 26)
    for key in sorted(results.keys()):
        pdu_r = results[key].get("PDU", {}).get("R", float("nan"))
        blade_r = results[key].get("BLADE", {}).get("R", float("nan"))
        print(f"{key:<8} {pdu_r:>8.3f} {blade_r:>8.3f}")


if __name__ == "__main__":
    main()
