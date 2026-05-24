"""Aggregate VILA and LoKU eval results and update tofu.md"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import hmean

EVAL_DIR = Path("/data/open-unlearning/saves/eval")
SEEDS = [42, 123, 456, 789, 1337]
SPLITS = ["forget01", "forget05", "forget10"]

def load_eval(method, split, seed):
    if method == "vila":
        path = EVAL_DIR / f"tofu_vila_3b_{split}_s{seed}" / "TOFU_EVAL.json"
    else:
        path = EVAL_DIR / f"tofu_loku_3b_{split}_s{seed}" / "TOFU_EVAL.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_judge_csv():
    """Load all judge results from the CSV into a dict keyed by (method, split, seed)."""
    import csv
    csv_path = Path("/data/open-unlearning/results/tofu_llm_judge.csv")
    if not csv_path.exists():
        return {}
    results = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            method_name = row["method"]
            if method_name == "VILA":
                method_key = "vila"
            elif method_name == "LoKU":
                method_key = "loku"
            else:
                continue
            key = (method_key, row["split"], int(row["seed"]))
            results[key] = {
                "forget_leakage": float(row["forget_leakage"]),
                "retain_accuracy": float(row["retain_accuracy"]),
                "retain_rq": float(row["retain_rq"]),
                "forget_rq": float(row["forget_rq"]),
            }
    return results

_JUDGE_CACHE = None

def load_judge(method, split, seed):
    global _JUDGE_CACHE
    if _JUDGE_CACHE is None:
        _JUDGE_CACHE = load_judge_csv()
    return _JUDGE_CACHE.get((method, split, seed))

def compute_hm(mu, fgt_prob, fgt_rouge):
    """HM = hmean(MU, 1-fgt_Prob, 1-fgt_ROUGE)"""
    vals = [mu, 1 - fgt_prob, 1 - fgt_rouge]
    if any(v <= 0 for v in vals):
        return 0.0
    return hmean(vals)

def get_metrics(method, split):
    mus, fqs, ess, fgt_probs, fgt_rouges, hms = [], [], [], [], [], []
    for seed in SEEDS:
        data = load_eval(method, split, seed)
        if data is None:
            continue
        mu = data["model_utility"]["agg_value"]
        fq = data["forget_quality"]["agg_value"]
        es = data["extraction_strength"]["agg_value"]
        fgt_prob = data["forget_Q_A_Prob"]["agg_value"]
        fgt_rouge = data["forget_Q_A_ROUGE"]["agg_value"]
        hm_val = compute_hm(mu, fgt_prob, fgt_rouge)

        mus.append(mu)
        fqs.append(fq)
        ess.append(es)
        fgt_probs.append(fgt_prob)
        fgt_rouges.append(fgt_rouge)
        hms.append(hm_val)

    if not mus:
        return None

    return {
        "MU": (np.mean(mus), np.std(mus)),
        "FQ": (np.mean(fqs), np.std(fqs)),
        "ES": (np.mean(ess), np.std(ess)),
        "fgt_Prob": (np.mean(fgt_probs), np.std(fgt_probs)),
        "fgt_ROUGE": (np.mean(fgt_rouges), np.std(fgt_rouges)),
        "HM": (np.mean(hms), np.std(hms)),
        "n_seeds": len(mus),
    }

def get_judge_metrics(method, split):
    fls, ras, ret_rqs, fgt_rqs, hms = [], [], [], [], []
    for seed in SEEDS:
        data = load_judge(method, split, seed)
        if data is None:
            continue
        fl = data["forget_leakage"]
        ra = data["retain_accuracy"]
        ret_rq = data["retain_rq"]
        fgt_rq = data["forget_rq"]

        # HM = hmean(1-FL/2, RA/2, ret_RQ/2)
        vals = [1 - fl/2, ra/2, ret_rq/2]
        if any(v <= 0 for v in vals):
            hm_val = 0.0
        else:
            hm_val = hmean(vals)

        fls.append(fl)
        ras.append(ra)
        ret_rqs.append(ret_rq)
        fgt_rqs.append(fgt_rq)
        hms.append(hm_val)

    if not fls:
        return None

    return {
        "FL": (np.mean(fls), np.std(fls)),
        "RA": (np.mean(ras), np.std(ras)),
        "ret_RQ": (np.mean(ret_rqs), np.std(ret_rqs)),
        "fgt_RQ": (np.mean(fgt_rqs), np.std(fgt_rqs)),
        "HM": (np.mean(hms), np.std(hms)),
        "n_seeds": len(fls),
    }

def fmt(mean, std):
    return f"{mean:.3f}±{std:.3f}"

if __name__ == "__main__":
    print("=" * 60)
    print("TOFU Eval Results — LoRA Models (Llama-3.2-3B-Instruct)")
    print("=" * 60)

    for split in SPLITS:
        print(f"\n### {split}")
        print(f"| Method | MU↑ | FQ↑ | ES↓ | fgt_Prob↓ | fgt_ROUGE↓ | HM↑ |")
        print(f"| --- | --- | --- | --- | --- | --- | --- |")

        for method, name in [("vila", "VILA"), ("loku", "LoKU")]:
            m = get_metrics(method, split)
            if m is None:
                print(f"| {name} | - | - | - | - | - | - |")
                continue
            print(f"| {name} ({m['n_seeds']} seeds) | {fmt(*m['MU'])} | {fmt(*m['FQ'])} | {fmt(*m['ES'])} | {fmt(*m['fgt_Prob'])} | {fmt(*m['fgt_ROUGE'])} | {fmt(*m['HM'])} |")

    print("\n")
    print("=" * 60)
    print("LLM Judge Results")
    print("=" * 60)

    for split in SPLITS:
        print(f"\n### {split}")
        print(f"| Method | FL↓ | RA↑ | ret_RQ↑ | fgt_RQ | HM↑ |")
        print(f"| --- | --- | --- | --- | --- | --- |")

        for method, name in [("vila", "VILA"), ("loku", "LoKU")]:
            m = get_judge_metrics(method, split)
            if m is None:
                print(f"| {name} | - | - | - | - | - |")
                continue
            print(f"| {name} ({m['n_seeds']} seeds) | {fmt(*m['FL'])} | {fmt(*m['RA'])} | {fmt(*m['ret_RQ'])} | {fmt(*m['fgt_RQ'])} | {fmt(*m['HM'])} |")
