"""Consolidate KnowUnDo results into a single table.
Reads KnowUnDo_SUMMARY.json and LMEval_SUMMARY.json from each method's evals/ dir.
Computes HM = harmonic_mean(1 - forget_ROUGE, retain_ROUGE, mmlu).
Outputs a CSV and prints a markdown table.
"""
import json
import os
import sys
import numpy as np
from scipy.stats import hmean
from pathlib import Path

DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "copyright"
SAVES_DIR = Path("saves/unlearn")

METHODS = ["GA", "GradDiff", "NPO", "SimNPO", "RMU", "BLURNPO", "PDU", "BLADE", "MemFlex"]
SEEDS = [42]

rows = []

# Gold standard: finetuned target model
target_dir = SAVES_DIR / f"knowundo_Llama-2-7b-chat_{DOMAIN}_ft" / "evals"
if target_dir.exists():
    ku_file = target_dir / "KnowUnDo_SUMMARY.json"
    lm_file = target_dir / "LMEval_SUMMARY.json"
    row = {"method": "FT (target)"}
    if ku_file.exists():
        row.update(json.loads(ku_file.read_text()))
    if lm_file.exists():
        lm_data = json.loads(lm_file.read_text())
        if "mmlu/acc" in lm_data:
            row["mmlu"] = lm_data["mmlu/acc"]
    rows.append(row)

for method in METHODS:
    for seed in SEEDS:
        task = f"knowundo_{method}_{DOMAIN}_s{seed}"
        eval_dir = SAVES_DIR / task / "evals"
        if not eval_dir.exists():
            continue
        ku_file = eval_dir / "KnowUnDo_SUMMARY.json"
        lm_file = eval_dir / "LMEval_SUMMARY.json"
        row = {"method": method, "seed": seed}
        if ku_file.exists():
            row.update(json.loads(ku_file.read_text()))
        if lm_file.exists():
            lm_data = json.loads(lm_file.read_text())
            if "mmlu/acc" in lm_data:
                row["mmlu"] = lm_data["mmlu/acc"]
        rows.append(row)

# Compute HM
for row in rows:
    forget_rouge = row.get("forget_ROUGE")
    retain_rouge = row.get("retain_ROUGE")
    mmlu = row.get("mmlu")
    hm_vals = []
    if forget_rouge is not None:
        hm_vals.append(1.0 - forget_rouge)
    if retain_rouge is not None:
        hm_vals.append(retain_rouge)
    if mmlu is not None:
        hm_vals.append(mmlu)
    if len(hm_vals) == 3 and all(v > 0 for v in hm_vals):
        row["HM"] = float(hmean(hm_vals))
    else:
        row["HM"] = None

# Print markdown table
cols = ["method", "forget_Acc", "forget_ROUGE", "retain_Acc", "retain_ROUGE", "mmlu", "HM"]
header = "| " + " | ".join(cols) + " |"
sep = "| " + " | ".join(["---"] * len(cols)) + " |"
print(f"\n## KnowUnDo {DOMAIN} Results\n")
print(header)
print(sep)
for row in rows:
    vals = []
    for c in cols:
        v = row.get(c)
        if v is None:
            vals.append("-")
        elif isinstance(v, float):
            vals.append(f"{v:.4f}")
        else:
            vals.append(str(v))
    print("| " + " | ".join(vals) + " |")

# Save CSV
out_path = f"saves/eval/knowundo_{DOMAIN}_results.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write(",".join(cols) + "\n")
    for row in rows:
        vals = []
        for c in cols:
            v = row.get(c)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        f.write(",".join(vals) + "\n")
print(f"\nSaved to {out_path}")
