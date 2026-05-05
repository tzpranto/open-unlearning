#!/usr/bin/env python3
"""Run LLM judge on all KnowUnDo 5-fold results (GD/SimNPO/PDU/NPO).
Expects eval results in saves/eval/knowundo_{domain}_{method}_s{seed}/KnowUnDo_EVAL.json.
Uses Haiku 3.5 (Opus 4.7 refuses on copyrighted content)."""
import sys
import os
import csv
import time

os.chdir("/datadrive/forked/open-unlearning")
os.environ["LLM_JUDGE_MODEL"] = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
os.environ["AWS_REGION"] = "us-east-1"

sys.path.insert(0, ".")
from scripts.llm_judge import get_client, evaluate_knowundo_eval

SEEDS = [42, 123, 456, 789, 1024]
DOMAINS = ["copyright", "privacy"]
METHODS_MAP = {
    "grad_diff": "GradDiff",
    "simnpo": "SimNPO",
    "pdu": "PDU",
    "npo": "NPO",
}
OUTPUT_PATH = "results/knowundo_llm_judge.csv"

already_done = set()
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["split"], row["method"], str(row["seed"]))
            already_done.add(key)

print(f"Already done: {len(already_done)} entries")

client_tuple = get_client()
print(f"Using: {client_tuple[0]}, model: {client_tuple[2]}")

fieldnames = ["model", "split", "method", "seed",
              "forget_leakage", "retain_accuracy", "response_quality",
              "forget_rq", "retain_rq", "n_forget", "n_retain"]

entries = []
for method_key, method_name in METHODS_MAP.items():
    for domain in DOMAINS:
        for seed in SEEDS:
            key = ("Llama-2-7b-chat", domain, method_name, str(seed))
            if key in already_done:
                continue
            eval_path = f"saves/eval/knowundo_{domain}_{method_key}_s{seed}/KnowUnDo_EVAL.json"
            if not os.path.exists(eval_path):
                continue
            entries.append({
                "model": "Llama-2-7b-chat",
                "split": domain,
                "method": method_name,
                "seed": str(seed),
                "method_key": method_key,
                "eval_path": eval_path,
            })

print(f"\nJudging {len(entries)} entries")
for e in entries:
    print(f"  {e['split']}/{e['method']}/s{e['seed']}")

with open(OUTPUT_PATH, "a", newline="") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)

    for i, entry in enumerate(entries):
        tag = f"{entry['split']}/{entry['method']}/s{entry['seed']}"
        print(f"\n[{i+1}/{len(entries)}] {tag}")

        try:
            scores = evaluate_knowundo_eval(
                client_tuple,
                entry["eval_path"],
                max_workers=8,
            )
            row = {
                "model": entry["model"],
                "split": entry["split"],
                "method": entry["method"],
                "seed": entry["seed"],
                **scores,
            }
            writer.writerow(row)
            out_f.flush()
            print(f"  FL={scores['forget_leakage']:.3f} RA={scores['retain_accuracy']:.3f} RQ={scores['response_quality']:.3f}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            continue

print(f"\nDone. Results in {OUTPUT_PATH}")
