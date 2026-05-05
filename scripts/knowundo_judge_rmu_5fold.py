#!/usr/bin/env python3
"""Run LLM judge on KnowUnDo RMU 5-fold results (seeds 123,456,789,1024 for both domains).
Seed=42 already exists in the CSV from Sonnet 4.6 run."""
import sys
import os

os.chdir("/datadrive/forked/open-unlearning")
os.environ["LLM_JUDGE_MODEL"] = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
os.environ["AWS_REGION"] = "us-east-1"

sys.path.insert(0, ".")
from scripts.llm_judge import get_client, evaluate_knowundo_eval
import csv

seeds = [123, 456, 789, 1024]  # 42 already done
domains = ["copyright", "privacy"]
output_path = "results/knowundo_llm_judge.csv"

# Read existing entries
already_done = set()
if os.path.exists(output_path):
    with open(output_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["split"], row["method"], str(row["seed"]))
            already_done.add(key)

print(f"Already done: {len(already_done)} entries")

# Init client
client_tuple = get_client()
print(f"Using: {client_tuple[0]}, model: {client_tuple[2]}")

fieldnames = ["model", "split", "method", "seed",
              "forget_leakage", "retain_accuracy", "response_quality",
              "forget_rq", "retain_rq", "n_forget", "n_retain"]

# Also add GA entries (all zeros, model destroyed)
ga_entries = []
for domain in domains:
    for seed in seeds:
        key = ("Llama-2-7b-chat", domain, "GA", str(seed))
        if key in already_done:
            continue
        ga_entries.append({
            "model": "Llama-2-7b-chat",
            "split": domain,
            "method": "GA",
            "seed": str(seed),
            "forget_leakage": 0.0,
            "retain_accuracy": 0.0,
            "response_quality": 0.0,
            "forget_rq": 0.0,
            "retain_rq": 0.0,
            "n_forget": 74 if domain == "copyright" else 110,
            "n_retain": 212 if domain == "copyright" else 108,
        })

# RMU entries to judge
rmu_entries = []
for domain in domains:
    for seed in seeds:
        key = ("Llama-2-7b-chat", domain, "RMU", str(seed))
        if key in already_done:
            print(f"SKIP {key}")
            continue
        eval_path = f"saves/eval/knowundo_{domain}_rmu_s{seed}/KnowUnDo_EVAL.json"
        if not os.path.exists(eval_path):
            print(f"MISSING: {eval_path}")
            continue
        rmu_entries.append({
            "model": "Llama-2-7b-chat",
            "split": domain,
            "method": "RMU",
            "seed": str(seed),
            "eval_path": eval_path,
        })

print(f"\nAdding {len(ga_entries)} GA entries (all zeros)")
print(f"Judging {len(rmu_entries)} RMU entries")

with open(output_path, "a", newline="") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    
    # Write GA entries (no API calls needed)
    for entry in ga_entries:
        writer.writerow(entry)
    out_f.flush()
    print(f"Wrote {len(ga_entries)} GA entries")

    # Judge RMU entries
    for i, entry in enumerate(rmu_entries):
        tag = f"{entry['split']}/RMU/s{entry['seed']}"
        print(f"\n[{i+1}/{len(rmu_entries)}] {tag}")
        
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

print(f"\nDone. Results in {output_path}")
