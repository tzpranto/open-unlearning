#!/usr/bin/env python3
"""LLM judge on MUSE sust v2 steps 2-4."""
import sys, os
os.chdir("/datadrive/forked/open-unlearning")
os.environ["LLM_JUDGE_MODEL"] = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
os.environ["AWS_REGION"] = "us-east-1"

sys.path.insert(0, ".")
from scripts.llm_judge import get_client, evaluate_muse_eval
import csv

output_path = "results/evidence/muse_news/sust_scal/llm_judge.csv"

# Check existing
already_done = set()
if os.path.exists(output_path):
    with open(output_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["split"], row["method"], row["seed"])
            already_done.add(key)

print(f"Already done: {len(already_done)}")

client_tuple = get_client()
print(f"Using: {client_tuple[0]}, model: {client_tuple[2]}")

fieldnames = ["model", "split", "method", "seed",
              "forget_leakage", "forget_leakage_knowmem", "forget_leakage_verbmem",
              "retain_accuracy", "response_quality",
              "forget_rq", "retain_rq",
              "n_forget_knowmem", "n_forget_verbmem", "n_retain"]

entries = []
for step in [2, 3, 4]:
    key = ("Llama-2-7b-hf", "News", f"PDU_sust_v2_step{step}", "42")
    if key in already_done:
        print(f"SKIP step {step}")
        continue
    eval_path = f"saves/eval/muse_Llama-2-7b-hf_News_PDU_sust_v2_step{step}/MUSE_EVAL.json"
    if not os.path.exists(eval_path):
        print(f"MISSING: {eval_path}")
        continue
    entries.append({
        "model": "Llama-2-7b-hf",
        "split": "News",
        "method": f"PDU_sust_v2_step{step}",
        "seed": "42",
        "eval_path": eval_path,
    })

print(f"Judging {len(entries)} sust v2 entries")

with open(output_path, "a", newline="") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if len(already_done) == 0:
        writer.writeheader()

    for i, entry in enumerate(entries):
        print(f"\n[{i+1}/{len(entries)}] {entry['method']}")
        try:
            scores = evaluate_muse_eval(client_tuple, entry["eval_path"], max_workers=8)
            row = {"model": entry["model"], "split": entry["split"],
                   "method": entry["method"], "seed": entry["seed"], **scores}
            writer.writerow(row)
            out_f.flush()
            print(f"  FL={scores['forget_leakage']:.3f} RA={scores['retain_accuracy']:.3f} ret_RQ={scores['retain_rq']:.3f}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()

print("\nDone.")
