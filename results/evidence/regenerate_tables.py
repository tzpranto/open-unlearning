#!/usr/bin/env python3
"""
Regenerate MUSE 5-fold tables from evidence JSON/CSV files.

Usage:
    python results/evidence/regenerate_tables.py

Reads from:
    results/evidence/muse_news/eval_outputs/eval_results.json
    results/evidence/muse_books/eval_outputs/eval_results.json
    results/evidence/muse_news/llm_judge/muse_llm_judge_news_5fold.csv
    results/evidence/muse_books/llm_judge/muse_llm_judge_books_5fold.csv

Outputs: markdown tables matching results/muse_news.md and results/muse_books.md
"""
import json, csv, os, sys
import numpy as np

EVIDENCE_DIR = os.path.dirname(os.path.abspath(__file__))

def hmean3(a, b, c):
    """Harmonic mean of 3 values; returns 0 if any <= 0."""
    if a <= 0 or b <= 0 or c <= 0:
        return 0.0
    return 3.0 / (1.0/a + 1.0/b + 1.0/c)

def compute_eval_table(split):
    """Compute 5-fold eval table for a given split (News/Books)."""
    path = os.path.join(EVIDENCE_DIR, f"muse_{split.lower()}", "eval_outputs", "eval_results.json")
    with open(path) as f:
        data = json.load(f)

    # Group by method
    methods = {}
    for key, vals in data.items():
        method = key.rsplit("_s", 1)[0]
        if method not in methods:
            methods[method] = []
        methods[method].append(vals)

    print(f"\n## {split} 5-fold Eval Results")
    print(f"\nHM = harmonic mean of (1−forget_knowmem, 1−verbmem, retain).\n")
    print("| Method | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | HM↑ |")
    print("| --- | --- | --- | --- | --- | --- |")

    method_order = ["GradAscent", "GradDiff", "NPO", "SimNPO", "RMU", "BLURNPO", "PDU"]
    for method in method_order:
        if method not in methods:
            continue
        entries = methods[method]
        fk = np.array([e.get("forget_knowmem_ROUGE", 0) for e in entries])
        vm = np.array([e.get("forget_verbmem_ROUGE", 0) for e in entries])
        ret = np.array([e.get("retain_knowmem_ROUGE", 0) for e in entries])
        ext = np.array([e.get("extraction_strength", 0) for e in entries])
        hm = np.array([hmean3(1-f, 1-v, r) for f, v, r in zip(fk, vm, ret)])

        print(f"| {method:12s} | {fk.mean():.3f} ± {fk.std():.3f} | "
              f"{vm.mean():.3f} ± {vm.std():.3f} | {ret.mean():.3f} ± {ret.std():.3f} | "
              f"{ext.mean():.3f} ± {ext.std():.3f} | {hm.mean():.3f} ± {hm.std():.3f} |")

def compute_judge_table(split):
    """Compute 5-fold LLM judge table for a given split."""
    path = os.path.join(EVIDENCE_DIR, f"muse_{split.lower()}", "llm_judge",
                        f"muse_llm_judge_{split.lower()}_5fold.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))

    print(f"\n## {split} 5-fold LLM Judge Results")
    print(f"\nHM = hmean(1−FL/2, RA/2, ret_RQ/2).\n")
    print("| Method | FL↓ | RA↑ | ret_RQ↑ | HM↑ |")
    print("| --- | --- | --- | --- | --- |")

    method_order = ["GradAscent", "GradDiff", "NPO", "SimNPO", "RMU", "BLURNPO", "PDU"]
    for method in method_order:
        mrows = [r for r in rows if r["method"] == method]
        if not mrows:
            continue
        fl = np.array([float(r["forget_leakage"]) for r in mrows])
        ra = np.array([float(r["retain_accuracy"]) for r in mrows])
        rq = np.array([float(r["retain_rq"]) for r in mrows])
        hm = np.array([hmean3(1 - f/2, a/2, q/2) for f, a, q in zip(fl, ra, rq)])

        print(f"| {method:10s} | {fl.mean():.2f} ± {fl.std():.2f} | "
              f"{ra.mean():.2f} ± {ra.std():.2f} | {rq.mean():.2f} ± {rq.std():.2f} | "
              f"{hm.mean():.3f} ± {hm.std():.3f} |")

if __name__ == "__main__":
    for split in ["News", "Books"]:
        eval_path = os.path.join(EVIDENCE_DIR, f"muse_{split.lower()}", "eval_outputs", "eval_results.json")
        judge_path = os.path.join(EVIDENCE_DIR, f"muse_{split.lower()}", "llm_judge",
                                  f"muse_llm_judge_{split.lower()}_5fold.csv")
        if os.path.exists(eval_path):
            compute_eval_table(split)
        if os.path.exists(judge_path):
            compute_judge_table(split)
