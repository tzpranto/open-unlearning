#!/usr/bin/env python
"""
Analyze SIBL implicit debug artifacts emitted by sibl.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import numpy as np


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = {}
    residuals = []
    ratios = []
    for row in rows:
        status = row.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        metrics = row.get("metrics", {})
        if "linear_residual" in metrics and metrics["linear_residual"] is not None:
            residuals.append(float(metrics["linear_residual"]))
        cond = metrics.get("condition_proxy", {})
        ratio = cond.get("rayleigh_ratio")
        if ratio is not None and np.isfinite(ratio):
            ratios.append(float(ratio))
    out: Dict[str, Any] = {
        "num_rows": len(rows),
        "status_counts": statuses,
        "residual_mean": float(np.mean(residuals)) if residuals else None,
        "residual_max": float(np.max(residuals)) if residuals else None,
        "rayleigh_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "rayleigh_ratio_max": float(np.max(ratios)) if ratios else None,
    }
    return out


def write_markdown(path: str, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    lines = [
        "# Implicit Debug Analysis",
        "",
        f"- Rows: {summary['num_rows']}",
        f"- Status counts: `{summary['status_counts']}`",
        f"- Mean linear residual: `{summary['residual_mean']}`",
        f"- Max linear residual: `{summary['residual_max']}`",
        f"- Mean rayleigh ratio: `{summary['rayleigh_ratio_mean']}`",
        f"- Max rayleigh ratio: `{summary['rayleigh_ratio_max']}`",
        "",
        "## Per-step table",
        "",
        "| outer_iter | solver | variant | status | linear_residual | rayleigh_ratio | nonpos_count |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row.get("metrics", {})
        cond = metrics.get("condition_proxy", {})
        lines.append(
            f"| {row.get('outer_iter')} | {row.get('solver')} | {row.get('variant')} | "
            f"{row.get('status')} | {metrics.get('linear_residual')} | "
            f"{cond.get('rayleigh_ratio')} | {cond.get('nonpos_count')} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def list_arrays(debug_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(debug_dir, "outer_*_*.npy")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-dir", type=str, required=True)
    args = parser.parse_args()

    jsonl_path = os.path.join(args.debug_dir, "implicit_debug.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Missing {jsonl_path}")

    rows = load_jsonl(jsonl_path)
    summary = summarize(rows)
    summary["array_files"] = list_arrays(args.debug_dir)
    summary["array_file_count"] = len(summary["array_files"])

    out_json = os.path.join(args.debug_dir, "implicit_analysis.json")
    out_md = os.path.join(args.debug_dir, "implicit_analysis.md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_markdown(out_md, rows, summary)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Detected {summary['array_file_count']} numpy array files")


if __name__ == "__main__":
    main()
