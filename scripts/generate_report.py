#!/usr/bin/env python3
"""
Generate comparison tables for unlearning benchmarks.

Usage:
    python scripts/generate_report.py --benchmark muse_news
    python scripts/generate_report.py --benchmark muse_books
    python scripts/generate_report.py --benchmark wmdp
    python scripts/generate_report.py --all              # regenerate all
    python scripts/generate_report.py --benchmark muse_news --out docs/results/muse_news.md
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAVES_UNLEARN = REPO_ROOT / "saves" / "unlearn"
SAVES_EVAL = REPO_ROOT / "saves" / "eval"
REPORTS_DIR = REPO_ROOT / "docs" / "results"

# ─── Benchmark configs ────────────────────────────────────────────────────────

BENCHMARKS = {
    "muse_news": {
        "title": "MUSE News (Llama-2-7b-hf)",
        "model": "Llama-2-7b-hf",
        "data_split": "News",
        "retrain_path": "muse_Llama-2-7b-hf_News_retrain",
        "target_path": "muse_Llama-2-7b-hf_News_target",
        "metrics": ["forget_knowmem_ROUGE", "forget_verbmem_ROUGE",
                    "retain_knowmem_ROUGE", "extraction_strength", "privleak"],
        "metric_labels": ["forget_knowmem↓", "verbmem↓", "retain↑", "extract↓", "privleak"],
        "methods": [
            # (display_name, task_name, eval_subdir)
            # task_name=None means use saves/eval path
            ("Gold (retrain)", None, "muse_Llama-2-7b-hf_News_retrain"),
            ("Target (pre-unlearn)", None, "muse_Llama-2-7b-hf_News_target"),
            ("GradAscent", "muse_Llama-2-7b-hf_News_GradAscent", None),
            ("GradDiff", "muse_Llama-2-7b-hf_News_GradDiff", None),
            ("NPO", "muse_Llama-2-7b-hf_News_NPO", None),
            ("SimNPO", "muse_Llama-2-7b-hf_News_SimNPO", None),
            ("BLURNPO", "muse_Llama-2-7b-hf_News_BLURNPO", None),
            ("RMU", "muse_Llama-2-7b-hf_News_RMU", None),
            ("DS-BiAL (ours)", "muse_Llama-2-7b-hf_News_DSBiAL", None),
            # Exp8r as backup if canonical DS-BiAL not yet present
            ("DS-BiAL Exp8r (ref)", "research_exp8r_fd_beta2", "checkpoint-10"),
        ],
        "output_file": REPORTS_DIR / "muse_news.md",
    },
    "muse_books": {
        "title": "MUSE Books (Llama-2-7b-hf)",
        "model": "Llama-2-7b-hf",
        "data_split": "Books",
        "retrain_path": "muse_Llama-2-7b-hf_Books_retrain",
        "target_path": "muse_Llama-2-7b-hf_Books_target",
        "metrics": ["forget_knowmem_ROUGE", "forget_verbmem_ROUGE",
                    "retain_knowmem_ROUGE", "extraction_strength", "privleak"],
        "metric_labels": ["forget_knowmem↓", "verbmem↓", "retain↑", "extract↓", "privleak"],
        "methods": [
            ("Gold (retrain)", None, "muse_Llama-2-7b-hf_Books_retrain"),
            ("Target (pre-unlearn)", None, "muse_Llama-2-7b-hf_Books_target"),
            ("GradAscent", "muse_Llama-2-7b-hf_Books_GradAscent", None),
            ("GradDiff", "muse_Llama-2-7b-hf_Books_GradDiff", None),
            ("NPO", "muse_Llama-2-7b-hf_Books_NPO", None),
            ("SimNPO", "muse_Llama-2-7b-hf_Books_SimNPO", None),
            ("BLURNPO", "muse_Llama-2-7b-hf_Books_BLURNPO", None),
            ("RMU", "muse_Llama-2-7b-hf_Books_RMU", None),
            ("DS-BiAL (ours)", "muse_Llama-2-7b-hf_Books_DSBiAL", None),
        ],
        "output_file": REPORTS_DIR / "muse_books.md",
    },
    "wmdp": {
        "title": "WMDP (Zephyr-7b-beta)",
        "model": "zephyr-7b-beta",
        "metrics": ["wmdp_cyber", "mmlu_avg"],
        "metric_labels": ["wmdp_cyber↓", "mmlu_avg↑"],
        "methods": [
            ("GradAscent", "wmdp_zephyr-7b-beta_GradAscent", None),
            ("GradDiff", "wmdp_zephyr-7b-beta_GradDiff", None),
            ("NPO", "wmdp_zephyr-7b-beta_NPO", None),
            ("BLURNPO", "wmdp_zephyr-7b-beta_BLURNPO", None),
            ("RMU", "wmdp_zephyr-7b-beta_RMU", None),
            ("DS-BiAL (ours)", "wmdp_zephyr-7b-beta_DSBiAL", None),
        ],
        "output_file": REPORTS_DIR / "wmdp.md",
    },
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_muse_eval(path: Path) -> dict | None:
    """Load MUSE_EVAL.json and return {metric: agg_value}."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return {k: v.get("agg_value", v.get("value")) for k, v in data.items()
                if isinstance(v, dict)}
    except Exception:
        return None


def load_wmdp_eval(path: Path) -> dict | None:
    """Load WMDP_EVAL.json and return {metric: value}."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data  # expected to be flat {metric: value}
    except Exception:
        return None


def get_train_time(task_name: str) -> str | None:
    """
    Get training wall-clock time for a task.
    Priority:
    1. trainer_state.json → last log_history entry with 'train_runtime' (seconds)
    2. SIBL.log → parse first/last timestamp diff
    Returns human-readable string like "68m 24s" or None if unavailable.
    """
    task_dir = SAVES_UNLEARN / task_name

    # Check canonical trainer_state.json (root)
    state_file = task_dir / "trainer_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
            for entry in reversed(state.get("log_history", [])):
                if "train_runtime" in entry:
                    secs = entry["train_runtime"]
                    return _fmt_secs(secs)
        except Exception:
            pass

    # Check *.log files (custom trainers like SIBL, NPO, etc.)
    ts_pattern = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    for log_file in task_dir.glob("*.log"):
        try:
            lines = log_file.read_text().splitlines()
            timestamps = []
            for line in lines:
                m = ts_pattern.match(line)
                if m:
                    timestamps.append(datetime.fromisoformat(m.group(1)))
            if len(timestamps) >= 2:
                delta = (timestamps[-1] - timestamps[0]).total_seconds()
                # Sanity check: skip if > 7 days (likely multi-session log)
                if 0 < delta < 7 * 86400:
                    return _fmt_secs(delta)
        except Exception:
            continue

    # Check checkpoint trainer_state.json
    for ckpt in sorted(task_dir.glob("checkpoint-*/trainer_state.json")):
        try:
            with open(ckpt) as f:
                state = json.load(f)
            for entry in reversed(state.get("log_history", [])):
                if "train_runtime" in entry:
                    return _fmt_secs(entry["train_runtime"])
        except Exception:
            continue

    return None


def _fmt_secs(secs: float) -> str:
    secs = int(secs)
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def find_eval_json(task_name: str, eval_subdir: str | None,
                   is_ref: bool = False, benchmark_type: str = "muse") -> Path | None:
    """Find the MUSE_EVAL.json or WMDP_EVAL.json for a method."""
    eval_file = "MUSE_EVAL.json" if benchmark_type == "muse" else "WMDP_EVAL.json"

    if is_ref:
        # Comes from saves/eval/
        p = SAVES_EVAL / task_name / eval_file
        return p if p.exists() else None

    task_dir = SAVES_UNLEARN / task_name
    if not task_dir.exists():
        return None

    if eval_subdir:
        p = task_dir / eval_subdir / "evals" / eval_file
        if p.exists():
            return p
        p = task_dir / eval_subdir / eval_file
        if p.exists():
            return p

    # Standard location
    p = task_dir / "evals" / eval_file
    if p.exists():
        return p

    # Any checkpoint with evals — use the highest-numbered checkpoint (most complete)
    checkpoints = sorted(
        task_dir.glob("checkpoint-*/evals/" + eval_file),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", str(p)).group(1))
    )
    if checkpoints:
        return checkpoints[-1]

    return None


# ─── Report generators ───────────────────────────────────────────────────────

def generate_muse_report(cfg: dict) -> str:
    """Generate markdown for a MUSE benchmark."""
    title = cfg["title"]
    metrics = cfg["metrics"]
    labels = cfg["metric_labels"]
    methods = cfg["methods"]

    ref_retrain = load_muse_eval(SAVES_EVAL / cfg["retrain_path"] / "MUSE_EVAL.json")

    rows = []
    for (display, task_name, eval_subdir) in methods:
        is_ref = task_name is None
        actual_task = eval_subdir if is_ref else task_name

        eval_path = find_eval_json(
            actual_task, eval_subdir,
            is_ref=is_ref, benchmark_type="muse"
        )
        scores = load_muse_eval(eval_path) if eval_path else None

        train_time = None
        if not is_ref and task_name:
            train_time = get_train_time(task_name)

        rows.append((display, scores, train_time, is_ref))

    # Markdown table
    time_col = "train_time"
    header = "| Method | " + " | ".join(labels) + f" | {time_col} |"
    sep = "| --- | " + " | ".join(["---"] * len(labels)) + " | --- |"

    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Results",
        "",
    ]

    # Add retrain reference values as a note
    if ref_retrain:
        ref_vals = " | ".join(
            f"{ref_retrain.get(m, float('nan')):.4f}" for m in metrics
        )
        lines.append(f"> **Gold (retrain):** {ref_vals}  *(forget_knowmem | verbmem | retain | extract | privleak)*")
        lines.append("")

    lines += [header, sep]

    for display, scores, train_time, is_ref in rows:
        if scores:
            vals = []
            for m in metrics:
                v = scores.get(m)
                if v is None:
                    vals.append("—")
                else:
                    s = f"{v:.4f}"
                    # Bold if close to retrain (within 10% relative) for retain metric
                    if m == "retain_knowmem_ROUGE" and ref_retrain:
                        ref_v = ref_retrain.get(m, 0)
                        if ref_v > 0 and abs(v - ref_v) / ref_v < 0.15:
                            s = f"**{s}**"
                    vals.append(s)
            time_str = "—" if is_ref else (train_time or "⏳")
            row = f"| {display} | " + " | ".join(vals) + f" | {time_str} |"
        else:
            placeholder = " | ".join(["⏳"] * len(metrics))
            time_str = "⏳" if not is_ref else "—"
            row = f"| {display} | {placeholder} | {time_str} |"
        lines.append(row)

    lines += [
        "",
        "## Notes",
        "",
        "- ↓ = lower is better (forgetting quality)",
        "- ↑ = higher is better (retain quality)",
        "- `⏳` = run in queue / not yet evaluated",
        "- **bold retain** = within 15% of gold retrain",
        "- `train_time` = wall-clock training only (excl. eval)",
        f"- Model: Llama-2-7b-hf, Data: {cfg['data_split']}",
    ]

    return "\n".join(lines) + "\n"


def generate_wmdp_report(cfg: dict) -> str:
    """Generate markdown for WMDP benchmark."""
    title = cfg["title"]
    metrics = cfg["metrics"]
    labels = cfg["metric_labels"]
    methods = cfg["methods"]

    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Results",
        "",
    ]

    header = "| Method | " + " | ".join(labels) + " | train_time |"
    sep = "| --- | " + " | ".join(["---"] * len(labels)) + " | --- |"
    lines += [header, sep]

    for (display, task_name, eval_subdir) in methods:
        eval_path = find_eval_json(task_name, eval_subdir, is_ref=False, benchmark_type="wmdp")
        scores = load_wmdp_eval(eval_path) if eval_path else None
        train_time = get_train_time(task_name) if task_name else None

        if scores:
            vals = [f"{scores.get(m, float('nan')):.4f}" for m in metrics]
            row = f"| {display} | " + " | ".join(vals) + f" | {train_time or '—'} |"
        else:
            row = f"| {display} | " + " | ".join(["⏳"] * len(metrics)) + f" | {train_time or '⏳'} |"
        lines.append(row)

    lines += [
        "",
        "## Notes",
        "",
        "- `wmdp_cyber↓`: WMDP-cyber accuracy (lower = more forget of hazardous knowledge)",
        "- `mmlu_avg↑`: MMLU average accuracy (higher = better retain of general knowledge)",
        "- No reference retrain model needed (MCQ-based evaluation)",
        "- `⏳` = run in queue / not yet evaluated",
    ]

    return "\n".join(lines) + "\n"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate unlearning benchmark reports")
    parser.add_argument("--benchmark", choices=list(BENCHMARKS.keys()),
                        help="Which benchmark to generate")
    parser.add_argument("--all", action="store_true", help="Generate all reports")
    parser.add_argument("--out", help="Override output file path")
    parser.add_argument("--print", action="store_true", dest="print_only",
                        help="Print to stdout instead of writing file")
    args = parser.parse_args()

    targets = list(BENCHMARKS.keys()) if args.all else ([args.benchmark] if args.benchmark else [])
    if not targets:
        parser.error("Specify --benchmark or --all")

    for bench_name in targets:
        cfg = BENCHMARKS[bench_name]
        if bench_name == "wmdp":
            content = generate_wmdp_report(cfg)
        else:
            content = generate_muse_report(cfg)

        if args.print_only:
            print(content)
        else:
            out_path = Path(args.out) if args.out else cfg["output_file"]
            out_path.write_text(content)
            print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
