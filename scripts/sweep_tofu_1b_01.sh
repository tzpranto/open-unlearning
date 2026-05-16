#!/bin/bash
# BLADE hyperparam sweep — TOFU 1B forget01, seed=42, 8 GPUs parallel
# 5 sweeps × 8 values = 40 runs total, done in 5 batches of 8
# Usage: bash scripts/sweep_tofu_1b_01.sh
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/open-unlearning

SEED=42
EXP_CONFIG="unlearn/tofu/lora_bial_1b.yaml"
MODEL_PATH="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
RETAIN_LOGS="saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json"
REPORT="results/sweep_tofu_1b_01.md"
LOGDIR="saves/unlearn/sweep_tofu_1b_01"
CSV="${LOGDIR}/all_results.csv"

mkdir -p "$LOGDIR"

# ── Preflight checks ───────────────────────────────────────
if [[ ! -f "$RETAIN_LOGS" ]]; then
    echo "[FATAL] Retain logs not found: $RETAIN_LOGS"
    exit 1
fi

if ! python -c "import torch; assert torch.cuda.device_count() >= 8" 2>/dev/null; then
    echo "[FATAL] Need 8 GPUs, found $(python -c 'import torch; print(torch.cuda.device_count())')"
    exit 1
fi

# ── Helpers ─────────────────────────────────────────────────
extract_metrics() {
    local eval_json=$1
    python3 << PYEOF
import json, sys
from statistics import harmonic_mean
try:
    d = json.load(open("$eval_json"))
except Exception as e:
    print("ERROR", file=sys.stderr)
    print("ERROR")
    sys.exit(1)
def val(x):
    return x['agg_value'] if isinstance(x, dict) else x
mu = val(d.get('model_utility', 0))
fp = val(d.get('fgt_Q_A_Prob', d.get('forget_Q_A_Prob', 0)))
fr = val(d.get('fgt_Q_A_ROUGE', d.get('forget_Q_A_ROUGE', 0)))
vals = [mu, 1-fp, 1-fr]
hm = harmonic_mean(vals) if all(v > 0 for v in vals) else 0.0
print(f"{mu:.4f},{fp:.4f},{fr:.4f},{hm:.4f}")
PYEOF
}

is_done() {
    local task_name=$1
    grep -q "^${task_name}," "$CSV" 2>/dev/null
}

run_single() {
    local gpu=$1 task_name=$2 extra_args=$3
    local model_dir="saves/unlearn/${task_name}"

    # Skip entirely if result already recorded
    if is_done "$task_name"; then
        echo "[GPU $gpu] SKIP $task_name (already in CSV)"
        return 0
    fi

    # Skip if eval already exists but not in CSV (re-record)
    if [[ -f "${model_dir}/evals/TOFU_EVAL.json" ]]; then
        echo "[GPU $gpu] SKIP $task_name (eval exists, recording)"
        local conv_step=$(grep -oP 'CONVERGED at step \K[0-9]+' "${model_dir}/train.log" 2>/dev/null | head -1)
        local stop=${conv_step:-250}
        local metrics=$(extract_metrics "${model_dir}/evals/TOFU_EVAL.json")
        if [[ "$metrics" != "ERROR" ]]; then
            echo "${task_name},${metrics},${stop}" >> "$CSV"
        fi
        return 0
    fi

    # Ensure output directory exists before writing logs
    mkdir -p "${model_dir}"

    # Train (skip if model already saved from a prior run)
    if [[ -f "${model_dir}/model.safetensors" ]]; then
        echo "[GPU $gpu] SKIP TRAIN $task_name (model exists)"
    else
        echo "[GPU $gpu] TRAIN $task_name"
        if ! CUDA_VISIBLE_DEVICES=$gpu python src/train.py --config-name=unlearn.yaml \
            experiment=${EXP_CONFIG} \
            task_name="$task_name" \
            model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
            forget_split=forget01 retain_split=retain99 holdout_split=holdout01 \
            trainer.method_args.T=250 \
            trainer.args.seed=${SEED} \
            retain_logs_path="$RETAIN_LOGS" \
            ${extra_args} > "${model_dir}/train.log" 2>&1; then
            echo "[GPU $gpu] TRAIN FAILED $task_name (see ${model_dir}/train.log)"
            return 1
        fi
    fi

    # Verify model was actually saved
    if [[ ! -f "${model_dir}/model.safetensors" ]]; then
        echo "[GPU $gpu] TRAIN FAILED $task_name (no model.safetensors produced)"
        return 1
    fi

    # Eval
    echo "[GPU $gpu] EVAL $task_name"
    if ! CUDA_VISIBLE_DEVICES=$gpu python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=forget01 holdout_split=holdout01 \
        model=Llama-3.2-1B-Instruct \
        task_name="${task_name}" \
        model.model_args.pretrained_model_name_or_path="${model_dir}" \
        paths.output_dir="${model_dir}/evals" \
        retain_logs_path="$RETAIN_LOGS" > "${model_dir}/eval.log" 2>&1; then
        echo "[GPU $gpu] EVAL FAILED $task_name (see ${model_dir}/eval.log)"
        return 1
    fi

    # Verify eval output exists
    local eval_json="${model_dir}/evals/TOFU_EVAL.json"
    if [[ ! -f "$eval_json" ]]; then
        echo "[GPU $gpu] EVAL FAILED $task_name (no TOFU_EVAL.json produced)"
        return 1
    fi

    # Record results
    local conv_step=$(grep -oP 'CONVERGED at step \K[0-9]+' "${model_dir}/train.log" 2>/dev/null | head -1)
    local stop=${conv_step:-250}
    local metrics=$(extract_metrics "$eval_json")
    if [[ "$metrics" == "ERROR" ]]; then
        echo "[GPU $gpu] METRIC EXTRACTION FAILED $task_name"
        return 1
    fi
    echo "${task_name},${metrics},${stop}" >> "$CSV"
    echo "[GPU $gpu] DONE $task_name: MU,Prob,ROUGE,HM=$metrics stop=$stop"

    # Clean model weights (keep evals + train.log + history)
    rm -f "${model_dir}/model.safetensors" "${model_dir}/training_args.bin" \
          "${model_dir}/trainer_state.json" 2>/dev/null
    rm -rf "${model_dir}/converged-best" 2>/dev/null
    return 0
}

update_report() {
    local sweep_name=$1
    shift
    local values=("$@")

    echo "[REPORT] Updating $sweep_name in $REPORT"
    python3 << PYEOF
import re

sweep_name = "$sweep_name"
values = "${values[*]}".split()
report_path = "$REPORT"
csv_path = "$CSV"

# Read all results from CSV
results = {}
try:
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6 and parts[0] != "task_name":
                results[parts[0]] = parts[1:]
except FileNotFoundError:
    pass

# Read report
with open(report_path) as f:
    lines = f.readlines()

# Update matching rows
updated = 0
for i, line in enumerate(lines):
    for val in values:
        task_name = f"sweep_{sweep_name}_{val}"
        if task_name not in results:
            continue
        mu, prob, rouge, hm, stop = results[task_name]
        # Match row starting with "| <val> " that still has empty cells
        if line.strip().startswith(f"| {val} ") and "| |" in line:
            # Count existing pipes to determine column count
            if sweep_name == "eta_in":
                # eta_in table has ratio column
                ratio_map = {"1e-5":"0.2x","2.5e-5":"0.5x","5e-5":"1x","1e-4":"2x","2e-4":"4x","5e-4":"10x","1e-3":"20x","2e-3":"40x"}
                ratio = ratio_map.get(val, "?")
                lines[i] = f"| {val} | {ratio} | {mu} | {prob} | {rouge} | {hm} | {stop} |\n"
            else:
                lines[i] = f"| {val} | {mu} | {prob} | {rouge} | {hm} | {stop} |\n"
            updated += 1
            break

with open(report_path, 'w') as f:
    f.writelines(lines)

print(f"[REPORT] Updated {sweep_name}: {updated}/{len(values)} rows filled")
PYEOF
}

# ── Batch runner: 8 jobs in parallel ────────────────────────
run_batch() {
    local sweep_name=$1
    local param_flag=$2
    shift 2
    local values=("$@")

    echo ""
    echo "================================================================"
    echo " SWEEP: $sweep_name (${#values[@]} runs on 8 GPUs)"
    echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    local pids=()
    local gpu=0

    for val in "${values[@]}"; do
        local task_name="sweep_${sweep_name}_${val}"
        local extra="${param_flag}=${val}"

        run_single $gpu "$task_name" "$extra" &
        pids+=($!)
        gpu=$((gpu + 1))
    done

    # Wait for all and track failures
    local fail=0
    for pid in "${pids[@]}"; do
        wait $pid || fail=$((fail + 1))
    done

    echo "[BATCH] $sweep_name done at $(date '+%H:%M:%S'). Failures: $fail/${#values[@]}"
    update_report "$sweep_name" "${values[@]}"
    echo ""
}

# ── Initialize CSV (append mode — don't overwrite existing results) ──
if [[ ! -f "$CSV" ]]; then
    echo "task_name,MU,fgt_Prob,fgt_ROUGE,HM,stop_step" > "$CSV"
fi

# ══════════════════════════════════════════════════════════════
# SWEEP 1: eps_mul
# ══════════════════════════════════════════════════════════════
run_batch "eps_mul" "trainer.method_args.epsilon_multiplier" \
    0.75 1.0 1.25 1.5 2.0 2.5 3.0 3.2

# ══════════════════════════════════════════════════════════════
# SWEEP 2: tau
# ══════════════════════════════════════════════════════════════
run_batch "tau" "trainer.method_args.clamped_entropy_tau" \
    0.1 0.2 0.3 0.4 0.5 0.6 0.9 1.0

# ══════════════════════════════════════════════════════════════
# SWEEP 3: alpha_dual (dual_decay_factor)
# ══════════════════════════════════════════════════════════════
run_batch "alpha_dual" "trainer.method_args.dual_decay_factor" \
    0.01 0.05 0.1 0.2 0.3 0.5 0.7 1.0

# ══════════════════════════════════════════════════════════════
# SWEEP 4: rho
# ══════════════════════════════════════════════════════════════
run_batch "rho" "trainer.method_args.rho" \
    0.01 0.03 0.05 0.1 0.2 0.5 1.0 2.0

# ══════════════════════════════════════════════════════════════
# SWEEP 5: eta_in (eta_theta fixed at 5e-5)
# ══════════════════════════════════════════════════════════════
run_batch "eta_in" "trainer.method_args.eta_in" \
    1e-5 2.5e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3

echo ""
echo "================================================================"
echo " ALL SWEEPS COMPLETE at $(date '+%Y-%m-%d %H:%M:%S')"
echo " Results: ${CSV}"
echo " Report:  ${REPORT}"
echo "================================================================"
cat "$CSV"
