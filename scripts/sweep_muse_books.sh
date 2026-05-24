#!/bin/bash
# BLADE hyperparam sweep — MUSE Books, seed=42, 8 GPUs parallel
# 5 sweeps × 8 values = 40 runs total (K=3)
# Usage: bash scripts/sweep_muse_books.sh
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1
cd /data/open-unlearning

SEED=42
EXP_CONFIG="unlearn/muse/lora_bial_adaptive_books.yaml"
LOGDIR="saves/unlearn/sweep_muse_books"
CSV="${LOGDIR}/all_results.csv"

mkdir -p "$LOGDIR"

# ── Preflight checks ───────────────────────────────────────
if ! python -c "import torch; assert torch.cuda.device_count() >= 8" 2>/dev/null; then
    echo "[FATAL] Need 8 GPUs, found $(python -c 'import torch; print(torch.cuda.device_count())')"
    exit 1
fi

python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('muse-bench/MUSE-Books_target')" 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo "[FATAL] Target model muse-bench/MUSE-Books_target not accessible"
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
fk = d.get('forget_knowmem_ROUGE', 0)
vm = d.get('forget_verbmem_ROUGE', 0)
rk = d.get('retain_knowmem_ROUGE', 0)
es = d.get('extraction_strength', 0)
pl = d.get('privleak', 0)
vals = [1 - fk, 1 - vm, rk]
hm = harmonic_mean(vals) if all(v > 0 for v in vals) else 0.0
print(f"{fk:.4f},{vm:.4f},{rk:.4f},{es:.4f},{pl:.2f},{hm:.4f}")
PYEOF
}

is_done() {
    local task_name=$1
    grep -q "^${task_name}," "$CSV" 2>/dev/null
}

model_exists() {
    local dir=$1
    [[ -f "${dir}/model.safetensors" ]] || [[ -f "${dir}/model.safetensors.index.json" ]]
}

find_eval_json() {
    local dir=$1
    local candidates=(
        "${dir}/checkpoint-0/evals/MUSE_SUMMARY.json"
        "${dir}/evals/MUSE_SUMMARY.json"
    )
    for f in "${candidates[@]}"; do
        if [[ -f "$f" ]]; then
            echo "$f"
            return 0
        fi
    done
    return 1
}

run_single() {
    local gpu=$1 task_name=$2 extra_args=$3
    local model_dir="saves/unlearn/${task_name}"

    if is_done "$task_name"; then
        echo "[GPU $gpu] SKIP $task_name (already in CSV)"
        return 0
    fi

    local existing_eval=$(find_eval_json "$model_dir")
    if [[ -n "$existing_eval" ]]; then
        echo "[GPU $gpu] SKIP $task_name (eval exists, recording)"
        local conv_step=$(grep -oP 'CONVERGED at step \K[0-9]+' "${model_dir}/train.log" 2>/dev/null | head -1)
        local stop=${conv_step:-250}
        local metrics=$(extract_metrics "$existing_eval")
        if [[ "$metrics" != "ERROR" ]]; then
            echo "${task_name},${metrics},${stop}" >> "$CSV"
        fi
        return 0
    fi

    mkdir -p "${model_dir}"

    if model_exists "$model_dir"; then
        echo "[GPU $gpu] SKIP TRAIN $task_name (model exists)"
    else
        echo "[GPU $gpu] TRAIN $task_name"
        if ! CUDA_VISIBLE_DEVICES=$gpu python src/train.py --config-name=unlearn.yaml \
            experiment=${EXP_CONFIG} \
            task_name="$task_name" \
            trainer.args.seed=${SEED} \
            ${extra_args} > "${model_dir}/train.log" 2>&1; then
            echo "[GPU $gpu] TRAIN FAILED $task_name (see ${model_dir}/train.log)"
            return 1
        fi
    fi

    if ! model_exists "$model_dir"; then
        echo "[GPU $gpu] TRAIN FAILED $task_name (no model saved)"
        return 1
    fi

    local eval_json=$(find_eval_json "$model_dir")
    if [[ -z "$eval_json" ]]; then
        echo "[GPU $gpu] EVAL $task_name"
        if ! CUDA_VISIBLE_DEVICES=$gpu python src/eval.py --config-name=eval.yaml \
            model=Llama-2-7b-hf \
            model.model_args.pretrained_model_name_or_path="${model_dir}" \
            eval=muse \
            eval.muse.data_split=Books \
            eval.muse.output_dir="${model_dir}/evals" \
            eval.muse.overwrite=true \
            task_name="${task_name}" \
            seed=${SEED} > "${model_dir}/eval.log" 2>&1; then
            echo "[GPU $gpu] EVAL FAILED $task_name (see ${model_dir}/eval.log)"
            return 1
        fi
        eval_json=$(find_eval_json "$model_dir")
        if [[ -z "$eval_json" ]]; then
            echo "[GPU $gpu] EVAL FAILED $task_name (no MUSE_SUMMARY.json)"
            return 1
        fi
    fi

    local conv_step=$(grep -oP 'CONVERGED at step \K[0-9]+' "${model_dir}/train.log" 2>/dev/null | head -1)
    local stop=${conv_step:-250}
    local metrics=$(extract_metrics "$eval_json")
    if [[ "$metrics" == "ERROR" ]]; then
        echo "[GPU $gpu] METRIC EXTRACTION FAILED $task_name"
        return 1
    fi
    echo "${task_name},${metrics},${stop}" >> "$CSV"
    echo "[GPU $gpu] DONE $task_name: $metrics stop=$stop"

    rm -f "${model_dir}"/model*.safetensors "${model_dir}/model.safetensors.index.json" \
          "${model_dir}/training_args.bin" "${model_dir}/trainer_state.json" 2>/dev/null
    rm -rf "${model_dir}/converged-best" 2>/dev/null
    return 0
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
        local task_name="${PREFIX}_sweep_${sweep_name}_${val}"
        local extra="${param_flag}=${val}"

        run_single $gpu "$task_name" "$extra" &
        pids+=($!)
        gpu=$((gpu + 1))
    done

    local fail=0
    for pid in "${pids[@]}"; do
        wait $pid || fail=$((fail + 1))
    done

    echo "[BATCH] $sweep_name done at $(date '+%H:%M:%S'). Failures: $fail/${#values[@]}"
    echo ""
}

run_all_sweeps() {
    run_batch "eps_mul" "trainer.method_args.epsilon_multiplier" \
        0.75 1.0 1.25 1.5 2.0 2.5 3.0 3.2

    run_batch "tau" "trainer.method_args.clamped_entropy_tau" \
        0.1 0.2 0.3 0.4 0.5 0.6 0.9 1.0

    run_batch "alpha_dual" "trainer.method_args.dual_decay_factor" \
        0.01 0.05 0.1 0.2 0.3 0.5 0.7 1.0

    run_batch "rho" "trainer.method_args.rho" \
        0.01 0.03 0.05 0.1 0.2 0.5 1.0 2.0

    run_batch "eta_in" "trainer.method_args.eta_in" \
        1e-5 2.5e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3
}

# ── Initialize CSV ──────────────────────────────────────────
if [[ ! -f "$CSV" ]]; then
    echo "task_name,fgt_know,fgt_verb,ret_know,extract,privleak,HM,stop_step" > "$CSV"
fi

# ══════════════════════════════════════════════════════════════
# K=3 (40 runs)
# ══════════════════════════════════════════════════════════════
echo ""
echo "################################################################"
echo " MUSE Books sweep: K=3 (40 runs)"
echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"
PREFIX="books_k3"
run_all_sweeps

echo ""
echo "================================================================"
echo " ALL SWEEPS COMPLETE at $(date '+%Y-%m-%d %H:%M:%S')"
echo " Results: ${CSV}"
echo "================================================================"
cat "$CSV"
