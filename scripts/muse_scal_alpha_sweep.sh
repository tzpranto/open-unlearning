#!/bin/bash
# MUSE News BLADE scalability: alpha_dual sweep (0.5, 1.0) × (f1, f2, f3, f4)
# 8 runs total, 1 GPU each, all in parallel
# Purpose: Show that symmetric dual updates (alpha=1.0) degrade at scale
# Uses same T values as original runs: f1=300, f2=300, f3=400, f4=500
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1
cd /data/open-unlearning

SEED=42
EXP_CONFIG="unlearn/muse/lora_bial_adaptive_books.yaml"
LOGFILE="logs/muse_scal_alpha_sweep.log"
mkdir -p logs

exec > >(tee -a "$LOGFILE") 2>&1
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

# Preflight
if ! python -c "import torch; assert torch.cuda.device_count() >= 8" 2>/dev/null; then
    echo "[FATAL] Need 8 GPUs, found $(python -c 'import torch; print(torch.cuda.device_count())')"
    exit 1
fi

# ── Run configs: (gpu, alpha, fold, split, T) ──
declare -a CONFIGS=(
    "0 1.0 f1 forget_1 300"
    "1 1.0 f2 forget_2 300"
    "2 1.0 f3 forget_3 400"
    "3 1.0 f4 forget_4 500"
    "4 0.5 f1 forget_1 300"
    "5 0.5 f2 forget_2 300"
    "6 0.5 f3 forget_3 400"
    "7 0.5 f4 forget_4 500"
)

run_one() {
    local gpu=$1 alpha=$2 fold=$3 split=$4 T=$5
    local task_name="muse_Llama-2-7b-hf_News_BLADE_scal_${fold}_alpha${alpha}_s${SEED}"
    local model_dir="saves/unlearn/${task_name}"

    mkdir -p "${model_dir}"
    echo "[GPU ${gpu}] START ${task_name} (alpha=${alpha}, split=${split}, T=${T})"

    CUDA_VISIBLE_DEVICES=$gpu python src/train.py --config-name=unlearn.yaml \
        experiment=${EXP_CONFIG} \
        data_split=News \
        model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-News_target \
        data.forget.MUSE_forget.args.hf_args.name=scal \
        data.forget.MUSE_forget.args.hf_args.split=${split} \
        trainer.method_args.T=${T} \
        trainer.method_args.dual_decay_factor=${alpha} \
        trainer.method_args.conv_patience=20 \
        trainer.args.seed=${SEED} \
        task_name="${task_name}" \
        > "${model_dir}/train.log" 2>&1

    if [[ $? -ne 0 ]]; then
        echo "[GPU ${gpu}] FAILED ${task_name} (see ${model_dir}/train.log)"
        return 1
    fi

    echo "[GPU ${gpu}] DONE ${task_name}"
    return 0
}

# Launch all 8 in parallel
pids=()
for cfg in "${CONFIGS[@]}"; do
    read -r gpu alpha fold split T <<< "$cfg"
    run_one $gpu $alpha $fold $split $T &
    pids+=($!)
done

# Wait for all
fail=0
for pid in "${pids[@]}"; do
    wait $pid || fail=$((fail + 1))
done

echo ""
echo "================================================================"
echo " COMPLETE at $(date '+%Y-%m-%d %H:%M:%S'). Failures: ${fail}/8"
echo "================================================================"
echo ""
echo "Results (lora_bial_history.json + evals) in:"
for cfg in "${CONFIGS[@]}"; do
    read -r gpu alpha fold split T <<< "$cfg"
    echo "  saves/unlearn/muse_Llama-2-7b-hf_News_BLADE_scal_${fold}_alpha${alpha}_s${SEED}/"
done
