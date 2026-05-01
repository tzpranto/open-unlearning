#!/bin/bash
# MUSE baselines — 5 seeds in parallel on 5 GPUs
# Matches params from scripts/muse_baselines.sh exactly
# Usage: bash scripts/run_muse_multi_gpu.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/conda_envs/unlearn
export HF_HOME=/data/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/open-unlearning

DATA_SPLIT="${1:-News}"
MODEL="Llama-2-7b-hf"
SEEDS=(42 123 456 789 1024)
BSZ=4
ACCUM=8
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"

# Split-specific overrides
if [[ "$DATA_SPLIT" == "News" ]]; then
    PDU_EPS=1.5
    BLUR_LR="2.5e-5"
    BLUR_BETA=0.05
elif [[ "$DATA_SPLIT" == "Books" ]]; then
    PDU_EPS=0.1
    BLUR_LR="1e-5"
    BLUR_BETA=0.4
fi

COMMON_TRAIN="trainer.args.gradient_checkpointing=true trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false"

launch_5seeds() {
    local method_label=$1
    shift
    local extra_args=("$@")

    local logdir="logs/muse_baselines/${DATA_SPLIT}/${method_label}"
    mkdir -p "$logdir"

    # Skip if first seed already has MUSE_SUMMARY
    local first_task="muse_${MODEL}_${DATA_SPLIT}_${method_label}_s${SEEDS[0]}"
    if [[ -f "saves/unlearn/${first_task}/evals/MUSE_SUMMARY.json" ]]; then
        echo "[$(date '+%H:%M:%S')] SKIP ${method_label} ${DATA_SPLIT} — already completed"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] Launching ${method_label} ${DATA_SPLIT} (5 seeds on GPUs 0-4)..."

    for i in "${!SEEDS[@]}"; do
        local gpu=$i
        local seed=${SEEDS[$i]}
        local task="muse_${MODEL}_${DATA_SPLIT}_${method_label}_s${seed}"
        local logfile="${logdir}/seed${seed}.log"

        CUDA_VISIBLE_DEVICES=$gpu nohup python src/train.py \
            --config-name=unlearn.yaml \
            "${extra_args[@]}" \
            trainer.args.seed=$seed \
            task_name=$task \
            $COMMON_TRAIN \
            > "$logfile" 2>&1 &
    done

    echo "[$(date '+%H:%M:%S')] Waiting for ${method_label} ${DATA_SPLIT}..."
    wait

    # Check for OOM / errors
    local oom=0; local err=0
    for seed in "${SEEDS[@]}"; do
        local lf="${logdir}/seed${seed}.log"
        if grep -q "OutOfMemoryError" "$lf" 2>/dev/null; then ((oom++)); fi
        if grep -q "Error\|Traceback" "$lf" 2>/dev/null && ! grep -q "FutureWarning" "$lf" 2>/dev/null; then
            # Check if it actually failed (no "train" completion)
            if ! grep -q "Training completed" "$lf" 2>/dev/null && ! grep -q "100%" "$lf" 2>/dev/null; then
                ((err++))
            fi
        fi
    done

    if [ $oom -gt 0 ]; then
        echo "[$(date '+%H:%M:%S')] ${method_label}: ${oom}/5 OOM'd — SKIPPING"
        return 1
    fi
    echo "[$(date '+%H:%M:%S')] ${method_label} ${DATA_SPLIT} DONE"
    return 0
}

echo "============================================"
echo "MUSE ${DATA_SPLIT} Baselines — $(date)"
echo "============================================"

# --- GradAscent ---
launch_5seeds "GradAscent" \
    experiment=unlearn/muse/default.yaml \
    model=${MODEL} \
    data_split=${DATA_SPLIT} \
    retain_logs_path=${RETAIN_LOGS} \
    trainer.args.per_device_train_batch_size=${BSZ} \
    trainer.args.gradient_accumulation_steps=${ACCUM} \
    || true

# --- GradDiff (has ref_model → bsz=2, accum=16) ---
launch_5seeds "GradDiff" \
    experiment=unlearn/muse/default.yaml \
    model=${MODEL} \
    data_split=${DATA_SPLIT} \
    retain_logs_path=${RETAIN_LOGS} \
    trainer=GradDiff \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=16 \
    || true

# --- NPO (has ref_model → bsz=2, accum=16) ---
launch_5seeds "NPO" \
    experiment=unlearn/muse/default.yaml \
    model=${MODEL} \
    data_split=${DATA_SPLIT} \
    retain_logs_path=${RETAIN_LOGS} \
    trainer=NPO \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=16 \
    || true

# --- SimNPO (has ref_model → bsz=2, accum=16) ---
launch_5seeds "SimNPO" \
    experiment=unlearn/muse/default.yaml \
    model=${MODEL} \
    data_split=${DATA_SPLIT} \
    retain_logs_path=${RETAIN_LOGS} \
    trainer=SimNPO \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=16 \
    || true

# --- RMU (no ref_model, dedicated config) ---
launch_5seeds "RMU" \
    experiment=unlearn/muse/rmu.yaml \
    data_split=${DATA_SPLIT} \
    retain_logs_path=${RETAIN_LOGS} \
    || true

# --- PDU (ref_model → bsz=2, accum=16, dedicated config) ---
launch_5seeds "PDU" \
    experiment=unlearn/muse/pdu.yaml \
    data_split=${DATA_SPLIT} \
    retain_logs_path=${RETAIN_LOGS} \
    trainer.method_args.retain_loss_eps=${PDU_EPS} \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=16 \
    || true

# --- BLUR-NPO (ref_model → bsz=2, accum=16, dedicated config) ---
echo "[$(date '+%H:%M:%S')] Attempting BLUR-NPO (may OOM on News)..."
launch_5seeds "BLURNPO" \
    experiment=unlearn/muse/blurnpo_muse.yaml \
    data_split=${DATA_SPLIT} \
    retain_logs_path=${RETAIN_LOGS} \
    trainer.method_args.beta=${BLUR_BETA} \
    trainer.args.learning_rate=${BLUR_LR} \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=16 \
    || true

echo ""
echo "============================================"
echo "ALL ${DATA_SPLIT} BASELINES DONE — $(date)"
echo "============================================"
