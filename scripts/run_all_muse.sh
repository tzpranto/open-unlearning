#!/bin/bash
# Master script: runs all MUSE baselines unattended
# 5 seeds x 5 GPUs per batch, methods sequential
# Skips BLUR-NPO News if OOM detected

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/conda_envs/unlearn
export HF_HOME=/data/hf_cache
cd /data/open-unlearning

SEEDS=(42 123 456 789 1024)
COMMON_ARGS="trainer.args.gradient_checkpointing=True trainer.args.eval_on_start=False trainer.args.eval_strategy=no trainer.args.do_eval=False"

# Methods that load a reference model need smaller batch size to fit in 40GB
REF_MODEL_ARGS="trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16"

launch_batch() {
    local method_cfg=$1
    local data_split=$2
    local task_prefix=$3
    shift 3
    local extra_args="$@"

    local logdir="logs/muse_baselines/${data_split}/${task_prefix}"
    mkdir -p "$logdir"

    echo "[$(date '+%H:%M:%S')] Launching ${task_prefix} ${data_split} (5 seeds)..."

    for i in "${!SEEDS[@]}"; do
        local gpu=$i
        local seed=${SEEDS[$i]}
        local task_name="muse_${task_prefix}_${data_split}_s${seed}"
        local logfile="${logdir}/seed${seed}.log"

        CUDA_VISIBLE_DEVICES=$gpu nohup python src/train.py \
            --config-name=unlearn.yaml \
            experiment=unlearn/muse/${method_cfg}.yaml \
            data_split=$data_split \
            task_name=$task_name \
            trainer.args.seed=$seed \
            $COMMON_ARGS \
            $extra_args \
            > "$logfile" 2>&1 &
    done

    echo "[$(date '+%H:%M:%S')] Waiting for ${task_prefix} ${data_split}..."
    wait

    # Check for OOM
    local oom_count=0
    for seed in "${SEEDS[@]}"; do
        if grep -q "OutOfMemoryError" "${logdir}/seed${seed}.log" 2>/dev/null; then
            ((oom_count++))
            echo "[$(date '+%H:%M:%S')] WARNING: ${task_prefix} ${data_split} seed=${seed} OOM!"
        fi
    done

    if [ $oom_count -gt 0 ]; then
        echo "[$(date '+%H:%M:%S')] ${task_prefix} ${data_split}: ${oom_count}/5 seeds OOM'd"
        return 1
    fi

    echo "[$(date '+%H:%M:%S')] ${task_prefix} ${data_split} DONE (all 5 seeds)"
    return 0
}

echo "============================================"
echo "MUSE Baselines — Started $(date)"
echo "============================================"

# ========== NEWS ==========
echo ""
echo ">>> MUSE NEWS <<<"

# GA is already running — wait for it
if pgrep -f "muse_GA_News" > /dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] GA News already running, waiting..."
    while pgrep -f "muse_GA_News" > /dev/null 2>&1; do sleep 30; done
    echo "[$(date '+%H:%M:%S')] GA News finished."
else
    launch_batch "default" "News" "GA"
fi

launch_batch "npo" "News" "NPO" "$REF_MODEL_ARGS"
launch_batch "simnpo" "News" "SimNPO" "$REF_MODEL_ARGS"
launch_batch "rmu" "News" "RMU"
launch_batch "ceu" "News" "CEU"
launch_batch "pdu" "News" "PDU" "$REF_MODEL_ARGS"

# BLUR-NPO News — skip on OOM
echo "[$(date '+%H:%M:%S')] Attempting BLUR-NPO News (may OOM)..."
if ! launch_batch "blurnpo" "News" "BLURNPO" "$REF_MODEL_ARGS"; then
    echo "[$(date '+%H:%M:%S')] BLUR-NPO News skipped due to OOM"
fi

# ========== BOOKS ==========
echo ""
echo ">>> MUSE BOOKS <<<"

launch_batch "default" "Books" "GA"
launch_batch "npo" "Books" "NPO" "$REF_MODEL_ARGS"
launch_batch "simnpo" "Books" "SimNPO" "$REF_MODEL_ARGS"
launch_batch "rmu" "Books" "RMU"
launch_batch "ceu" "Books" "CEU"
launch_batch "pdu" "Books" "PDU" "$REF_MODEL_ARGS trainer.method_args.retain_loss_eps=0.1 trainer.args.num_train_epochs=1"
launch_batch "blurnpo" "Books" "BLURNPO" "$REF_MODEL_ARGS trainer.method_args.beta=0.4 trainer.args.learning_rate=1e-5"

echo ""
echo "============================================"
echo "ALL MUSE BASELINES DONE — $(date)"
echo "============================================"
