#!/bin/bash
# Continue MUSE baselines from NPO News onward (GA News already done)
# Methods with ref model use batch_size=2, grad_accum=16 to fit 40GB

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/conda_envs/unlearn
export HF_HOME=/data/hf_cache
cd /data/open-unlearning

SEEDS=(42 123 456 789 1024)
COMMON_ARGS="trainer.args.gradient_checkpointing=True trainer.args.eval_on_start=False trainer.args.eval_strategy=no trainer.args.do_eval=False"
REF_MODEL_ARGS="trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16"

launch_batch() {
    local method_cfg=$1
    local data_split=$2
    local task_prefix=$3
    shift 3
    local extra_args="$@"

    local logdir="logs/muse_baselines/${data_split}/${task_prefix}"
    mkdir -p "$logdir"

    # Skip if already completed (check if model saved)
    local first_task="muse_${task_prefix}_${data_split}_s${SEEDS[0]}"
    if [ -f "saves/unlearn/${first_task}/model.safetensors" ] || [ -d "saves/unlearn/${first_task}/model" ]; then
        echo "[$(date '+%H:%M:%S')] ${task_prefix} ${data_split} already completed, skipping."
        return 0
    fi

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
echo "MUSE Baselines (continued) — Started $(date)"
echo "============================================"

# ========== NEWS (GA already done) ==========
echo ""
echo ">>> MUSE NEWS <<<"

launch_batch "npo" "News" "NPO" $REF_MODEL_ARGS || true
launch_batch "simnpo" "News" "SimNPO" $REF_MODEL_ARGS || true
launch_batch "rmu" "News" "RMU" || true
launch_batch "ceu" "News" "CEU" || true
launch_batch "pdu" "News" "PDU" $REF_MODEL_ARGS || true

# BLUR-NPO News — skip on OOM per user instruction
echo "[$(date '+%H:%M:%S')] Attempting BLUR-NPO News (may OOM)..."
if ! launch_batch "blurnpo" "News" "BLURNPO" $REF_MODEL_ARGS; then
    echo "[$(date '+%H:%M:%S')] BLUR-NPO News skipped due to OOM"
fi

# ========== BOOKS ==========
echo ""
echo ">>> MUSE BOOKS <<<"

launch_batch "default" "Books" "GA" || true
launch_batch "npo" "Books" "NPO" $REF_MODEL_ARGS || true
launch_batch "simnpo" "Books" "SimNPO" $REF_MODEL_ARGS || true
launch_batch "rmu" "Books" "RMU" || true
launch_batch "ceu" "Books" "CEU" || true
launch_batch "pdu" "Books" "PDU" $REF_MODEL_ARGS trainer.method_args.retain_loss_eps=0.1 trainer.args.num_train_epochs=1 || true
launch_batch "blurnpo" "Books" "BLURNPO" $REF_MODEL_ARGS trainer.method_args.beta=0.4 trainer.args.learning_rate=1e-5 || true

echo ""
echo "============================================"
echo "ALL MUSE BASELINES DONE — $(date)"
echo "============================================"
