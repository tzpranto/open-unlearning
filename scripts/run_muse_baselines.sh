#!/bin/bash
# Run MUSE baselines with 5 seeds on separate GPUs
# Usage: bash scripts/run_muse_baselines.sh [news|books|all]
# Each method x seed gets its own GPU via CUDA_VISIBLE_DEVICES

set -e
cd /data/open-unlearning

SPLIT="${1:-all}"
SEEDS=(42 123 456 789 1024)

# Accelerate config for single GPU (no deepspeed needed for baselines)
ACCEL_CFG="configs/accelerate/single_gpu_config.yaml"

launch_job() {
    local gpu=$1
    local method=$2
    local data_split=$3
    local seed=$4
    local task_name=$5
    shift 5
    local extra_args="$@"

    local logdir="logs/muse_baselines/${data_split}/${method}"
    mkdir -p "$logdir"
    local logfile="${logdir}/seed${seed}.log"

    echo "[GPU $gpu] Launching $method | $data_split | seed=$seed -> $logfile"

    CUDA_VISIBLE_DEVICES=$gpu nohup python src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/muse/${method}.yaml \
        data_split=$data_split \
        task_name=$task_name \
        trainer.args.seed=$seed \
        trainer.args.gradient_checkpointing=True \
        trainer.args.eval_on_start=False \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=False \
        $extra_args \
        > "$logfile" 2>&1 &
}

run_split() {
    local data_split=$1
    local gpu_offset=${2:-0}
    echo "========================================="
    echo "Running baselines for MUSE $data_split"
    echo "========================================="

    # --- GA ---
    for i in "${!SEEDS[@]}"; do
        gpu=$((gpu_offset + i))
        seed=${SEEDS[$i]}
        launch_job $gpu "default" $data_split $seed "muse_GA_${data_split}_s${seed}"
    done
    echo "Waiting for GA ($data_split) to finish..."
    wait

    # --- NPO ---
    for i in "${!SEEDS[@]}"; do
        gpu=$((gpu_offset + i))
        seed=${SEEDS[$i]}
        launch_job $gpu "npo" $data_split $seed "muse_NPO_${data_split}_s${seed}"
    done
    echo "Waiting for NPO ($data_split) to finish..."
    wait

    # --- SimNPO ---
    for i in "${!SEEDS[@]}"; do
        gpu=$((gpu_offset + i))
        seed=${SEEDS[$i]}
        launch_job $gpu "simnpo" $data_split $seed "muse_SimNPO_${data_split}_s${seed}"
    done
    echo "Waiting for SimNPO ($data_split) to finish..."
    wait

    # --- RMU ---
    for i in "${!SEEDS[@]}"; do
        gpu=$((gpu_offset + i))
        seed=${SEEDS[$i]}
        launch_job $gpu "rmu" $data_split $seed "muse_RMU_${data_split}_s${seed}"
    done
    echo "Waiting for RMU ($data_split) to finish..."
    wait

    # --- CEU ---
    for i in "${!SEEDS[@]}"; do
        gpu=$((gpu_offset + i))
        seed=${SEEDS[$i]}
        launch_job $gpu "ceu" $data_split $seed "muse_CEU_${data_split}_s${seed}"
    done
    echo "Waiting for CEU ($data_split) to finish..."
    wait

    # --- PDU ---
    local pdu_extra=""
    if [ "$data_split" == "Books" ]; then
        pdu_extra="trainer.method_args.retain_loss_eps=0.1 trainer.args.num_train_epochs=1"
    fi
    for i in "${!SEEDS[@]}"; do
        gpu=$((gpu_offset + i))
        seed=${SEEDS[$i]}
        launch_job $gpu "pdu" $data_split $seed "muse_PDU_${data_split}_s${seed}" $pdu_extra
    done
    echo "Waiting for PDU ($data_split) to finish..."
    wait

    # --- BLUR-NPO ---
    local blur_extra=""
    if [ "$data_split" == "Books" ]; then
        blur_extra="trainer.method_args.beta=0.4 trainer.args.learning_rate=1e-5"
    fi
    for i in "${!SEEDS[@]}"; do
        gpu=$((gpu_offset + i))
        seed=${SEEDS[$i]}
        launch_job $gpu "blurnpo" $data_split $seed "muse_BLURNPO_${data_split}_s${seed}" $blur_extra
    done
    echo "Waiting for BLUR-NPO ($data_split) to finish..."
    wait

    echo "========================================="
    echo "All baselines for MUSE $data_split DONE"
    echo "========================================="
}

if [ "$SPLIT" == "news" ] || [ "$SPLIT" == "all" ]; then
    run_split "News" 0
fi

if [ "$SPLIT" == "books" ] || [ "$SPLIT" == "all" ]; then
    run_split "Books" 0
fi

echo "ALL DONE."
