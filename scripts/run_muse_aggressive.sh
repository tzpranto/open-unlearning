#!/bin/bash
# Aggressive MUSE baseline schedule — use all 8 GPUs, 2 GPUs per ref-model method
# After PDU finishes: GD(2gpu×4seeds=8GPU), then NPO(2gpu×4seeds), RMU(1gpu×5seeds+spare)
# Then Books

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/conda_envs/unlearn
export HF_HOME=/data/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/open-unlearning

MODEL="Llama-2-7b-hf"
SEEDS=(42 123 456 789 1024)
COMMON="trainer.args.gradient_checkpointing=true trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false"

# For 2-GPU deepspeed runs
DS_CONFIG="configs/accelerate/default_config.yaml"

launch_2gpu() {
    local gpu1=$1
    local gpu2=$2
    local seed=$3
    local data_split=$4
    local task_name=$5
    local logfile=$6
    shift 6
    local extra_args="$@"

    CUDA_VISIBLE_DEVICES=${gpu1},${gpu2} accelerate launch \
        --config_file $DS_CONFIG \
        --num_processes 2 \
        src/train.py \
        --config-name=unlearn.yaml \
        trainer.args.seed=$seed \
        task_name=$task_name \
        $COMMON \
        $extra_args \
        > "$logfile" 2>&1 &
}

launch_1gpu() {
    local gpu=$1
    local seed=$2
    local data_split=$3
    local task_name=$4
    local logfile=$5
    shift 5
    local extra_args="$@"

    CUDA_VISIBLE_DEVICES=$gpu python src/train.py \
        --config-name=unlearn.yaml \
        trainer.args.seed=$seed \
        task_name=$task_name \
        $COMMON \
        $extra_args \
        > "$logfile" 2>&1 &
}

run_news() {
    local DATA_SPLIT="News"
    local RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"

    echo "============================================"
    echo "NEWS — Aggressive schedule — $(date)"
    echo "============================================"

    # --- GD: 4×2GPU + overlap 5th GD with 3 NPO seeds ---
    echo "[$(date '+%H:%M:%S')] GD News — 4 seeds on 8 GPUs (2 per seed)..."
    mkdir -p logs/muse_baselines/News/GradDiff
    mkdir -p logs/muse_baselines/News/NPO
    for i in 0 1 2 3; do
        seed=${SEEDS[$i]}
        gpu1=$((i * 2))
        gpu2=$((i * 2 + 1))
        task="muse_${MODEL}_${DATA_SPLIT}_GradDiff_s${seed}"
        launch_2gpu $gpu1 $gpu2 $seed $DATA_SPLIT $task \
            "logs/muse_baselines/News/GradDiff/seed${seed}.log" \
            experiment=unlearn/muse/default.yaml \
            model=${MODEL} data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer=GradDiff \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=8
    done
    wait
    echo "[$(date '+%H:%M:%S')] GD News (4/5) done. Launching GD seed5 + NPO seeds 1-3..."

    # 5th GD on GPUs 0,1 + 3 NPO seeds on GPUs 2-3, 4-5, 6-7
    seed=${SEEDS[4]}
    task="muse_${MODEL}_${DATA_SPLIT}_GradDiff_s${seed}"
    launch_2gpu 0 1 $seed $DATA_SPLIT $task \
        "logs/muse_baselines/News/GradDiff/seed${seed}.log" \
        experiment=unlearn/muse/default.yaml \
        model=${MODEL} data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
        trainer=GradDiff \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.gradient_accumulation_steps=8

    for i in 0 1 2; do
        seed=${SEEDS[$i]}
        gpu1=$((2 + i * 2))
        gpu2=$((3 + i * 2))
        task="muse_${MODEL}_${DATA_SPLIT}_NPO_s${seed}"
        launch_2gpu $gpu1 $gpu2 $seed $DATA_SPLIT $task \
            "logs/muse_baselines/News/NPO/seed${seed}.log" \
            experiment=unlearn/muse/default.yaml \
            model=${MODEL} data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer=NPO \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=4
    done
    wait
    echo "[$(date '+%H:%M:%S')] GD News DONE + NPO (3/5) done"

    # NPO seeds 4,5 on GPUs 0-1, 2-3 + RMU seeds 1-4 on GPUs 4,5,6,7
    echo "[$(date '+%H:%M:%S')] NPO seeds 4-5 + RMU seeds 1-4..."
    mkdir -p logs/muse_baselines/News/RMU
    for i in 3 4; do
        seed=${SEEDS[$i]}
        gpu1=$(( (i - 3) * 2 ))
        gpu2=$(( (i - 3) * 2 + 1 ))
        task="muse_${MODEL}_${DATA_SPLIT}_NPO_s${seed}"
        launch_2gpu $gpu1 $gpu2 $seed $DATA_SPLIT $task \
            "logs/muse_baselines/News/NPO/seed${seed}.log" \
            experiment=unlearn/muse/default.yaml \
            model=${MODEL} data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer=NPO \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=4
    done
    for i in 0 1 2 3; do
        seed=${SEEDS[$i]}
        gpu=$((4 + i))
        task="muse_${MODEL}_${DATA_SPLIT}_RMU_s${seed}"
        launch_1gpu $gpu $seed $DATA_SPLIT $task \
            "logs/muse_baselines/News/RMU/seed${seed}.log" \
            experiment=unlearn/muse/rmu.yaml \
            data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS}
    done
    wait
    echo "[$(date '+%H:%M:%S')] NPO News DONE + RMU (4/5) done"

    # RMU seed 5 on GPU 0 + BLURNPO seeds 1-3 on GPUs 2-3, 4-5, 6-7
    echo "[$(date '+%H:%M:%S')] RMU seed5 + BLURNPO seeds 1-3..."
    mkdir -p logs/muse_baselines/News/BLURNPO
    seed=${SEEDS[4]}
    task="muse_${MODEL}_${DATA_SPLIT}_RMU_s${seed}"
    launch_1gpu 0 $seed $DATA_SPLIT $task \
        "logs/muse_baselines/News/RMU/seed${seed}.log" \
        experiment=unlearn/muse/rmu.yaml \
        data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS}

    for i in 0 1 2; do
        seed=${SEEDS[$i]}
        gpu1=$((2 + i * 2))
        gpu2=$((3 + i * 2))
        task="muse_${MODEL}_${DATA_SPLIT}_BLURNPO_s${seed}"
        launch_2gpu $gpu1 $gpu2 $seed $DATA_SPLIT $task \
            "logs/muse_baselines/News/BLURNPO/seed${seed}.log" \
            experiment=unlearn/muse/blurnpo_muse.yaml \
            data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer.method_args.beta=0.05 \
            trainer.args.learning_rate=2.5e-5 \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=4
    done
    wait
    echo "[$(date '+%H:%M:%S')] RMU News DONE + BLURNPO (3/5) done"

    # BLURNPO seeds 4,5 on GPUs 0-1, 2-3
    echo "[$(date '+%H:%M:%S')] BLURNPO seeds 4-5..."
    for i in 3 4; do
        seed=${SEEDS[$i]}
        gpu1=$(( (i - 3) * 2 ))
        gpu2=$(( (i - 3) * 2 + 1 ))
        task="muse_${MODEL}_${DATA_SPLIT}_BLURNPO_s${seed}"
        launch_2gpu $gpu1 $gpu2 $seed $DATA_SPLIT $task \
            "logs/muse_baselines/News/BLURNPO/seed${seed}.log" \
            experiment=unlearn/muse/blurnpo_muse.yaml \
            data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer.method_args.beta=0.05 \
            trainer.args.learning_rate=2.5e-5 \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=4
    done
    wait
    echo "[$(date '+%H:%M:%S')] BLURNPO News DONE"

    echo "[$(date '+%H:%M:%S')] === ALL NEWS TRAINING DONE ==="
}

run_books() {
    local DATA_SPLIT="Books"
    local RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"

    echo ""
    echo "============================================"
    echo "BOOKS — Aggressive schedule — $(date)"
    echo "============================================"

    # Generate retain logs first
    if [ ! -f "$RETAIN_LOGS" ]; then
        echo "[$(date '+%H:%M:%S')] Generating Books retain logs on GPU 0..."
        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/muse/default.yaml \
            data_split=${DATA_SPLIT} \
            task_name=muse_${MODEL}_${DATA_SPLIT}_retrain \
            model=${MODEL} \
            model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-${DATA_SPLIT}_target \
            paths.output_dir=saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain \
            > logs/muse_baselines/Books/retain_logs.log 2>&1
        echo "[$(date '+%H:%M:%S')] Books retain logs done"
    fi

    # --- GA: 5×1GPU on 0-4, spare GPUs 5-7 idle briefly (GA is fast, no overlap needed) ---
    echo "[$(date '+%H:%M:%S')] GA Books — 5 seeds on GPUs 0-4..."
    mkdir -p logs/muse_baselines/Books/GradAscent
    mkdir -p logs/muse_baselines/Books/GradDiff
    for i in "${!SEEDS[@]}"; do
        seed=${SEEDS[$i]}
        task="muse_${MODEL}_${DATA_SPLIT}_GradAscent_s${seed}"
        launch_1gpu $i $seed $DATA_SPLIT $task \
            "logs/muse_baselines/Books/GradAscent/seed${seed}.log" \
            experiment=unlearn/muse/default.yaml \
            model=${MODEL} data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=8
    done
    wait
    echo "[$(date '+%H:%M:%S')] GA Books DONE"

    # --- GD 4×2GPU + overlap 5th with NPO ---
    echo "[$(date '+%H:%M:%S')] GD Books — 4 seeds (8 GPUs)..."
    mkdir -p logs/muse_baselines/Books/NPO
    for i in 0 1 2 3; do
        seed=${SEEDS[$i]}
        gpu1=$((i * 2))
        gpu2=$((i * 2 + 1))
        task="muse_${MODEL}_${DATA_SPLIT}_GradDiff_s${seed}"
        launch_2gpu $gpu1 $gpu2 $seed $DATA_SPLIT $task \
            "logs/muse_baselines/Books/GradDiff/seed${seed}.log" \
            experiment=unlearn/muse/default.yaml \
            model=${MODEL} data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer=GradDiff \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=8
    done
    wait

    # GD seed5 (0,1) + NPO seeds 1-3 (2-3, 4-5, 6-7)
    echo "[$(date '+%H:%M:%S')] GD seed5 + NPO seeds 1-3..."
    seed=${SEEDS[4]}
    task="muse_${MODEL}_${DATA_SPLIT}_GradDiff_s${seed}"
    launch_2gpu 0 1 $seed $DATA_SPLIT $task \
        "logs/muse_baselines/Books/GradDiff/seed${seed}.log" \
        experiment=unlearn/muse/default.yaml \
        model=${MODEL} data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
        trainer=GradDiff \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.gradient_accumulation_steps=8
    for i in 0 1 2; do
        seed=${SEEDS[$i]}
        gpu1=$((2 + i * 2))
        gpu2=$((3 + i * 2))
        task="muse_${MODEL}_${DATA_SPLIT}_NPO_s${seed}"
        launch_2gpu $gpu1 $gpu2 $seed $DATA_SPLIT $task \
            "logs/muse_baselines/Books/NPO/seed${seed}.log" \
            experiment=unlearn/muse/default.yaml \
            model=${MODEL} data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer=NPO \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=4
    done
    wait
    echo "[$(date '+%H:%M:%S')] GD Books DONE + NPO (3/5)"

    # NPO seeds 4-5 (0-1, 2-3) + SimNPO seeds 1-3 (4-5, 6-7, ...)
    # Only 2 SimNPO fit on remaining GPUs 4-5,6-7
    echo "[$(date '+%H:%M:%S')] NPO seeds 4-5 + SimNPO seeds 1-2..."
    mkdir -p logs/muse_baselines/Books/SimNPO
    for i in 3 4; do
        seed=${SEEDS[$i]}
        gpu1=$(( (i - 3) * 2 ))
        gpu2=$(( (i - 3) * 2 + 1 ))
        task="muse_${MODEL}_${DATA_SPLIT}_NPO_s${seed}"
        launch_2gpu $gpu1 $gpu2 $seed $DATA_SPLIT $task \
            "logs/muse_baselines/Books/NPO/seed${seed}.log" \
            experiment=unlearn/muse/default.yaml \
            model=${MODEL} data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer=NPO \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=4
    done
    for i in 0 1; do
        seed=${SEEDS[$i]}
        gpu1=$((4 + i * 2))
        gpu2=$((5 + i * 2))
        task="muse_${MODEL}_${DATA_SPLIT}_SimNPO_s${seed}"
        launch_2gpu $gpu1 $gpu2 $seed $DATA_SPLIT $task \
            "logs/muse_baselines/Books/SimNPO/seed${seed}.log" \
            experiment=unlearn/muse/default.yaml \
            model=${MODEL} data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer=SimNPO \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=4
    done
    wait
    echo "[$(date '+%H:%M:%S')] NPO Books DONE + SimNPO (2/5)"

    # SimNPO seeds 3-5 (3×2GPU = 6 GPUs) + RMU seeds 1-2 on GPUs 6,7
    echo "[$(date '+%H:%M:%S')] SimNPO seeds 3-5 + RMU seeds 1-2..."
    mkdir -p logs/muse_baselines/Books/RMU
    for i in 2 3 4; do
        seed=${SEEDS[$i]}
        gpu1=$(( (i - 2) * 2 ))
        gpu2=$(( (i - 2) * 2 + 1 ))
        task="muse_${MODEL}_${DATA_SPLIT}_SimNPO_s${seed}"
        launch_2gpu $gpu1 $gpu2 $seed $DATA_SPLIT $task \
            "logs/muse_baselines/Books/SimNPO/seed${seed}.log" \
            experiment=unlearn/muse/default.yaml \
            model=${MODEL} data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer=SimNPO \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=4
    done
    for i in 0 1; do
        seed=${SEEDS[$i]}
        gpu=$((6 + i))
        task="muse_${MODEL}_${DATA_SPLIT}_RMU_s${seed}"
        launch_1gpu $gpu $seed $DATA_SPLIT $task \
            "logs/muse_baselines/Books/RMU/seed${seed}.log" \
            experiment=unlearn/muse/rmu.yaml \
            data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS}
    done
    wait
    echo "[$(date '+%H:%M:%S')] SimNPO Books DONE + RMU (2/5)"

    # RMU seeds 3-5 on GPUs 0,1,2 + PDU seeds 1-2 on GPUs (3,4), (5,6) + spare GPU 7
    echo "[$(date '+%H:%M:%S')] RMU seeds 3-5 + PDU seeds 1-2..."
    mkdir -p logs/muse_baselines/Books/PDU
    for i in 2 3 4; do
        seed=${SEEDS[$i]}
        gpu=$((i - 2))
        task="muse_${MODEL}_${DATA_SPLIT}_RMU_s${seed}"
        launch_1gpu $gpu $seed $DATA_SPLIT $task \
            "logs/muse_baselines/Books/RMU/seed${seed}.log" \
            experiment=unlearn/muse/rmu.yaml \
            data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS}
    done
    for i in 0 1; do
        seed=${SEEDS[$i]}
        gpu1=$((3 + i * 2))
        gpu2=$((4 + i * 2))
        task="muse_${MODEL}_${DATA_SPLIT}_PDU_s${seed}"
        launch_2gpu $gpu1 $gpu2 $seed $DATA_SPLIT $task \
            "logs/muse_baselines/Books/PDU/seed${seed}.log" \
            experiment=unlearn/muse/pdu.yaml \
            data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer.method_args.retain_loss_eps=0.1 \
            trainer.args.num_train_epochs=1 \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=4
    done
    wait
    echo "[$(date '+%H:%M:%S')] RMU Books DONE + PDU (2/5)"

    # PDU seeds 3-5 (3×2GPU) + BLURNPO seeds 1 (2GPU)
    echo "[$(date '+%H:%M:%S')] PDU seeds 3-5 + BLURNPO seed 1..."
    mkdir -p logs/muse_baselines/Books/BLURNPO
    for i in 2 3 4; do
        seed=${SEEDS[$i]}
        gpu1=$(( (i - 2) * 2 ))
        gpu2=$(( (i - 2) * 2 + 1 ))
        task="muse_${MODEL}_${DATA_SPLIT}_PDU_s${seed}"
        launch_2gpu $gpu1 $gpu2 $seed $DATA_SPLIT $task \
            "logs/muse_baselines/Books/PDU/seed${seed}.log" \
            experiment=unlearn/muse/pdu.yaml \
            data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer.method_args.retain_loss_eps=0.1 \
            trainer.args.num_train_epochs=1 \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=4
    done
    seed=${SEEDS[0]}
    task="muse_${MODEL}_${DATA_SPLIT}_BLURNPO_s${seed}"
    launch_2gpu 6 7 $seed $DATA_SPLIT $task \
        "logs/muse_baselines/Books/BLURNPO/seed${seed}.log" \
        experiment=unlearn/muse/blurnpo_muse.yaml \
        data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
        trainer.method_args.beta=0.4 \
        trainer.args.learning_rate=1e-5 \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.gradient_accumulation_steps=4
    wait
    echo "[$(date '+%H:%M:%S')] PDU Books DONE + BLURNPO (1/5)"

    # BLURNPO seeds 2-5 (4×2GPU = 8 GPUs)
    echo "[$(date '+%H:%M:%S')] BLURNPO seeds 2-5..."
    for i in 1 2 3 4; do
        seed=${SEEDS[$i]}
        gpu1=$(( (i - 1) * 2 ))
        gpu2=$(( (i - 1) * 2 + 1 ))
        task="muse_${MODEL}_${DATA_SPLIT}_BLURNPO_s${seed}"
        launch_2gpu $gpu1 $gpu2 $seed $DATA_SPLIT $task \
            "logs/muse_baselines/Books/BLURNPO/seed${seed}.log" \
            experiment=unlearn/muse/blurnpo_muse.yaml \
            data_split=${DATA_SPLIT} retain_logs_path=${RETAIN_LOGS} \
            trainer.method_args.beta=0.4 \
            trainer.args.learning_rate=1e-5 \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=4
    done
    wait
    echo "[$(date '+%H:%M:%S')] BLURNPO Books DONE"

    echo "[$(date '+%H:%M:%S')] === ALL BOOKS TRAINING DONE ==="
}

# ========== MAIN ==========
run_news
run_books

echo ""
echo "============================================"
echo "ALL TRAINING COMPLETE — $(date)"
echo "============================================"
