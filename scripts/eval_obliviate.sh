#!/bin/bash
# Evaluate all OBLIVIATE models on TOFU
# Usage: bash scripts/eval_obliviate.sh [GPU_ID]

GPU=${1:-0}
SAVE_DIR="/data/baselines/obliviate/saves"
SEEDS=(42 123 456 789 1337)
SPLITS=("forget01" "forget05" "forget10")

for split in "${SPLITS[@]}"; do
    # Map forget split to holdout split
    if [ "$split" == "forget01" ]; then
        holdout="holdout01"
    elif [ "$split" == "forget05" ]; then
        holdout="holdout05"
    else
        holdout="holdout10"
    fi

    for seed in "${SEEDS[@]}"; do
        model_path="${SAVE_DIR}/${split}_s${seed}"
        task_name="tofu_obliviate_3b_${split}_s${seed}"
        output_dir="saves/eval/${task_name}"

        if [ ! -d "$model_path" ]; then
            echo "SKIP: $model_path does not exist"
            continue
        fi

        if [ -f "${output_dir}/TOFU_EVAL.json" ]; then
            echo "SKIP: $task_name already evaluated"
            continue
        fi

        echo "Evaluating: $task_name on GPU $GPU"
        CUDA_VISIBLE_DEVICES=$GPU python -m src.eval \
            experiment=eval/tofu/default.yaml \
            model=Llama-3.2-3B-Instruct \
            forget_split=$split \
            holdout_split=$holdout \
            task_name=$task_name \
            model.model_args.pretrained_model_name_or_path=$model_path \
            paths.output_dir=$output_dir \
            retain_logs_path=saves/eval/tofu_Llama-3.2-3B-Instruct_retain99/TOFU_EVAL.json \
            eval.tofu.overwrite=true
    done
done
echo "All OBLIVIATE evaluations complete."
