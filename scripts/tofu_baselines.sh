#!/bin/bash
# TOFU baselines — follows upstream tofu_unlearn.sh pattern exactly
# Single GPU, bs=8, accum=1 (eff_bs=8)
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /datadrive/forked/open-unlearning

models=(
    "Llama-3.2-1B-Instruct"
)
trainers_experiments=(
    "GradAscent unlearn/tofu/default.yaml"
    "GradDiff unlearn/tofu/default.yaml"
    "NPO unlearn/tofu/default.yaml"
    "SimNPO unlearn/tofu/default.yaml"
    "RMU unlearn/tofu/default.yaml"
    "BLURNPO unlearn/tofu/default.yaml"
    "PDU unlearn/tofu/default.yaml"
)
splits=(
    "forget01 holdout01 retain99"
)

per_device_train_batch_size=8
gradient_accumulation_steps=1

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            trainer=$(echo $trainer_experiment | cut -d' ' -f1)
            experiment=$(echo $trainer_experiment | cut -d' ' -f2)

            task_name=tofu_${model}_${forget_split}_${trainer}
            model_path=open-unlearning/tofu_${model}_full
            echo "${task_name}: Unlearning ${model_path} using ${trainer}"

            # Skip if already done
            if [[ -f "saves/unlearn/${task_name}/evals/TOFU_SUMMARY.json" ]]; then
                echo "[SKIP] ${task_name} — already done"
                cat "saves/unlearn/${task_name}/evals/TOFU_SUMMARY.json"
                continue
            fi

            # Train
            CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
            experiment=${experiment} \
            trainer=${trainer} \
            task_name=${task_name} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
            trainer.args.gradient_checkpointing=true \
            trainer.args.eval_strategy=no \
            trainer.args.do_eval=false \
            trainer.args.eval_on_start=false

            # Eval
            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            model=${model} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            paths.output_dir=saves/unlearn/${task_name}/evals \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

            echo "[DONE] ${task_name}"
            cat "saves/unlearn/${task_name}/evals/TOFU_SUMMARY.json" 2>/dev/null || true

            # Clean checkpoints
            rm -rf "saves/unlearn/${task_name}"/checkpoint-* 2>/dev/null
        done
    done
done

echo "ALL DONE"
