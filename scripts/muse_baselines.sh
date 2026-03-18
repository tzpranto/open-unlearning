#!/bin/bash

# Reduce CUDA fragmentation (can help with OOM when memory is almost full)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# echo "Master Port: $MASTER_PORT"


per_device_train_batch_size=1
gradient_accumulation_steps=2

model=Llama-2-7b-hf
debug_mode=${DEBUG_SIBL:-0}

# Resolve python from the active conda env first.
PYTHON_BIN=python
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
echo "[INFO] Using python: ${PYTHON_BIN} ($(${PYTHON_BIN} -V 2>&1))"

# Prefer stable attention fallback when flash_attn is unavailable.
attn_impl_override=()
if "${PYTHON_BIN}" -c "import importlib.util as u; raise SystemExit(0 if u.find_spec('flash_attn') else 1)"; then
    echo "[INFO] flash_attn detected: keeping model default attention implementation."
else
    echo "[INFO] flash_attn not found: overriding attention implementation to sdpa."
    attn_impl_override+=(model.model_args.attn_implementation=sdpa)
fi

data_splits=(
    "News"
    # "Books"
)

trainers_experiments=(
    "SIBL unlearn/muse/sibl.yaml"
    #"GradAscent unlearn/muse/default.yaml"
    #"GradDiff unlearn/muse/default.yaml"
    # "NPO unlearn/muse/default.yaml"
    # "SimNPO unlearn/muse/default.yaml"
    #"DPO unlearn/tofu/idk.yaml"
    #"RMU  unlearn/muse/default.yaml"
    #"BLURNPO unlearn/muse/default.yaml"
)


# trainers=(
#     "GradAscent"
#     "GradDiff"
#     "NPO"
#     "SimNPO"
# )

# #########################################################
# #################### MUSE Unlearning ####################
# #########################################################


for data_split in "${data_splits[@]}"; do
    for trainer_experiment in "${trainers_experiments[@]}"; do
        trainer=$(echo $trainer_experiment | cut -d' ' -f1)
        experiment=$(echo $trainer_experiment | cut -d' ' -f2)

        task_name=muse_${model}_${data_split}_${trainer}
        extra_overrides=()
        if [[ "${debug_mode}" == "1" && "${trainer}" == "SIBL" ]]; then
            task_name=${task_name}_debug_implicit
            extra_overrides+=(
                trainer.method_args.use_implicit=true
                trainer.method_args.debug_implicit=true
                trainer.method_args.debug_save_arrays=true
                trainer.method_args.debug_stop_after_outer=2
                trainer.method_args.T=3
                trainer.method_args.K=1
                trainer.method_args.neumann_variant=richardson
                trainer.method_args.implicit_solver=neumann
            )
        fi

        # CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu_config.yaml --main_process_port $MASTER_PORT \
        if CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" \
        src/train.py --config-name=unlearn.yaml \
        experiment=${experiment} \
        model=${model} \
        data_split=${data_split} \
        trainer=${trainer} \
        task_name=${task_name} \
        retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        "${attn_impl_override[@]}" \
        "${extra_overrides[@]}"; then
            CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" src/eval.py \
            experiment=eval/muse/default.yaml \
            data_split=${data_split} \
            task_name=${task_name} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            "${attn_impl_override[@]}" \
            paths.output_dir=saves/unlearn/${task_name}/evals \
            retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
        else
            echo "[WARN] Training failed for ${task_name}, skipping eval."
        fi
    done
done



# #########################################################
# ########### MUSE News Unlearning Scalability ############
# #########################################################


# for data_split in "${data_splits[@]}"; do
#     for trainer in "${trainers[@]}"; do
#         for scal in "forget_1" "forget_2" "forget_3" "forget_4"; do
            
#             task_name=muse_${model}_${data_split}_${trainer}_scal_${scal} \
            
#             CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu_config.yaml --main_process_port $MASTER_PORT \
#             src/train.py --config-name=unlearn.yaml \
#             experiment=unlearn/muse/scalability.yaml \
#             model=${model} \
#             data_split=${data_split} \
#             forget_split=${scal} \
#             trainer=${trainer} \
#             task_name=${task_name} \
#             retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
#             trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
#             trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
#             trainer.args.ddp_find_unused_parameters=true \
#             trainer.args.gradient_checkpointing=true

#             CUDA_VISIBLE_DEVICES=0 python src/eval.py \
#             experiment=eval/muse/default.yaml \
#             data_split=${data_split} \ 
#             task_name=${task_name} \
#             model=${model} \
#             model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
#             paths.output_dir=saves/unlearn/${trainer}/evals \
#             retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
#         done
#     done
# done



# #########################################################
# ########### MUSE News Unlearning sustainability #########
# #########################################################


# for data_split in "${data_splits[@]}"; do
#     for trainer in "${trainers[@]}"; do
#         model_path=muse-bench/MUSE-${data_split}_target
#         for sust in "forget_1" "forget_2" "forget_3" "forget_4"; do
            
#             task_name=muse_${model}_${data_split}_${trainer}_sust_${sust}

#             CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu_config.yaml --main_process_port $MASTER_PORT \
#             src/train.py --config-name=unlearn.yaml \
#             experiment=unlearn/muse/sustainabilty.yaml \
#             model=${model} \
#             model.model_args.pretrained_model_name_or_path=${model_path} \
#             data_split=${data_split} \
#             trainer=${trainer} \
#             task_name=${task_name} \
#             retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
#             trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
#             trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
#             trainer.args.ddp_find_unused_parameters=true \
#             trainer.args.gradient_checkpointing=true

#             CUDA_VISIBLE_DEVICES=0 python src/eval.py \
#             experiment=eval/muse/default.yaml \
#             data_split=${data_split} \ 
#             task_name=${task_name} \
#             model=${model} \
#             model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
#             paths.output_dir=saves/unlearn/${trainer}/evals \
#             retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json

#             model_path=saves/unlearn/${task_name}
#         done
#     done
# done
