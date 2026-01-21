#!/bin/bash

per_device_train_batch_size=8
gradient_accumulation_steps=1

model=Llama-2-7b-hf

data_splits=(
    "News"
    # "Books"
)

trainers_experiments=(
    #"SIBL unlearn/muse/sibl.yaml"
    #"GradAscent unlearn/muse/default.yaml"
    #"GradDiff unlearn/muse/default.yaml"
    "NPO unlearn/muse/default.yaml"
    "SimNPO unlearn/muse/default.yaml"
    "DPO unlearn/tofu/idk.yaml"
    "RMU  unlearn/muse/default.yaml"
    # "BLURNPO unlearn/muse/default.yaml"
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

        CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
        experiment=${experiment} \
        model=${model} \
        data_split=${data_split} \
        trainer=${trainer} \
        task_name=${task_name} \
        retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true

        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=${experiment} \
        data_split=${data_split} \ 
        task_name=${task_name} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
        paths.output_dir=saves/unlearn/${trainer}/evals \
        retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
    done
done



# #########################################################
# ########### MUSE News Unlearning Scalability ############
# #########################################################


# for data_split in "${data_splits[@]}"; do
#     for trainer in "${trainers[@]}"; do
#         for scal in "forget_1" "forget_2" "forget_3" "forget_4"; do
            
#             task_name=muse_${model}_${data_split}_${trainer}_scal_${scal} \
            
#             CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
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

#             CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
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
