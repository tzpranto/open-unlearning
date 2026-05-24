#!/bin/bash
# Re-learning attack following OpenUnlearning protocol (Dorna et al., arXiv 2506.12618)
# Protocol: fine-tune unlearned model on forget set for 1 epoch, lr=2e-5, bs=32
# Metric: HM = hmean(MU, 1-fgt_Prob, 1-fgt_ROUGE)
# Recovery Rate: R = min((m^a_ret - m^b_ret)/(m^a_unl - m^b_unl), 1)
set -e
cd /data/open-unlearning

SEED=42
BACKUP="/data/open-unlearning-h100-backup/saves/unlearn"

# Common overrides for re-learning (1 epoch fine-tune on forget set)
RELEARN_ARGS="trainer.args.learning_rate=2e-5 trainer.args.num_train_epochs=1 trainer.args.save_strategy=epoch +trainer.args.save_total_limit=1 trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false trainer.args.warmup_epochs=0 ~eval seed=${SEED}"

# Common overrides for unlearning with weight saving
SAVE_ARGS="trainer.method_args.checkpoint_every_epoch=true seed=${SEED}"

###############################################################################
# STEP 1: Retrain PDU + BLADE to get weights
###############################################################################
step1_retrain() {
    echo "=== STEP 1: Retraining PDU and BLADE with weight saving ==="

    # PDU 1B (3 splits)
    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        model=Llama-3.2-1B-Instruct \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        trainer=PDU forget_split=forget01 retain_split=retain99 holdout_split=holdout01 \
        trainer.args.save_strategy=epoch +trainer.args.save_total_limit=1 seed=${SEED} \
        task_name=relearn_src_PDU_1b_01_s${SEED} &

    CUDA_VISIBLE_DEVICES=1 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        model=Llama-3.2-1B-Instruct \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        trainer=PDU forget_split=forget05 retain_split=retain95 holdout_split=holdout05 \
        trainer.args.save_strategy=epoch +trainer.args.save_total_limit=1 seed=${SEED} \
        task_name=relearn_src_PDU_1b_05_s${SEED} &

    CUDA_VISIBLE_DEVICES=2 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        model=Llama-3.2-1B-Instruct \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        trainer=PDU forget_split=forget10 retain_split=retain90 holdout_split=holdout10 \
        trainer.args.save_strategy=epoch +trainer.args.save_total_limit=1 seed=${SEED} \
        task_name=relearn_src_PDU_1b_10_s${SEED} &

    # PDU 3B (3 splits)
    CUDA_VISIBLE_DEVICES=3 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        model=Llama-3.2-3B-Instruct \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-3B-Instruct_full \
        trainer=PDU forget_split=forget01 retain_split=retain99 holdout_split=holdout01 \
        trainer.args.save_strategy=epoch +trainer.args.save_total_limit=1 seed=${SEED} \
        task_name=relearn_src_PDU_3b_01_s${SEED} &

    CUDA_VISIBLE_DEVICES=4 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        model=Llama-3.2-3B-Instruct \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-3B-Instruct_full \
        trainer=PDU forget_split=forget05 retain_split=retain95 holdout_split=holdout05 \
        trainer.args.save_strategy=epoch +trainer.args.save_total_limit=1 seed=${SEED} \
        task_name=relearn_src_PDU_3b_05_s${SEED} &

    CUDA_VISIBLE_DEVICES=5 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        model=Llama-3.2-3B-Instruct \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-3B-Instruct_full \
        trainer=PDU forget_split=forget10 retain_split=retain90 holdout_split=holdout10 \
        trainer.args.save_strategy=epoch +trainer.args.save_total_limit=1 seed=${SEED} \
        task_name=relearn_src_PDU_3b_10_s${SEED} &

    # BLADE 1B forget01 (05/10 also needed)
    CUDA_VISIBLE_DEVICES=6 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/lora_bial_1b.yaml \
        forget_split=forget01 retain_split=retain99 holdout_split=holdout01 \
        ${SAVE_ARGS} \
        task_name=relearn_src_BLADE_1b_01_s${SEED} &

    # BLADE 3B forget01 (05/10 weights exist in backup)
    CUDA_VISIBLE_DEVICES=7 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/lora_bial_3b.yaml \
        forget_split=forget01 retain_split=retain99 holdout_split=holdout01 \
        ${SAVE_ARGS} \
        task_name=relearn_src_BLADE_3b_01_s${SEED} &

    wait
    echo "Batch 1 done."

    # BLADE 1B forget05 and forget10
    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/lora_bial_1b.yaml \
        forget_split=forget05 retain_split=retain95 holdout_split=holdout05 \
        ${SAVE_ARGS} \
        task_name=relearn_src_BLADE_1b_05_s${SEED} &

    CUDA_VISIBLE_DEVICES=1 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/lora_bial_1b.yaml \
        forget_split=forget10 retain_split=retain90 holdout_split=holdout10 \
        ${SAVE_ARGS} \
        task_name=relearn_src_BLADE_1b_10_s${SEED} &

    wait
    echo "Step 1 complete. All source models trained with weights."
}

###############################################################################
# STEP 2: Re-learn (fine-tune on forget set, 1 epoch)
###############################################################################
step2_relearn() {
    echo "=== STEP 2: Re-learning attack (1 epoch on forget set) ==="

    # Helper to find the final checkpoint dir
    find_ckpt() {
        local base=$1
        # Look for checkpoint-* dir with safetensors, or converged-best
        if [ -d "${base}/converged-best" ] && ls ${base}/converged-best/*.safetensors >/dev/null 2>&1; then
            echo "${base}/converged-best"
        elif ls ${base}/checkpoint-*/*.safetensors >/dev/null 2>&1; then
            ls -d ${base}/checkpoint-* | sort -V | tail -1
        elif ls ${base}/*.safetensors >/dev/null 2>&1; then
            echo "${base}"
        else
            echo "ERROR_NO_CKPT_${base}"
        fi
    }

    # Determine checkpoint paths
    PDU_1b_01=$(find_ckpt "saves/unlearn/relearn_src_PDU_1b_01_s${SEED}")
    PDU_1b_05=$(find_ckpt "saves/unlearn/relearn_src_PDU_1b_05_s${SEED}")
    PDU_1b_10=$(find_ckpt "saves/unlearn/relearn_src_PDU_1b_10_s${SEED}")
    PDU_3b_01=$(find_ckpt "saves/unlearn/relearn_src_PDU_3b_01_s${SEED}")
    PDU_3b_05=$(find_ckpt "saves/unlearn/relearn_src_PDU_3b_05_s${SEED}")
    PDU_3b_10=$(find_ckpt "saves/unlearn/relearn_src_PDU_3b_10_s${SEED}")

    BLADE_1b_01=$(find_ckpt "saves/unlearn/relearn_src_BLADE_1b_01_s${SEED}")
    BLADE_1b_05=$(find_ckpt "saves/unlearn/relearn_src_BLADE_1b_05_s${SEED}")
    BLADE_1b_10=$(find_ckpt "saves/unlearn/relearn_src_BLADE_1b_10_s${SEED}")
    BLADE_3b_01=$(find_ckpt "saves/unlearn/relearn_src_BLADE_3b_01_s${SEED}")
    BLADE_3b_05="${BACKUP}/adaptive_Llama-3.2-3B-Instruct_forget05_s${SEED}/converged-best"
    BLADE_3b_10="${BACKUP}/adaptive_Llama-3.2-3B-Instruct_forget10_s${SEED}/converged-best"

    echo "Checkpoints:"
    echo "  PDU_1b_01=$PDU_1b_01"
    echo "  BLADE_3b_10=$BLADE_3b_10"

    # --- Batch 1: 1B models (PDU + BLADE + Retain) x 3 splits = 9 jobs, fit in 8 GPUs ---
    # 1B forget01
    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model.model_args.pretrained_model_name_or_path=${PDU_1b_01} \
        data.train.TOFU_QA_full.args.hf_args.name=forget01 \
        forget_split=forget01 holdout_split=holdout01 \
        ${RELEARN_ARGS} task_name=relearned_PDU_1b_01_s${SEED} &

    CUDA_VISIBLE_DEVICES=1 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model.model_args.pretrained_model_name_or_path=${BLADE_1b_01} \
        data.train.TOFU_QA_full.args.hf_args.name=forget01 \
        forget_split=forget01 holdout_split=holdout01 \
        ${RELEARN_ARGS} task_name=relearned_BLADE_1b_01_s${SEED} &

    CUDA_VISIBLE_DEVICES=2 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_retain99 \
        data.train.TOFU_QA_full.args.hf_args.name=forget01 \
        forget_split=forget01 holdout_split=holdout01 \
        ${RELEARN_ARGS} task_name=relearned_Retain_1b_01_s${SEED} &

    # 1B forget05
    CUDA_VISIBLE_DEVICES=3 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model.model_args.pretrained_model_name_or_path=${PDU_1b_05} \
        data.train.TOFU_QA_full.args.hf_args.name=forget05 \
        forget_split=forget05 holdout_split=holdout05 \
        ${RELEARN_ARGS} task_name=relearned_PDU_1b_05_s${SEED} &

    CUDA_VISIBLE_DEVICES=4 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model.model_args.pretrained_model_name_or_path=${BLADE_1b_05} \
        data.train.TOFU_QA_full.args.hf_args.name=forget05 \
        forget_split=forget05 holdout_split=holdout05 \
        ${RELEARN_ARGS} task_name=relearned_BLADE_1b_05_s${SEED} &

    CUDA_VISIBLE_DEVICES=5 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95 \
        data.train.TOFU_QA_full.args.hf_args.name=forget05 \
        forget_split=forget05 holdout_split=holdout05 \
        ${RELEARN_ARGS} task_name=relearned_Retain_1b_05_s${SEED} &

    # 1B forget10
    CUDA_VISIBLE_DEVICES=6 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model.model_args.pretrained_model_name_or_path=${PDU_1b_10} \
        data.train.TOFU_QA_full.args.hf_args.name=forget10 \
        forget_split=forget10 holdout_split=holdout10 \
        ${RELEARN_ARGS} task_name=relearned_PDU_1b_10_s${SEED} &

    CUDA_VISIBLE_DEVICES=7 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model.model_args.pretrained_model_name_or_path=${BLADE_1b_10} \
        data.train.TOFU_QA_full.args.hf_args.name=forget10 \
        forget_split=forget10 holdout_split=holdout10 \
        ${RELEARN_ARGS} task_name=relearned_BLADE_1b_10_s${SEED} &

    wait
    echo "Batch 2a done (1B PDU+BLADE). Running 1B Retain forget10 + 3B..."

    # 1B retain forget10 (missed above)
    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
        data.train.TOFU_QA_full.args.hf_args.name=forget10 \
        forget_split=forget10 holdout_split=holdout10 \
        ${RELEARN_ARGS} task_name=relearned_Retain_1b_10_s${SEED} &

    # --- Batch 2: 3B models ---
    # 3B forget01
    CUDA_VISIBLE_DEVICES=1 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model=Llama-3.2-3B-Instruct \
        model.model_args.pretrained_model_name_or_path=${PDU_3b_01} \
        data.train.TOFU_QA_full.args.hf_args.name=forget01 \
        forget_split=forget01 holdout_split=holdout01 \
        ${RELEARN_ARGS} task_name=relearned_PDU_3b_01_s${SEED} &

    CUDA_VISIBLE_DEVICES=2 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model=Llama-3.2-3B-Instruct \
        model.model_args.pretrained_model_name_or_path=${BLADE_3b_01} \
        data.train.TOFU_QA_full.args.hf_args.name=forget01 \
        forget_split=forget01 holdout_split=holdout01 \
        ${RELEARN_ARGS} task_name=relearned_BLADE_3b_01_s${SEED} &

    CUDA_VISIBLE_DEVICES=3 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model=Llama-3.2-3B-Instruct \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-3B-Instruct_retain99 \
        data.train.TOFU_QA_full.args.hf_args.name=forget01 \
        forget_split=forget01 holdout_split=holdout01 \
        ${RELEARN_ARGS} task_name=relearned_Retain_3b_01_s${SEED} &

    # 3B forget05
    CUDA_VISIBLE_DEVICES=4 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model=Llama-3.2-3B-Instruct \
        model.model_args.pretrained_model_name_or_path=${PDU_3b_05} \
        data.train.TOFU_QA_full.args.hf_args.name=forget05 \
        forget_split=forget05 holdout_split=holdout05 \
        ${RELEARN_ARGS} task_name=relearned_PDU_3b_05_s${SEED} &

    CUDA_VISIBLE_DEVICES=5 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model=Llama-3.2-3B-Instruct \
        model.model_args.pretrained_model_name_or_path=${BLADE_3b_05} \
        data.train.TOFU_QA_full.args.hf_args.name=forget05 \
        forget_split=forget05 holdout_split=holdout05 \
        ${RELEARN_ARGS} task_name=relearned_BLADE_3b_05_s${SEED} &

    CUDA_VISIBLE_DEVICES=6 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model=Llama-3.2-3B-Instruct \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-3B-Instruct_retain95 \
        data.train.TOFU_QA_full.args.hf_args.name=forget05 \
        forget_split=forget05 holdout_split=holdout05 \
        ${RELEARN_ARGS} task_name=relearned_Retain_3b_05_s${SEED} &

    # 3B forget10
    CUDA_VISIBLE_DEVICES=7 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model=Llama-3.2-3B-Instruct \
        model.model_args.pretrained_model_name_or_path=${PDU_3b_10} \
        data.train.TOFU_QA_full.args.hf_args.name=forget10 \
        forget_split=forget10 holdout_split=holdout10 \
        ${RELEARN_ARGS} task_name=relearned_PDU_3b_10_s${SEED} &

    wait

    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model=Llama-3.2-3B-Instruct \
        model.model_args.pretrained_model_name_or_path=${BLADE_3b_10} \
        data.train.TOFU_QA_full.args.hf_args.name=forget10 \
        forget_split=forget10 holdout_split=holdout10 \
        ${RELEARN_ARGS} task_name=relearned_BLADE_3b_10_s${SEED} &

    CUDA_VISIBLE_DEVICES=1 python src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default.yaml \
        model=Llama-3.2-3B-Instruct \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-3B-Instruct_retain90 \
        data.train.TOFU_QA_full.args.hf_args.name=forget10 \
        forget_split=forget10 holdout_split=holdout10 \
        ${RELEARN_ARGS} task_name=relearned_Retain_3b_10_s${SEED} &

    wait
    echo "Step 2 complete. All models re-learned."
}

###############################################################################
# STEP 3: Eval all re-learned models
###############################################################################
step3_eval() {
    echo "=== STEP 3: Evaluating re-learned models ==="

    find_ckpt() {
        local base=$1
        if [ -d "${base}/converged-best" ] && ls ${base}/converged-best/*.safetensors >/dev/null 2>&1; then
            echo "${base}/converged-best"
        elif ls ${base}/checkpoint-*/*.safetensors >/dev/null 2>&1; then
            ls -d ${base}/checkpoint-* | sort -V | tail -1
        elif ls ${base}/*.safetensors >/dev/null 2>&1; then
            echo "${base}"
        else
            echo "ERROR_NO_CKPT_${base}"
        fi
    }

    # Eval 1B models (9 evals)
    gpu=0
    for method in PDU BLADE Retain; do
        for split_key in "01 forget01 holdout01" "05 forget05 holdout05" "10 forget10 holdout10"; do
            read -r pct forget holdout <<< "$split_key"
            ckpt=$(find_ckpt "saves/finetune/relearned_${method}_1b_${pct}_s${SEED}")
            CUDA_VISIBLE_DEVICES=${gpu} python src/eval.py \
                experiment=eval/tofu/default.yaml \
                model=Llama-3.2-1B-Instruct \
                model.model_args.pretrained_model_name_or_path=${ckpt} \
                eval.tofu.forget_split=${forget} \
                eval.tofu.holdout_split=${holdout} \
                eval.tofu.overwrite=true \
                paths.output_dir=saves/relearn/eval_${method}_1b_${pct}_s${SEED} &
            gpu=$(( (gpu + 1) % 8 ))
        done
    done
    wait
    echo "1B evals done."

    # Eval 3B models (9 evals)
    gpu=0
    for method in PDU BLADE Retain; do
        for split_key in "01 forget01 holdout01" "05 forget05 holdout05" "10 forget10 holdout10"; do
            read -r pct forget holdout <<< "$split_key"
            ckpt=$(find_ckpt "saves/finetune/relearned_${method}_3b_${pct}_s${SEED}")
            CUDA_VISIBLE_DEVICES=${gpu} python src/eval.py \
                experiment=eval/tofu/default.yaml \
                model=Llama-3.2-3B-Instruct \
                model.model_args.pretrained_model_name_or_path=${ckpt} \
                eval.tofu.forget_split=${forget} \
                eval.tofu.holdout_split=${holdout} \
                eval.tofu.overwrite=true \
                paths.output_dir=saves/relearn/eval_${method}_3b_${pct}_s${SEED} &
            gpu=$(( (gpu + 1) % 8 ))
        done
    done
    wait
    echo "Step 3 complete. All evals done."
}

###############################################################################
# MAIN
###############################################################################
STEP=${1:-all}
case $STEP in
    step1) step1_retrain ;;
    step2) step2_relearn ;;
    step3) step3_eval ;;
    all)
        step1_retrain
        step2_relearn
        step3_eval
        echo "All done. Run: python scripts/compute_relearn_recovery.py"
        ;;
    *) echo "Usage: $0 {step1|step2|step3|all}" ;;
esac
