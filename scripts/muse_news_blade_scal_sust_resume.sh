#!/bin/bash
# MUSE News BLADE: Resume from scale fold 3 onward
# Folds 1-2 already completed.

set -e
cd /datadrive/forked/open-unlearning

SEED=42
BASE_DIR="./saves/unlearn"
COMMON_OVERRIDES="data_split=News model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-News_target"

# ============================================================
# SCALE: folds 3-4
# ============================================================
SCALE_T=(400 500)
for fold in 3 4; do
    T=${SCALE_T[$((fold-3))]}
    OUT="${BASE_DIR}/muse_Llama-2-7b-hf_News_BLADE_scal_f${fold}_s${SEED}"
    echo "============================================================"
    echo "[SCALE FOLD ${fold}] $(date) Training ${OUT##*/} (T=${T})..."

    CUDA_VISIBLE_DEVICES=0 python src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/muse/lora_bial_adaptive_books \
        ${COMMON_OVERRIDES} \
        data.forget.MUSE_forget.args.hf_args.name=scal \
        data.forget.MUSE_forget.args.hf_args.split=forget_${fold} \
        trainer.method_args.T=${T} \
        trainer.method_args.conv_patience=20 \
        trainer.args.seed=${SEED} \
        task_name=muse_Llama-2-7b-hf_News_BLADE_scal_f${fold}_s${SEED}

    # Print results
    EVAL_FILE="${OUT}/eval_results.json"
    if [ ! -f "$EVAL_FILE" ]; then
        EVAL_FILE="${OUT}/checkpoint-0/evals/eval_results.json"
    fi
    if [ -f "$EVAL_FILE" ]; then
        echo "[DONE] ${OUT##*/}"
        cat "$EVAL_FILE"
    else
        echo "[WARN] No eval_results.json found for ${OUT##*/}"
    fi
done

# ============================================================
# SUSTAIN: sequential folds, LoRA chaining
# ============================================================
PREV_LORA=""
for fold in 1 2 3 4; do
    OUT="${BASE_DIR}/muse_Llama-2-7b-hf_News_BLADE_sust_f${fold}_s${SEED}"
    echo "============================================================"
    echo "[SUSTAIN FOLD ${fold}] $(date) Training ${OUT##*/}..."

    LORA_ARGS=""
    if [ -n "$PREV_LORA" ]; then
        LORA_ARGS="trainer.method_args.lora_init_path=${PREV_LORA}"
    fi

    CUDA_VISIBLE_DEVICES=0 python src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/muse/lora_bial_adaptive_books \
        ${COMMON_OVERRIDES} \
        data.forget.MUSE_forget.args.hf_args.name=sust \
        data.forget.MUSE_forget.args.hf_args.split=forget_${fold} \
        trainer.method_args.T=300 \
        trainer.method_args.conv_patience=20 \
        trainer.method_args.save_lora_only=true \
        trainer.args.seed=${SEED} \
        task_name=muse_Llama-2-7b-hf_News_BLADE_sust_f${fold}_s${SEED} \
        ${LORA_ARGS}

    PREV_LORA="${OUT}/lora_adapters"

    # Print results
    EVAL_FILE="${OUT}/eval_results.json"
    if [ ! -f "$EVAL_FILE" ]; then
        EVAL_FILE="${OUT}/checkpoint-0/evals/eval_results.json"
    fi
    if [ -f "$EVAL_FILE" ]; then
        echo "[DONE] ${OUT##*/}"
        cat "$EVAL_FILE"
    else
        echo "[WARN] No eval_results.json found for ${OUT##*/}"
    fi
done

echo "============================================================"
echo "ALL DONE $(date)"
