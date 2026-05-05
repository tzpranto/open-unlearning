#!/bin/bash
# MUSE News BLADE: Sustainability (sequential LoRA chaining)
# Each fold uses T=300, conv_patience=20 (same as vanilla)
# Fold 1 from scratch (= vanilla), folds 2-4 init from previous LoRA

set -e
cd /datadrive/forked/open-unlearning

SEED=42
BASE_DIR="./saves/unlearn"
COMMON_OVERRIDES="data_split=News model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-News_target"

PREV_LORA=""
for fold in 1 2 3 4; do
    OUT="${BASE_DIR}/muse_Llama-2-7b-hf_News_BLADE_sust_f${fold}_s${SEED}"

    # Skip if already completed (has model weights)
    if [ -f "${OUT}/model.safetensors.index.json" ] || [ -f "${OUT}/model.safetensors" ]; then
        echo "[SKIP] ${OUT##*/} already completed"
        PREV_LORA="${OUT}/lora_adapters"
        continue
    fi

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
