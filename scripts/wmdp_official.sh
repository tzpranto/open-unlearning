#!/bin/bash
# Run official RMU and BLUR-RMU code for WMDP baselines
# Then evaluate with our lm_eval pipeline
set -eo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAVES="/datadrive/forked/open-unlearning/saves/unlearn"
SEED=42

# ══════════════════════════════════════════════════════════════
# 1. Official RMU (Zephyr notebook params)
# ══════════════════════════════════════════════════════════════
RMU_OUT="${SAVES}/wmdp_zephyr-7b-beta_bio_cyber_RMU_official_s${SEED}"
if [[ -f "${RMU_OUT}/evals/LMEval_SUMMARY.json" ]]; then
    echo "[SKIP] RMU official (already done)"
else
    echo "[RMU] $(date) Training with official code..."
    cd /datadrive/forked/wmdp
    CUDA_VISIBLE_DEVICES=0 python -m rmu.unlearn \
        --model_name_or_path HuggingFaceH4/zephyr-7b-beta \
        --max_num_batches 150 \
        --batch_size 4 \
        --retain_corpora wikitext,wikitext \
        --forget_corpora bio-forget-corpus,cyber-forget-corpus \
        --steering_coeffs 6.5,6.5 \
        --alpha 1200,1200 \
        --lr 5e-5 \
        --seed ${SEED} \
        --layer_id 7 \
        --layer_ids 5,6,7 \
        --param_ids 6 \
        --output_dir "${RMU_OUT}"
    echo "[RMU] $(date) Training done. Evaluating..."

    cd /datadrive/forked/open-unlearning
    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/wmdp/default.yaml \
        task_name=wmdp_zephyr-7b-beta_bio_cyber_RMU_official_s${SEED} \
        model=zephyr-7b-beta \
        model.model_args.pretrained_model_name_or_path=${RMU_OUT} \
        paths.output_dir=${RMU_OUT}/evals \
        "eval.lm_eval.tasks=[wmdp_bio,wmdp_cyber,mmlu]"
    echo "[RMU] $(date) Done."
    cat "${RMU_OUT}/evals/LMEval_SUMMARY.json"
fi

# ══════════════════════════════════════════════════════════════
# 2. BLUR-RMU (bilevel gradient projection variant)
# ══════════════════════════════════════════════════════════════
BLUR_OUT="${SAVES}/wmdp_zephyr-7b-beta_bio_cyber_BLURRMU_official_s${SEED}"
if [[ -f "${BLUR_OUT}/evals/LMEval_SUMMARY.json" ]]; then
    echo "[SKIP] BLUR-RMU official (already done)"
else
    echo "[BLUR-RMU] $(date) Training with official code..."
    cd /datadrive/forked/BLURLLMUnlearning/WMDP
    CUDA_VISIBLE_DEVICES=0 python -m rmu.unlearn_bi \
        --model_name_or_path HuggingFaceH4/zephyr-7b-beta \
        --max_num_batches 150 \
        --batch_size 4 \
        --retain_corpora wikitext,wikitext \
        --forget_corpora bio-forget-corpus,cyber-forget-corpus \
        --steering_coeffs 6.5,6.5 \
        --alpha 800,800 \
        --lr 5e-5 \
        --seed ${SEED} \
        --layer_id 7 \
        --layer_ids 5,6,7 \
        --param_ids 6 \
        --output_dir "${BLUR_OUT}"
    echo "[BLUR-RMU] $(date) Training done. Evaluating..."

    cd /datadrive/forked/open-unlearning
    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/wmdp/default.yaml \
        task_name=wmdp_zephyr-7b-beta_bio_cyber_BLURRMU_official_s${SEED} \
        model=zephyr-7b-beta \
        model.model_args.pretrained_model_name_or_path=${BLUR_OUT} \
        paths.output_dir=${BLUR_OUT}/evals \
        "eval.lm_eval.tasks=[wmdp_bio,wmdp_cyber,mmlu]"
    echo "[BLUR-RMU] $(date) Done."
    cat "${BLUR_OUT}/evals/LMEval_SUMMARY.json"
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "ALL DONE at $(date)"
echo "══════════════════════════════════════════════════════════════"
