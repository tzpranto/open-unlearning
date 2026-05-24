#!/bin/bash
# Comprehensive eval + judge + results for both VILA and LoKU lora models
set -e
eval "$(conda shell.bash hook)"
conda activate unlearn
cd /data/open-unlearning

VILA_SAVES=/data/baselines/vila/TOFU/saves
LOKU_SAVES=/data/baselines/loku/TOFU/saves
SEEDS="42 123 456 789 1337"
SPLITS="forget01 forget05 forget10"

# ============================================================
# STEP 1: Eval all models
# ============================================================
echo "===== STEP 1: Evaluation ====="

for SPLIT in $SPLITS; do
    if [ "$SPLIT" = "forget01" ]; then
        HOLDOUT="holdout01"
        RETAIN_LOGS="saves/eval/tofu_Llama-3.2-3B-Instruct_retain99/TOFU_EVAL.json"
    elif [ "$SPLIT" = "forget05" ]; then
        HOLDOUT="holdout05"
        RETAIN_LOGS="saves/eval/tofu_Llama-3.2-3B-Instruct_retain95/TOFU_EVAL.json"
    elif [ "$SPLIT" = "forget10" ]; then
        HOLDOUT="holdout10"
        RETAIN_LOGS="saves/eval/tofu_Llama-3.2-3B-Instruct_retain90/TOFU_EVAL.json"
    fi

    for SEED in $SEEDS; do
        # VILA eval
        VILA_MODEL=${VILA_SAVES}/vila_3b_gd_${SPLIT}_s${SEED}
        VILA_EVAL=saves/eval/tofu_vila_3b_${SPLIT}_s${SEED}
        if [ -d "$VILA_MODEL" ] && [ ! -f "${VILA_EVAL}/TOFU_EVAL.json" ]; then
            echo "  [RUN] VILA eval $SPLIT seed=$SEED"
            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                experiment=eval/tofu/default.yaml \
                model=Llama-3.2-3B-Instruct \
                forget_split=${SPLIT} \
                holdout_split=${HOLDOUT} \
                task_name=tofu_vila_3b_${SPLIT}_s${SEED} \
                model.model_args.pretrained_model_name_or_path=${VILA_MODEL} \
                paths.output_dir=${VILA_EVAL} \
                retain_logs_path=${RETAIN_LOGS} \
                eval.tofu.overwrite=true
            echo "  [DONE] VILA eval: $VILA_EVAL"
        elif [ -f "${VILA_EVAL}/TOFU_EVAL.json" ]; then
            echo "  [SKIP] VILA eval exists: $VILA_EVAL"
        else
            echo "  [SKIP] VILA model missing: $VILA_MODEL"
        fi

        # LoKU eval
        LOKU_MODEL=${LOKU_SAVES}/llama3.2-3b_${SPLIT}_seed${SEED}
        LOKU_EVAL=saves/eval/tofu_loku_3b_${SPLIT}_s${SEED}
        if [ -d "$LOKU_MODEL" ] && [ ! -f "${LOKU_EVAL}/TOFU_EVAL.json" ]; then
            echo "  [RUN] LoKU eval $SPLIT seed=$SEED"
            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                experiment=eval/tofu/default.yaml \
                model=Llama-3.2-3B-Instruct \
                forget_split=${SPLIT} \
                holdout_split=${HOLDOUT} \
                task_name=tofu_loku_3b_${SPLIT}_s${SEED} \
                model.model_args.pretrained_model_name_or_path=${LOKU_MODEL} \
                paths.output_dir=${LOKU_EVAL} \
                retain_logs_path=${RETAIN_LOGS} \
                eval.tofu.overwrite=true
            echo "  [DONE] LoKU eval: $LOKU_EVAL"
        elif [ -f "${LOKU_EVAL}/TOFU_EVAL.json" ]; then
            echo "  [SKIP] LoKU eval exists: $LOKU_EVAL"
        else
            echo "  [SKIP] LoKU model missing: $LOKU_MODEL"
        fi
    done
done

# ============================================================
# STEP 2: LLM Judge (Opus 4.7)
# ============================================================
echo "===== STEP 2: LLM Judge ====="
export LLM_JUDGE_MODEL="claude-opus-4-7-20250514"

for SPLIT in $SPLITS; do
    for SEED in $SEEDS; do
        # VILA judge
        VILA_EVAL=saves/eval/tofu_vila_3b_${SPLIT}_s${SEED}
        if [ -f "${VILA_EVAL}/TOFU_EVAL.json" ] && [ ! -f "${VILA_EVAL}/TOFU_JUDGE.json" ]; then
            echo "  [RUN] VILA judge $SPLIT seed=$SEED"
            python scripts/llm_judge.py --eval-dir ${VILA_EVAL} --benchmark tofu
            echo "  [DONE] VILA judge: $VILA_EVAL"
        fi

        # LoKU judge
        LOKU_EVAL=saves/eval/tofu_loku_3b_${SPLIT}_s${SEED}
        if [ -f "${LOKU_EVAL}/TOFU_EVAL.json" ] && [ ! -f "${LOKU_EVAL}/TOFU_JUDGE.json" ]; then
            echo "  [RUN] LoKU judge $SPLIT seed=$SEED"
            python scripts/llm_judge.py --eval-dir ${LOKU_EVAL} --benchmark tofu
            echo "  [DONE] LoKU judge: $LOKU_EVAL"
        fi
    done
done

echo "===== ALL EVAL + JUDGE COMPLETE ====="
