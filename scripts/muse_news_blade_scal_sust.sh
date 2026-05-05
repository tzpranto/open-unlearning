#!/bin/bash
# MUSE News BLADE: Scale + Sustain experiments
# Scale: increasing data size (889→3554), independent runs
#   - scal_f1 MUST reproduce vanilla News result (same data, same config)
#   - T scaled proportionally to data size (more data = more steps needed)
# Sustain: sequential LoRA chaining, same data size (889), T=300
#
# Vanilla News config (the champion result):
#   T=300, conv_patience=20, eps_mul=3.2, eta_theta=3e-5, K=3
#   LR auto-calibrated to ~6e-5, ran all 300 steps without convergence
#   Result: fgt_know=0.550, fgt_verb=0.173, ret_know=0.475, HM=0.542

set -e
cd /datadrive/forked/open-unlearning

SEED=42
BASE_DIR="./saves/unlearn"
# Books config gives eps_mul=3.2, eta_theta=3e-5 — matches vanilla News run
COMMON_OVERRIDES="data_split=News model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-News_target"

# ============================================================
# SCALE: 4 independent folds
# scal_f1 = 889 rows (= vanilla), T=300
# scal_f2 = 1778 rows (2×), T=300
# scal_f3 = 2667 rows (3×), T=400
# scal_f4 = 3554 rows (4×), T=500
# Rationale: vanilla needed 300 steps (5.4 epochs) for 889 rows.
# Larger data has more samples/epoch, test if BLADE scales sublinearly.
# ============================================================
SCALE_T=(300 300 400 500)
for fold in 1 2 3 4; do
    T=${SCALE_T[$((fold-1))]}
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
# Each fold uses T=300, conv_patience=20 (same as vanilla)
# Fold 1 trains from scratch (= vanilla), folds 2-4 init from previous LoRA
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
