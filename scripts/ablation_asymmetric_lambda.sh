#!/bin/bash
# Asymmetric Lambda Ablation — proves asymmetric dual update matters
# eps_mul=1.05 (extremely tight constraint) forces large r swings
# 1) decay_factor=0.1 (asymmetric, current design)
# 2) decay_factor=1.0 (symmetric — should oscillate and fail)
#
# MUSE Books, Llama-2-7b-hf, seed=42
# Usage: nohup bash scripts/ablation_asymmetric_lambda.sh > saves/ablation_asymmetric_lambda.log 2>&1 &
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

BASE_EXP="unlearn/muse/lora_bial_adaptive_books"
SEED=42

run_ablation() {
    local NAME=$1
    shift
    local TASK="asym_lambda_${NAME}_s${SEED}"
    local OUTDIR="saves/unlearn/${TASK}"

    if [[ -f "${OUTDIR}/evals/MUSE_EVAL.json" ]]; then
        echo "[SKIP] ${TASK} — already evaluated"
        return 0
    fi

    echo "================================================================"
    echo "[RUN] ${TASK} — $(date '+%H:%M:%S')"
    echo "================================================================"

    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
        experiment=${BASE_EXP} \
        task_name="${TASK}" \
        trainer.args.seed=${SEED} \
        trainer.method_args.epsilon_multiplier=1.05 \
        "$@"

    echo "[DONE] ${TASK} — $(date '+%H:%M:%S')"
    echo ""
}

echo "========================================"
echo "ASYMMETRIC LAMBDA ABLATION (eps_mul=1.05)"
echo "Started: $(date)"
echo "========================================"

# ─── Asymmetric (current design): decay_factor=0.1 ──────────────────
echo ""
echo ">>> Asymmetric (decay=0.1): λ drops slowly when feasible"
echo ""

run_ablation "asym" \
    trainer.method_args.dual_decay_factor=0.1

# ─── Symmetric: decay_factor=1.0 ────────────────────────────────────
echo ""
echo ">>> Symmetric (decay=1.0): λ drops as fast as it rises"
echo ""

run_ablation "sym" \
    trainer.method_args.dual_decay_factor=1.0

echo ""
echo "========================================"
echo "ASYMMETRIC LAMBDA ABLATION DONE — $(date)"
echo "========================================"
