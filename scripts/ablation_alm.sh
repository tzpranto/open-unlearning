#!/bin/bash
# ALM Contribution Ablation — proves the constraint mechanism matters
# 1) ALM-off: lambda=0, rho=0 (no constraint, pure bilevel)
# 2) logit_margin loss: alternative forget objective with full ALM
#
# Both on MUSE Books, Llama-2-7b-hf, seed=42
# Usage: nohup bash scripts/ablation_alm.sh > saves/ablation_alm.log 2>&1 &
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

BASE_EXP="unlearn/muse/lora_bial_adaptive_books"
SEED=42

run_ablation() {
    local NAME=$1
    shift
    local TASK="alm_${NAME}_s${SEED}"
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
        "$@"

    echo "[DONE] ${TASK} — $(date '+%H:%M:%S')"
    echo ""
}

echo "========================================"
echo "ALM CONTRIBUTION ABLATION"
echo "Started: $(date)"
echo "========================================"

# ─── ALM-off: no constraint at all ───────────────────────────────────
# lambda_init=0, lambda_min=0, lambda_max=0, rho=0
# This is pure bilevel LoRA with clamped_entropy but NO retain constraint
echo ""
echo ">>> ALM-off (λ=0, ρ=0): No constraint mechanism"
echo ""

run_ablation "off" \
    trainer.method_args.lambda_init=0.0 \
    trainer.method_args.lambda_min=0.0 \
    trainer.method_args.lambda_max=0.0 \
    trainer.method_args.rho=0.0 \
    trainer.method_args.T=250

# ─── logit_margin loss ────────────────────────────────────────────────
# Same ALM params as champion but with logit_margin forget loss
echo ""
echo ">>> logit_margin: Alternative forget loss with full ALM"
echo ""

run_ablation "logit_margin" \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.T=250

# ─── Swapped: inner=forget, outer=retain ──────────────────────────────
# Uses LoRABiALSwapped trainer: inner loop minimizes forget, outer minimizes
# retain with ALM constraint that forget stays high
echo ""
echo ">>> Swapped (inner=forget, outer=retain+ALM)"
echo ""

TASK="alm_swapped_s${SEED}"
OUTDIR="saves/unlearn/${TASK}"
if [[ -f "${OUTDIR}/evals/MUSE_EVAL.json" ]]; then
    echo "[SKIP] ${TASK} — already evaluated"
else
    echo "================================================================"
    echo "[RUN] ${TASK} — $(date '+%H:%M:%S')"
    echo "================================================================"

    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
        experiment=${BASE_EXP} \
        task_name="${TASK}" \
        trainer=LoRABiALSwapped \
        trainer.args.seed=${SEED} \
        trainer.method_args.T=250

    echo "[DONE] ${TASK} — $(date '+%H:%M:%S')"
fi

echo ""
echo "========================================"
echo "ALM ABLATION DONE — $(date)"
echo "========================================"
