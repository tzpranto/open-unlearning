#!/bin/bash
# Bilevel Stress Test — proves inner loop enables aggressive forgetting
# K=3 vs K=0 at 3× learning rate (eta_theta=9e-5)
# At this aggression, K=0 should collapse while K=3 survives via spike recovery
#
# MUSE Books, Llama-2-7b-hf, seed=42
# Usage: nohup bash scripts/ablation_bilevel_stress.sh > saves/ablation_bilevel_stress.log 2>&1 &
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

BASE_EXP="unlearn/muse/lora_bial_adaptive_books"
SEED=42
ETA=9e-5

run_ablation() {
    local NAME=$1
    shift
    local TASK="bilevel_stress_${NAME}_s${SEED}"
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
        trainer.method_args.eta_theta=${ETA} \
        "$@"

    echo "[DONE] ${TASK} — $(date '+%H:%M:%S')"
    echo ""
}

echo "========================================"
echo "BILEVEL STRESS TEST (eta_theta=9e-5)"
echo "Started: $(date)"
echo "========================================"

# ─── K=3 at 3× LR — inner loop should survive ──────────────────────
echo ""
echo ">>> K=3, eta_theta=9e-5: Bilevel with aggressive outer"
echo ""

run_ablation "K3" \
    trainer.method_args.K=3

# ─── K=0 at 3× LR — should collapse ────────────────────────────────
echo ""
echo ">>> K=0, eta_theta=9e-5: No inner loop with aggressive outer"
echo ""

run_ablation "K0" \
    trainer.method_args.K=0

echo ""
echo "========================================"
echo "BILEVEL STRESS TEST DONE — $(date)"
echo "========================================"
