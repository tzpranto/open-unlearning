#!/bin/bash
# Stressed Ablations (GROUP 6) — 4 pairs, control vs test
# A14v2: asymmetric dual decay (0.1) vs symmetric (1.0)
# A1v2: inner loop K=3 vs K=0
# A10v2: adaptive λ vs fixed λ=1.0
# A12v2: ρ=0.5 vs ρ=0.0
#
# All on MUSE Books, Llama-2-7b-hf, seed=42
# Usage: nohup bash scripts/ablation_stressed.sh > saves/ablation_stressed.log 2>&1 &
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

BASE_EXP="unlearn/muse/lora_bial_adaptive_books"
SEED=42

run_ablation() {
    local NAME=$1
    shift
    local TASK="stressed_${NAME}_s${SEED}"
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
echo "STRESSED ABLATIONS — GROUP 6"
echo "Started: $(date)"
echo "========================================"

# ─── A14v2: Asymmetric Dual ───────────────────────────────────────────
# Shared: eps_mul=1.5, rho=0.5, eta_theta=5e-5, lambda_init=0.5
echo ""
echo ">>> A14v2: Asymmetric Dual Decay"
echo ""

# Control: decay_factor=0.1 (asymmetric)
run_ablation "A14v2_control" \
    trainer.method_args.epsilon_multiplier=1.5 \
    trainer.method_args.rho=0.5 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.lambda_init=0.5 \
    trainer.method_args.lambda_min=0.1 \
    trainer.method_args.dual_decay_factor=0.1 \
    trainer.method_args.T=250

# Test: decay_factor=1.0 (symmetric)
run_ablation "A14v2_test" \
    trainer.method_args.epsilon_multiplier=1.5 \
    trainer.method_args.rho=0.5 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.lambda_init=0.5 \
    trainer.method_args.lambda_min=0.1 \
    trainer.method_args.dual_decay_factor=1.0 \
    trainer.method_args.T=250

# ─── A1v2: Inner Loop ─────────────────────────────────────────────────
# Shared: eps_mul=1.5, rho=0.05, eta_theta=8e-5, lambda_init=1.0
echo ""
echo ">>> A1v2: Inner Loop K"
echo ""

# Control: K=3
run_ablation "A1v2_control" \
    trainer.method_args.epsilon_multiplier=1.5 \
    trainer.method_args.rho=0.05 \
    trainer.method_args.eta_theta=8e-5 \
    trainer.method_args.lambda_init=1.0 \
    trainer.method_args.K=3 \
    trainer.method_args.T=250

# Test: K=0
run_ablation "A1v2_test" \
    trainer.method_args.epsilon_multiplier=1.5 \
    trainer.method_args.rho=0.05 \
    trainer.method_args.eta_theta=8e-5 \
    trainer.method_args.lambda_init=1.0 \
    trainer.method_args.K=0 \
    trainer.method_args.T=250

# ─── A10v2: Adaptive λ ────────────────────────────────────────────────
# Shared: eps_mul=1.5, rho=0.4, eta_theta=3e-5, K=1
echo ""
echo ">>> A10v2: Adaptive λ"
echo ""

# Control: adaptive (lambda_init=1.0, normal min/max)
run_ablation "A10v2_control" \
    trainer.method_args.epsilon_multiplier=1.5 \
    trainer.method_args.rho=0.4 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.K=1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.method_args.lambda_min=0.1 \
    trainer.method_args.T=250

# Test: fixed λ=1.0 (lambda_min=1.0, lambda_max=1.0)
run_ablation "A10v2_test" \
    trainer.method_args.epsilon_multiplier=1.5 \
    trainer.method_args.rho=0.4 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.K=1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.method_args.lambda_min=1.0 \
    trainer.method_args.lambda_max=1.0 \
    trainer.method_args.T=250

# ─── A12v2: ρ Penalty ─────────────────────────────────────────────────
# Shared: eps_mul=1.3, eta_theta=5e-5, K=2, lambda_init=0.5
echo ""
echo ">>> A12v2: ρ Penalty Augmentation"
echo ""

# Control: rho=0.5
run_ablation "A12v2_control" \
    trainer.method_args.epsilon_multiplier=1.3 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.K=2 \
    trainer.method_args.lambda_init=0.5 \
    trainer.method_args.rho=0.5 \
    trainer.method_args.T=250

# Test: rho=0.0
run_ablation "A12v2_test" \
    trainer.method_args.epsilon_multiplier=1.3 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.K=2 \
    trainer.method_args.lambda_init=0.5 \
    trainer.method_args.rho=0.0 \
    trainer.method_args.T=250

echo ""
echo "========================================"
echo "ALL STRESSED ABLATIONS DONE — $(date)"
echo "========================================"
