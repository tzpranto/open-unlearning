#!/bin/bash
# Research Experiments: Progressive Ablation Study for Surgical Unlearning
# MUSE News / Llama-2-7b / SIBL Framework
#
# Usage: bash run_experiments.sh [exp_number]
#   e.g., bash run_experiments.sh 1    # Run only Exp1
#         bash run_experiments.sh       # Run all pending experiments
#
# Each experiment checks if results already exist and skips if so.

set -e
cd /datadrive/forked/open-unlearning

PYTHON="/datadrive/conda/envs/unlearning/bin/python"
TRACES="/datadrive/forked/open-unlearning/trace_analysis/figures/traces/analysis/neuron_traces.pt"
SAVE_BASE="saves/unlearn"
TARGET_EXP="${1:-all}"

run_if_needed() {
    local exp_name="$1"
    local save_dir="$2"
    shift 2

    # Check if results already exist
    if find "${save_dir}" -name "MUSE_SUMMARY.json" 2>/dev/null | grep -q .; then
        echo ">>> [SKIP] ${exp_name}: Results already exist in ${save_dir}"
        return 0
    fi

    echo ">>> [START] ${exp_name} at $(date)"
    echo ">>> Save dir: ${save_dir}"

    $PYTHON src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/sibl.yaml \
        "$@" 2>&1 | tee "${save_dir}_train.log" || {
        echo ">>> [FAILED] ${exp_name} at $(date)"
        return 1
    }

    echo ">>> [DONE] ${exp_name} at $(date)"
}

# ============================================================
# Exp1: Raw SIBL — no mask, no implicit, no freeze
# ============================================================
if [[ "$TARGET_EXP" == "all" || "$TARGET_EXP" == "1" ]]; then
    run_if_needed "Exp1: Raw SIBL" "${SAVE_BASE}/research_exp1_raw_sibl" \
        trainer.args.output_dir="${SAVE_BASE}/research_exp1_raw_sibl" \
        task_name=research_exp1_raw_sibl \
        trainer.method_args.use_sparsity=false \
        trainer.method_args.use_implicit=false \
        trainer.method_args.neuron_traces_path=null \
        trainer.method_args.neuron_bitmap_path=null \
        trainer.method_args.outer_freeze_layers=null \
        trainer.method_args.mask_freeze_layers=null \
        trainer.method_args.T=20 \
        trainer.method_args.K=10 \
        trainer.method_args.eta_theta=1e-4 \
        trainer.method_args.eta_in=1e-4 \
        trainer.method_args.rho=0.5 \
        trainer.method_args.epsilon=0.1 \
        trainer.method_args.forget_loss_type=logit_margin \
        trainer.method_args.regularization_type=none
fi

# ============================================================
# Exp2: Freeze layers 0-7 in outer step (no mask, no implicit)
# ============================================================
if [[ "$TARGET_EXP" == "all" || "$TARGET_EXP" == "2" ]]; then
    run_if_needed "Exp2: Freeze 0-7" "${SAVE_BASE}/research_exp2_freeze8" \
        trainer.args.output_dir="${SAVE_BASE}/research_exp2_freeze8" \
        task_name=research_exp2_freeze8 \
        trainer.method_args.use_sparsity=false \
        trainer.method_args.use_implicit=false \
        trainer.method_args.neuron_traces_path=null \
        trainer.method_args.neuron_bitmap_path=null \
        'trainer.method_args.outer_freeze_layers=[0,1,2,3,4,5,6,7]' \
        trainer.method_args.mask_freeze_layers=null \
        trainer.method_args.T=20 \
        trainer.method_args.K=10 \
        trainer.method_args.eta_theta=1e-4 \
        trainer.method_args.eta_in=1e-4 \
        trainer.method_args.rho=0.5 \
        trainer.method_args.epsilon=0.1 \
        trainer.method_args.forget_loss_type=logit_margin \
        trainer.method_args.regularization_type=none
fi

# ============================================================
# Exp3: Freeze 0-7 + implicit correction (layers 8-31)
# ============================================================
if [[ "$TARGET_EXP" == "all" || "$TARGET_EXP" == "3" ]]; then
    run_if_needed "Exp3: Freeze 0-7 + Implicit" "${SAVE_BASE}/research_exp3_freeze8_implicit" \
        trainer.args.output_dir="${SAVE_BASE}/research_exp3_freeze8_implicit" \
        task_name=research_exp3_freeze8_implicit \
        trainer.method_args.use_sparsity=false \
        trainer.method_args.use_implicit=true \
        trainer.method_args.implicit_solver=neumann \
        trainer.method_args.implicit_blockwise=true \
        'trainer.method_args.implicit_block_skip_layers=[0,1,2,3,4,5,6,7]' \
        trainer.method_args.implicit_block_include_attn=true \
        trainer.method_args.implicit_block_include_mlp=true \
        trainer.method_args.neumann_variant=richardson \
        trainer.method_args.neumann_steps=2 \
        trainer.method_args.neumann_mu=1.0 \
        trainer.method_args.neumann_alpha_default=0.01 \
        trainer.method_args.neumann_use_probe_alpha=false \
        trainer.method_args.neumann_max_growth_ratio=5.0 \
        trainer.method_args.neuron_traces_path=null \
        trainer.method_args.neuron_bitmap_path=null \
        'trainer.method_args.outer_freeze_layers=[0,1,2,3,4,5,6,7]' \
        trainer.method_args.mask_freeze_layers=null \
        trainer.method_args.T=20 \
        trainer.method_args.K=10 \
        trainer.method_args.eta_theta=1e-4 \
        trainer.method_args.eta_in=1e-4 \
        trainer.method_args.rho=0.5 \
        trainer.method_args.epsilon=0.1 \
        trainer.method_args.forget_loss_type=logit_margin \
        trainer.method_args.regularization_type=none
fi

# ============================================================
# Exp4: Exp3 + forget-neuron mask (ratio >= 1.0 only)
# ============================================================
if [[ "$TARGET_EXP" == "all" || "$TARGET_EXP" == "4" ]]; then
    run_if_needed "Exp4: Forget Mask" "${SAVE_BASE}/research_exp4_forget_mask" \
        trainer.args.output_dir="${SAVE_BASE}/research_exp4_forget_mask" \
        task_name=research_exp4_forget_mask \
        trainer.method_args.use_sparsity=false \
        trainer.method_args.use_implicit=true \
        trainer.method_args.implicit_solver=neumann \
        trainer.method_args.implicit_blockwise=true \
        'trainer.method_args.implicit_block_skip_layers=[0,1,2,3,4,5,6,7]' \
        trainer.method_args.implicit_block_include_attn=true \
        trainer.method_args.implicit_block_include_mlp=true \
        trainer.method_args.neumann_variant=richardson \
        trainer.method_args.neumann_steps=2 \
        trainer.method_args.neumann_mu=1.0 \
        trainer.method_args.neumann_alpha_default=0.01 \
        trainer.method_args.neumann_use_probe_alpha=false \
        trainer.method_args.neumann_max_growth_ratio=5.0 \
        trainer.method_args.neuron_traces_path="${TRACES}" \
        trainer.method_args.mask_th_low=1.0 \
        trainer.method_args.mask_th_high=1.0 \
        trainer.method_args.proportional_outer_lr=false \
        'trainer.method_args.outer_freeze_layers=[0,1,2,3,4,5,6,7]' \
        trainer.method_args.mask_freeze_layers=null \
        trainer.method_args.T=20 \
        trainer.method_args.K=10 \
        trainer.method_args.eta_theta=1e-4 \
        trainer.method_args.eta_in=1e-4 \
        trainer.method_args.rho=0.5 \
        trainer.method_args.epsilon=0.1 \
        trainer.method_args.forget_loss_type=logit_margin \
        trainer.method_args.regularization_type=none
fi

# ============================================================
# Exp5: Exp4 + mixed neurons (ratio >= 0.5, proportional LR)
# ============================================================
if [[ "$TARGET_EXP" == "all" || "$TARGET_EXP" == "5" ]]; then
    run_if_needed "Exp5: Mixed Mask" "${SAVE_BASE}/research_exp5_mixed_mask" \
        trainer.args.output_dir="${SAVE_BASE}/research_exp5_mixed_mask" \
        task_name=research_exp5_mixed_mask \
        trainer.method_args.use_sparsity=false \
        trainer.method_args.use_implicit=true \
        trainer.method_args.implicit_solver=neumann \
        trainer.method_args.implicit_blockwise=true \
        'trainer.method_args.implicit_block_skip_layers=[0,1,2,3,4,5,6,7]' \
        trainer.method_args.implicit_block_include_attn=true \
        trainer.method_args.implicit_block_include_mlp=true \
        trainer.method_args.neumann_variant=richardson \
        trainer.method_args.neumann_steps=2 \
        trainer.method_args.neumann_mu=1.0 \
        trainer.method_args.neumann_alpha_default=0.01 \
        trainer.method_args.neumann_use_probe_alpha=false \
        trainer.method_args.neumann_max_growth_ratio=5.0 \
        trainer.method_args.neuron_traces_path="${TRACES}" \
        trainer.method_args.mask_th_low=0.5 \
        trainer.method_args.mask_th_high=1.0 \
        trainer.method_args.proportional_outer_lr=true \
        'trainer.method_args.outer_freeze_layers=[0,1,2,3,4,5,6,7]' \
        trainer.method_args.mask_freeze_layers=null \
        trainer.method_args.T=20 \
        trainer.method_args.K=10 \
        trainer.method_args.eta_theta=1e-4 \
        trainer.method_args.eta_in=1e-4 \
        trainer.method_args.rho=0.5 \
        trainer.method_args.epsilon=0.1 \
        trainer.method_args.forget_loss_type=logit_margin \
        trainer.method_args.regularization_type=none
fi

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
