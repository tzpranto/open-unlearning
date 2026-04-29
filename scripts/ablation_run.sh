#!/bin/bash
# LoRA-BiAL Ablation Runner — TOFU forget01, 1B, single seed
# Usage:
#   nohup bash scripts/ablation_run.sh > saves/unlearn/ablation_run.log 2>&1 &
#   SEEDS="42 123 456" bash scripts/ablation_run.sh   # multi-seed
#   TIERS="1" bash scripts/ablation_run.sh             # tier 1 only
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

# ── Config ──────────────────────────────────────────────────
SEEDS="${SEEDS:-42}"
TIERS="${TIERS:-1 2 3}"
MODEL="Llama-3.2-1B-Instruct"
FORGET="forget01"
RETAIN="retain99"
HOLDOUT="holdout01"
RETAIN_LOGS="saves/eval/tofu_${MODEL}_${RETAIN}/TOFU_EVAL.json"

# ── Common Hydra overrides (matches A0) ─────────────────────
BASE_OVERRIDES=(
    experiment=unlearn/tofu/lora_bial_1b.yaml
    trainer=LoRABiALAdaptive
    model=${MODEL}
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${MODEL}_full
    forget_split=${FORGET}
    retain_split=${RETAIN}
    holdout_split=${HOLDOUT}
    trainer.args.eval_strategy=no
    trainer.args.do_eval=false
    trainer.args.eval_on_start=false
    trainer.method_args.checkpoint_every_epoch=false
    trainer.method_args.eta_theta=5e-5
    trainer.method_args.T=250
    trainer.method_args.epsilon_multiplier=0.85
    retain_logs_path=${RETAIN_LOGS}
)

# ── Ablation definitions: ID|TIER|LABEL|OVERRIDES ──────────
# A0 is already done — skip it.
ABLATIONS=(
    # GROUP 1: Bilevel
    "A1|1|K0_no_inner|trainer.method_args.K=0 trainer.method_args.inner_warmup_steps=999999"
    "A2|2|K1_minimal|trainer.method_args.K=1"
    "A3|3|K10_excessive|trainer.method_args.K=10"
    # GROUP 2: Forget loss
    "A4|1|GA_forget|trainer.method_args.forget_loss_type=ga"
    "A5|1|NPO_forget|trainer.method_args.forget_loss_type=npo trainer.method_args.npo_beta=4.0"
    "A6|2|tau_0.3|trainer.method_args.clamped_entropy_tau=0.3"
    "A7|2|tau_0.5|trainer.method_args.clamped_entropy_tau=0.5"
    "A8|2|tau_0.9|trainer.method_args.clamped_entropy_tau=0.9"
    "A9|1|tau_1.0_unclamped|trainer.method_args.clamped_entropy_tau=1.0"
    # GROUP 3: ALM + dual
    "A10|1|fixed_lambda_1.0|trainer.method_args.lambda_init=1.0 trainer.method_args.lambda_min=1.0 trainer.method_args.lambda_max=1.0"
    "A11|2|fixed_lambda_5.0|trainer.method_args.lambda_init=5.0 trainer.method_args.lambda_min=5.0 trainer.method_args.lambda_max=5.0"
    "A12|1|rho_0_no_penalty|trainer.method_args.rho=0.0"
    "A13|2|rho_1.0_aggressive|trainer.method_args.rho=1.0"
    "A14|1|symmetric_dual|trainer.method_args.dual_decay_factor=1.0"
    # GROUP 4: Epsilon sensitivity
    "A15|1|eps_mul_0.75|trainer.method_args.epsilon_multiplier=0.75"
    "A16|1|eps_mul_1.0|trainer.method_args.epsilon_multiplier=1.0"
    "A17|2|eps_mul_1.3|trainer.method_args.epsilon_multiplier=1.3"
    "A18|2|eps_mul_1.5|trainer.method_args.epsilon_multiplier=1.5"
    "A19|1|eps_mul_2.0|trainer.method_args.epsilon_multiplier=2.0"
    "A20|2|eps_mul_3.0|trainer.method_args.epsilon_multiplier=3.0"
    # GROUP 5: Structural
    "A22|2|sanity_GA_no_bilevel_no_ALM|trainer.method_args.forget_loss_type=ga trainer.method_args.K=0 trainer.method_args.inner_warmup_steps=999999 trainer.method_args.rho=0.0 trainer.method_args.lambda_init=0.0 trainer.method_args.lambda_min=0.0"
)

# ── Runner ──────────────────────────────────────────────────
TOTAL=0
DONE=0
SKIP=0
FAIL=0

for entry in "${ABLATIONS[@]}"; do
    IFS='|' read -r aid tier label overrides <<< "$entry"

    # Tier filter
    if [[ ! " $TIERS " =~ " $tier " ]]; then
        continue
    fi

    for seed in $SEEDS; do
        TOTAL=$((TOTAL+1))
        TASK="ablation_${aid}_${label}_s${seed}"
        OUTDIR="saves/unlearn/${TASK}"
        EVAL_DIR="${OUTDIR}/evals"

        # Skip if already evaluated
        if [[ -f "${EVAL_DIR}/TOFU_SUMMARY.json" ]]; then
            echo "[SKIP] ${TASK}"
            SKIP=$((SKIP+1))
            continue
        fi

        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "[TRAIN] $(date) ${TASK}"
        echo "  ${aid} | tier=${tier} | ${overrides}"
        echo "════════════════════════════════════════════════════════════"

        # Build override array
        IFS=' ' read -ra EXTRA <<< "$overrides"

        # Train
        if ! CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
            "${BASE_OVERRIDES[@]}" \
            task_name=${TASK} \
            trainer.args.seed=${seed} \
            "${EXTRA[@]}"; then
            echo "[TRAIN FAILED] ${TASK}"
            FAIL=$((FAIL+1))
            continue
        fi

        # Eval
        echo "[EVAL] $(date) ${TASK}"
        if ! CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/tofu/default.yaml \
            task_name=${TASK} \
            model=${MODEL} \
            model.model_args.pretrained_model_name_or_path=${OUTDIR} \
            paths.output_dir=${EVAL_DIR} \
            forget_split=${FORGET} \
            holdout_split=${HOLDOUT} \
            retain_logs_path=${RETAIN_LOGS}; then
            echo "[EVAL FAILED] ${TASK}"
            FAIL=$((FAIL+1))
            continue
        fi

        echo "[DONE] $(date) ${TASK}"
        cat "${EVAL_DIR}/TOFU_SUMMARY.json" 2>/dev/null || true

        # Clean weights to save space
        rm -f "$OUTDIR"/model*.safetensors "$OUTDIR"/model.safetensors.index.json 2>/dev/null
        rm -f "$OUTDIR"/pytorch_model* "$OUTDIR"/config.json "$OUTDIR"/generation_config* 2>/dev/null
        rm -f "$OUTDIR"/tokenizer* "$OUTDIR"/special_tokens* "$OUTDIR"/added_tokens* 2>/dev/null
        rm -f "$OUTDIR"/optimizer* "$OUTDIR"/scheduler* "$OUTDIR"/training_args* 2>/dev/null
        rm -rf "$OUTDIR"/checkpoint-* 2>/dev/null
        find "$OUTDIR" -maxdepth 1 -name "*.safetensors" -delete 2>/dev/null
        find "$OUTDIR" -maxdepth 1 -name "*.bin" -delete 2>/dev/null

        DONE=$((DONE+1))
    done
done

echo ""
echo "════════════════════════════════════════"
echo "ALL DONE: total=${TOTAL} done=${DONE} skip=${SKIP} fail=${FAIL}"
echo "════════════════════════════════════════"

# ── Summary table ───────────────────────────────────────────
echo ""
echo "ABLATION RESULTS:"
echo "ID | Label | Seed | forget_quality | model_utility | fgt_ROUGE | fgt_Prob"
echo "---|-------|------|----------------|---------------|-----------|--------"
for entry in "${ABLATIONS[@]}"; do
    IFS='|' read -r aid tier label overrides <<< "$entry"
    if [[ ! " $TIERS " =~ " $tier " ]]; then continue; fi
    for seed in $SEEDS; do
        TASK="ablation_${aid}_${label}_s${seed}"
        summary="saves/unlearn/${TASK}/evals/TOFU_SUMMARY.json"
        if [[ -f "$summary" ]]; then
            python3 -c "
import json
with open('$summary') as f:
    d = json.load(f)
print(f'${aid} | ${label} | ${seed} | {d[\"forget_quality\"]:.4f} | {d[\"model_utility\"]:.4f} | {d.get(\"forget_Q_A_ROUGE\",0):.4f} | {d.get(\"forget_Q_A_Prob\",0):.6f}')
" 2>/dev/null || true
        fi
    done
done

# Also include A0 reference
echo "--- A0 reference ---"
for seed in $SEEDS; do
    summary="saves/unlearn/adaptive_${MODEL}_${FORGET}_s${seed}/evals/TOFU_SUMMARY.json"
    if [[ -f "$summary" ]]; then
        python3 -c "
import json
with open('$summary') as f:
    d = json.load(f)
print(f'A0 | full_method | ${seed} | {d[\"forget_quality\"]:.4f} | {d[\"model_utility\"]:.4f} | {d.get(\"forget_Q_A_ROUGE\",0):.4f} | {d.get(\"forget_Q_A_Prob\",0):.6f}')
" 2>/dev/null || true
    fi
done
