#!/bin/bash
# U series: Exploiting T8f synergy (log weighting + contrastive inner)
# T8f broke frontier at δ=+0.056 (fk=0.428, rk=0.461) — best frontier break!
# But fk=0.428 is too far from gold (0.328). Need to push fk lower.
#
# Key insight: log weighting + contrastive stabilize each other. Neither works alone.
# The synergy creates L_ret=0.65 at step 40 (constraint satisfied!) — unique among all experiments.
#
# Experiments:
#   U0: T8f + 200 steps (more ALM buildup → will fk drop with more steps?)
#   U1: sqrt + contrastive (sqrt was stable in T8b; try with contrastive for synergy)
#   U2: T8f + npo_beta=1.0 (stronger NPO → push fk harder)
#   U3: T8f + rho=0.05 (slower ALM → more NPO-dominant early phase)
#   U4: T8f + K=5 (more inner correction → better per-step disentanglement)
#   U5: T8f + contrastive γ=1.0 (stronger forget push in inner)
#   U6: Re-run T8d: log + Fisher α=1.0 (config now fixed)
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/U_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting U series (exploit T8f synergy)" | tee "$PROGRESS"

print_metrics() {
    local EVAL_FILE=$1
    "${PYTHON_BIN}" -c "
import json
with open('$EVAL_FILE') as f: d = json.load(f)
def agg(d, k):
    v = d.get(k, {})
    return v.get('agg_value', v) if isinstance(v, dict) else v
fk=float(agg(d,'forget_knowmem_ROUGE')); rk=float(agg(d,'retain_knowmem_ROUGE'))
fv=float(agg(d,'forget_verbmem_ROUGE')); ex=float(agg(d,'extraction_strength'))
pred_rk = 0.55*fk + 0.17
delta = rk - pred_rk
frontier = 'ABOVE' if delta > 0.02 else ('ON' if delta > -0.02 else 'BELOW')
print(f'  fk={fk:.4f} rk={rk:.4f} fv={fv:.4f} ex={ex:.4f} | frontier: {frontier} ({delta:+.3f})')
" 2>/dev/null || echo "  [metrics parse failed]"
}

run_eval() {
    local TASK=$1
    echo "[$(date '+%H:%M:%S')] Evaluating ${TASK}" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/eval.py \
        experiment=eval/muse/default.yaml data_split=${DATA_SPLIT} \
        task_name=${TASK} model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK} \
        "${ATTN_ARGS[@]}" \
        paths.output_dir=saves/unlearn/${TASK}/evals \
        retain_logs_path=${RETAIN_LOGS} \
        2>&1 | tee "${LOG_DIR}/eval_${TASK}.log"
    local EVAL_FILE="saves/unlearn/${TASK}/evals/MUSE_EVAL.json"
    [ -f "$EVAL_FILE" ] && print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"
}

run_sibl() {
    local TASK=$1 EXPERIMENT=$2
    shift 2
    local EXTRA_ARGS=("$@")
    local TASK_DIR="saves/unlearn/${TASK}"
    if compgen -G "${TASK_DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] ${TASK}: weights exist" | tee -a "$PROGRESS"
        local EVAL_FILE="${TASK_DIR}/evals/MUSE_EVAL.json"
        if [ ! -f "$EVAL_FILE" ]; then run_eval ${TASK}
        else echo "[CACHED]" | tee -a "$PROGRESS"; print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"; fi
        return 0
    fi
    echo "[$(date '+%H:%M:%S')] [TRAIN] ${TASK}" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=${EXPERIMENT} model=${MODEL} data_split=${DATA_SPLIT} \
        trainer=SIBL task_name=${TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        "${ATTN_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/train_${TASK}.log"
    local STATUS=$?
    [ $STATUS -ne 0 ] && echo "[WARN] ${TASK} failed (exit $STATUS)" | tee -a "$PROGRESS" && return $STATUS
    run_eval ${TASK}
}

# ── U0: T8f + 200 steps (more ALM to push fk lower) ──────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [U0] T8f config + 200 steps ===" | tee -a "$PROGRESS"
run_sibl ablation_U0_t8f_200 unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=199 \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=0.5 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] U0 failed" | tee -a "$PROGRESS"

# ── U1: sqrt + contrastive (different weighting scheme) ───────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [U1] sqrt weighting + contrastive inner ===" | tee -a "$PROGRESS"
run_sibl ablation_U1_sqrt_contrastive unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.forget_weight_scheme=sqrt \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=0.5 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] U1 failed" | tee -a "$PROGRESS"

# ── U2: T8f + npo_beta=1.0 (stronger NPO to push fk lower) ──────────────
echo "" | tee -a "$PROGRESS"
echo "=== [U2] T8f + npo_beta=1.0 (stronger NPO) ===" | tee -a "$PROGRESS"
run_sibl ablation_U2_stronger_npo unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.npo_beta=1.0 \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=0.5 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] U2 failed" | tee -a "$PROGRESS"

# ── U3: T8f + rho=0.05 (slower ALM for more NPO headroom) ────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [U3] T8f + rho=0.05 (slower ALM) ===" | tee -a "$PROGRESS"
run_sibl ablation_U3_slow_alm unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.rho=0.005 \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=0.5 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] U3 failed" | tee -a "$PROGRESS"

# ── U4: T8f + K=5 (more inner correction per step) ───────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [U4] T8f + K=5 inner steps ===" | tee -a "$PROGRESS"
run_sibl ablation_U4_k5 unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.K=5 \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=0.5 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] U4 failed" | tee -a "$PROGRESS"

# ── U5: T8f + stronger forget push (γ=1.0) ───────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [U5] T8f + contrastive γ=1.0 (stronger push) ===" | tee -a "$PROGRESS"
run_sibl ablation_U5_strong_push unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=1.0 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] U5 failed" | tee -a "$PROGRESS"

# ── U6: T8d rerun: log + Fisher α=1.0 (config now fixed) ─────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [U6] S5 + log weighting + Fisher α=1.0 ===" | tee -a "$PROGRESS"
run_sibl ablation_U6_weighted_fisher unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.use_fisher_weighting=true \
    trainer.method_args.fisher_alpha=1.0 \
    trainer.method_args.fisher_n_samples=64 \
    || echo "[WARN] U6 failed" | tee -a "$PROGRESS"

# ── Summary ──────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] U series DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "  CE frontier: rk ≈ 0.55*fk + 0.17" | tee -a "$PROGRESS"
echo "  T8f baseline (log+contrastive): fk=0.428 rk=0.461 | ABOVE (+0.056)" | tee -a "$PROGRESS"
echo "  S5 baseline (vanilla): fk=0.346 rk=0.382 | ABOVE (+0.022)" | tee -a "$PROGRESS"
for task in ablation_U0_t8f_200 ablation_U1_sqrt_contrastive ablation_U2_stronger_npo ablation_U3_slow_alm ablation_U4_k5 ablation_U5_strong_push ablation_U6_weighted_fisher; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done
