#!/bin/bash
# T8 series: Score-weighted forget sampling experiments
# Builds on S5's frontier break (fk=0.346, rk=0.382, δ=+0.022)
# Adds memorization-score-weighted sampling to preferentially target hard-to-forget samples.
#
# Weighting schemes tested:
#   T8:  S5 + log weighting, 100 steps (baseline weighted)
#   T8b: S5 + sqrt weighting, 100 steps (more aggressive)
#   T8c: S5 + log weighting, 75 steps (optimistic — does frontier break earlier with targeting?)
#   T8d: S5 + log weighting + Fisher α=1.0 (weighted sampling + Fisher dampening)
#   T8e: G1 + log weighting (does weighting help with accum=32 too?)
#   T8f: S5 + log weighting + contrastive inner (everything combined)
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/T8_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting T8 series (score-weighted forget sampling)" | tee "$PROGRESS"

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

# ── T8: S5 + log weighting, 100 steps ──────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [T8] S5 + log-weighted sampling (100 steps, accum=1, K=3) ===" | tee -a "$PROGRESS"
run_sibl ablation_T8_weighted_s5 unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    || echo "[WARN] T8 failed" | tee -a "$PROGRESS"

# ── T8b: S5 + sqrt weighting, 100 steps (more aggressive) ──────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [T8b] S5 + sqrt-weighted sampling (100 steps) ===" | tee -a "$PROGRESS"
run_sibl ablation_T8b_weighted_sqrt unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.forget_weight_scheme=sqrt \
    trainer.method_args.debug_stop_after_outer=99 \
    || echo "[WARN] T8b failed" | tee -a "$PROGRESS"

# ── T8c: S5 + log weighting, 75 steps (earlier frontier break?) ────────────
echo "" | tee -a "$PROGRESS"
echo "=== [T8c] S5 + log-weighted, 75 steps (optimal early stopping?) ===" | tee -a "$PROGRESS"
run_sibl ablation_T8c_weighted_75 unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=74 \
    || echo "[WARN] T8c failed" | tee -a "$PROGRESS"

# ── T8d: S5 + log weighting + Fisher (weighted + dampened) ──────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [T8d] S5 + log-weighted + Fisher α=1.0 ===" | tee -a "$PROGRESS"
run_sibl ablation_T8d_weighted_fisher unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.use_fisher_weighting=true \
    trainer.method_args.fisher_alpha=1.0 \
    trainer.method_args.fisher_n_samples=64 \
    || echo "[WARN] T8d failed" | tee -a "$PROGRESS"

# ── T8e: G1 + log weighting (does weighting help with accum=32?) ────────────
echo "" | tee -a "$PROGRESS"
echo "=== [T8e] G1 + log-weighted sampling (accum=32, K=1, ρ=0.1) ===" | tee -a "$PROGRESS"
run_sibl ablation_T8e_weighted_g1 unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=32 \
    trainer.method_args.K=1 \
    trainer.method_args.rho=0.1 \
    || echo "[WARN] T8e failed" | tee -a "$PROGRESS"

# ── T8f: S5 + log weighting + contrastive (everything) ─────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [T8f] S5 + log-weighted + contrastive inner ===" | tee -a "$PROGRESS"
run_sibl ablation_T8f_weighted_contrastive unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=0.5 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] T8f failed" | tee -a "$PROGRESS"

# ── Summary ──────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] T8 series DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "  CE frontier: rk ≈ 0.55*fk + 0.17" | tee -a "$PROGRESS"
echo "  S5 baseline (no weighting): fk=0.346 rk=0.382 | ABOVE (+0.022)" | tee -a "$PROGRESS"
for task in ablation_T8_weighted_s5 ablation_T8b_weighted_sqrt ablation_T8c_weighted_75 ablation_T8d_weighted_fisher ablation_T8e_weighted_g1 ablation_T8f_weighted_contrastive; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done
