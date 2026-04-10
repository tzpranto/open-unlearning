#!/bin/bash
# T series: Targeted disentanglement — breaking the CE Pareto frontier
# KEY INSIGHT: 8r broke frontier with K/step ratio of 10:1.
# All other experiments used 1:1 or 3:1 → inner loop never converges → rk degrades.
#
# Experiments:
#   T0: K=10, 5 outer steps (8r-like ratio, proper batching)
#   T1: K=5, 10 outer steps (intermediate ratio)
#   T2: Fisher-weighted outer gradient (novel: dampens NPO on retain-important params)
#   T2b: Fisher α=10 (stronger dampening)
#   T3: Contrastive inner loop (push-pull: pull retain, push forget in activation space)
#   T3b: Contrastive + K=3 (stronger inner with contrastive)
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/T_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting T series (targeted disentanglement)" | tee "$PROGRESS"

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

# ── T0: K=10, 5 steps (8r-like K/step ratio, proper accum=32) ──────────────
echo "" | tee -a "$PROGRESS"
echo "=== [T0] K=10, 5 outer steps × accum=32 (8r-like ratio, 160 samples) ===" | tee -a "$PROGRESS"
run_sibl ablation_T0_high_K unlearn/muse/ablation_T0_high_K \
    trainer.args.gradient_accumulation_steps=32 \
    trainer.method_args.debug_stop_after_outer=4 \
    || echo "[WARN] T0 failed" | tee -a "$PROGRESS"

# ── T1: K=5, 10 steps (intermediate ratio) ─────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [T1] K=5, 10 outer steps × accum=32 (intermediate ratio) ===" | tee -a "$PROGRESS"
run_sibl ablation_T1_mid_K unlearn/muse/ablation_T0_high_K \
    trainer.args.gradient_accumulation_steps=32 \
    trainer.method_args.K=5 \
    trainer.method_args.debug_stop_after_outer=9 \
    || echo "[WARN] T1 failed" | tee -a "$PROGRESS"

# ── T2: Fisher-weighted outer gradient (α=1.0) ─────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [T2] Fisher-weighted outer (α=1.0, G1 base) ===" | tee -a "$PROGRESS"
run_sibl ablation_T2_fisher unlearn/muse/ablation_T2_fisher \
    trainer.args.gradient_accumulation_steps=32 \
    || echo "[WARN] T2 failed" | tee -a "$PROGRESS"

# ── T2b: Fisher-weighted outer gradient (α=10.0, stronger dampening) ────────
echo "" | tee -a "$PROGRESS"
echo "=== [T2b] Fisher-weighted outer (α=10.0, stronger dampening) ===" | tee -a "$PROGRESS"
run_sibl ablation_T2b_fisher_strong unlearn/muse/ablation_T2_fisher \
    trainer.args.gradient_accumulation_steps=32 \
    trainer.method_args.fisher_alpha=10.0 \
    || echo "[WARN] T2b failed" | tee -a "$PROGRESS"

# ── T3: Contrastive inner (β=1.0 pull, γ=0.5 push) ────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [T3] Contrastive inner (β=1.0 pull retain, γ=0.5 push forget) ===" | tee -a "$PROGRESS"
run_sibl ablation_T3_contrastive unlearn/muse/ablation_T3_contrastive \
    trainer.args.gradient_accumulation_steps=32 \
    || echo "[WARN] T3 failed" | tee -a "$PROGRESS"

# ── T3b: Contrastive inner + K=3 (stronger inner correction) ───────────────
echo "" | tee -a "$PROGRESS"
echo "=== [T3b] Contrastive inner + K=3 (more inner correction + disentanglement) ===" | tee -a "$PROGRESS"
run_sibl ablation_T3b_contrastive_K3 unlearn/muse/ablation_T3_contrastive \
    trainer.args.gradient_accumulation_steps=32 \
    trainer.method_args.K=3 \
    || echo "[WARN] T3b failed" | tee -a "$PROGRESS"

# ── T5: S5 config + Fisher weighting (dampen early NPO damage) ──────────────
echo "" | tee -a "$PROGRESS"
echo "=== [T5] S5 (accum=1,K=3,100 steps) + Fisher α=1.0 ===" | tee -a "$PROGRESS"
run_sibl ablation_T5_s5_fisher unlearn/muse/ablation_T2_fisher \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.K=3 \
    trainer.method_args.rho=0.01 \
    trainer.method_args.debug_stop_after_outer=99 \
    || echo "[WARN] T5 failed" | tee -a "$PROGRESS"

# ── T6: S5 config + contrastive inner (active disentanglement) ──────────────
echo "" | tee -a "$PROGRESS"
echo "=== [T6] S5 (accum=1,K=3,100 steps) + contrastive inner ===" | tee -a "$PROGRESS"
run_sibl ablation_T6_s5_contrastive unlearn/muse/ablation_T3_contrastive \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.K=3 \
    trainer.method_args.rho=0.01 \
    trainer.method_args.debug_stop_after_outer=99 \
    || echo "[WARN] T6 failed" | tee -a "$PROGRESS"

# ── T7: S5 config + Fisher + contrastive (kitchen sink on S5 base) ──────────
echo "" | tee -a "$PROGRESS"
echo "=== [T7] S5 + Fisher α=1.0 + contrastive inner ===" | tee -a "$PROGRESS"
run_sibl ablation_T7_s5_full unlearn/muse/ablation_T3_contrastive \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.K=3 \
    trainer.method_args.rho=0.01 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.use_fisher_weighting=true \
    trainer.method_args.fisher_alpha=1.0 \
    trainer.method_args.fisher_n_samples=64 \
    || echo "[WARN] T7 failed" | tee -a "$PROGRESS"

# ── Summary ──────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] T series DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "  CE frontier: rk ≈ 0.55*fk + 0.17 | G1: (0.274, 0.327) ON | S5: (0.346, 0.382) ABOVE" | tee -a "$PROGRESS"
for task in ablation_T0_high_K ablation_T1_mid_K ablation_T2_fisher ablation_T2b_fisher_strong ablation_T3_contrastive ablation_T3b_contrastive_K3 ablation_T5_s5_fisher ablation_T6_s5_contrastive ablation_T7_s5_full; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done
