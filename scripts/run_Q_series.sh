#!/bin/bash
# Q series: Two-phase ALM — NPO-heavy phase then λ boost for retain recovery
# KEY INSIGHT: P0/P1 showed strong ALM gives rk=0.465 but fk=0.524.
# G1 gets fk=0.274 with weak ALM. The fix: two phases.
# Phase 1: Weak ALM (λ=0, ρ=0.1) — aggressive NPO drives fk down (like G1)
# Phase 2: Strong ALM (λ→5.0, ρ→1.0) — retain recovery with fk headroom
#
# Experiments:
#   Q0: Boost at step 10 (fk~0.275 at boost, 15 steps for retain recovery)
#   Q1: Boost at step 5 (fk~0.246, more recovery steps, more rk headroom to fill)
#   Q2: Q0 + K=2 inner + eta_in=3e-4 (stronger inner during both phases)
#
# Anchors: P0 fk=0.524 rk=0.465 | G1 fk=0.274 rk=0.327 | Gold: fk<=0.328 rk>=0.560
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/Q_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting Q series (Two-phase ALM)" | tee "$PROGRESS"
echo "Anchors: P0 fk=0.524 rk=0.465 | G1 fk=0.274 rk=0.327 | Gold: fk<=0.328 rk>=0.560" | tee -a "$PROGRESS"

print_metrics() {
    local EVAL_FILE=$1
    "${PYTHON_BIN}" -c "
import json
with open('$EVAL_FILE') as f: d = json.load(f)
def agg(d, k):
    v = d.get(k, {})
    return v.get('agg_value', v) if isinstance(v, dict) else v
fk=agg(d,'forget_knowmem_ROUGE'); rk=agg(d,'retain_knowmem_ROUGE')
fv=agg(d,'forget_verbmem_ROUGE'); ex=agg(d,'extraction_strength')
fk_gold = 'GOLD' if fk<=0.328 else ('OK' if fk<=0.340 else 'HURT')
rk_gold = 'GOLD' if rk>=0.560 else f'{rk:.3f}'
fk_delta = fk - 0.274
rk_delta = rk - 0.327
print(f'  fk={fk:.4f}[{fk_gold}]({fk_delta:+.3f} vs G1) rk={rk:.4f}[{rk_gold}]({rk_delta:+.3f} vs G1) fv={fv:.4f} ex={ex:.4f}')
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
        trainer.args.gradient_accumulation_steps=32 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        "${ATTN_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/train_${TASK}.log"
    local STATUS=$?
    [ $STATUS -ne 0 ] && echo "[WARN] ${TASK} failed (exit $STATUS)" | tee -a "$PROGRESS" && return $STATUS
    run_eval ${TASK}
}

# ── Q0: Two-phase boost at step 10 (MOST PROMISING) ──────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [Q0] Two-phase: NPO steps 0-9, then λ=5 ρ=1 steps 10-24 ===" | tee -a "$PROGRESS"
echo "  Tests: G1-like NPO for 10 steps → strong ALM for 15 steps" | tee -a "$PROGRESS"
run_sibl ablation_Q0_twophase_s10 unlearn/muse/ablation_Q0_twophase_s10 \
    || echo "[WARN] Q0 failed" | tee -a "$PROGRESS"

# ── Q1: Two-phase boost at step 5 (earlier transition) ───────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [Q1] Two-phase: NPO steps 0-4, then λ=5 ρ=1 steps 5-24 ===" | tee -a "$PROGRESS"
echo "  Tests: 5 NPO steps (fk~0.246) → 20 steps retain recovery" | tee -a "$PROGRESS"
run_sibl ablation_Q1_twophase_s5 unlearn/muse/ablation_Q1_twophase_s5 \
    || echo "[WARN] Q1 failed" | tee -a "$PROGRESS"

# ── Q2: Q0 + K=2 inner + higher inner LR ─────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [Q2] Two-phase s10 + K=2 inner + eta_in=3e-4 ===" | tee -a "$PROGRESS"
echo "  Tests: stronger inner loop throughout + two-phase ALM" | tee -a "$PROGRESS"
run_sibl ablation_Q2_twophase_s10_K2 unlearn/muse/ablation_Q2_twophase_s10_K2 \
    || echo "[WARN] Q2 failed" | tee -a "$PROGRESS"

# ── Summary ──────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] Q series DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "  Gold targets: fk<=0.328, rk>=0.560" | tee -a "$PROGRESS"
echo "  Anchor G1: fk=0.274 rk=0.327 | P0: fk=0.524 rk=0.465" | tee -a "$PROGRESS"
for task in ablation_Q0_twophase_s10 ablation_Q1_twophase_s5 ablation_Q2_twophase_s10_K2; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done
