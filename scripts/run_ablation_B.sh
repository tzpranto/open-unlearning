#!/bin/bash
# Ablation Series B: SIBL experiments on MUSE News / Llama-2-7b
# NOTE: B1/B2 gradient projection skipped — projection hurts for MUSE News (85% domain overlap,
#   forget/retain gradients too aligned → projection removes forget signal entirely)
# NOTE: B4 steering skipped (secondary; test after ALM hyperparams fixed in C series)
#
# Running experiments:
#   B5: logit_margin + post_unlearn_inner_steps=50  (retention recovery)
#   B6: pdu (squared margin) loss  (stronger shift function)
#   B7: proportional_outer_lr (ratio as LR scale, soft mask)
#   B8: retain_protection(layer31, 3x) + inner_repr_anchor(layer31)
#   B10: grad_ascent + gradproj(layer, rescale)  [stronger forget signal]
#
# Date: 2026-04-08
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
PROGRESS="/tmp/ablation_B_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting Ablation Series B (revised)" | tee "$PROGRESS"
echo "Gold: fk<=0.328 rk>=0.560 fv<=0.202 ex<=0.024 | A1a: fk=0.524 rk=0.500" | tee -a "$PROGRESS"

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
pl=agg(d,'privleak')
print(f'  fk={fk:.4f} rk={rk:.4f} fv={fv:.4f} ex={ex:.4f} pl={pl:.2f}')
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
        2>&1 | tee /tmp/eval_${TASK}.log
    local EVAL_FILE="saves/unlearn/${TASK}/evals/MUSE_EVAL.json"
    [ -f "$EVAL_FILE" ] && print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"
}

run_sibl() {
    local TASK=$1 EXPERIMENT=$2
    local TASK_DIR="saves/unlearn/${TASK}"

    if compgen -G "${TASK_DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] ${TASK}: weights exist" | tee -a "$PROGRESS"
        local EVAL_FILE="${TASK_DIR}/evals/MUSE_EVAL.json"
        if [ ! -f "$EVAL_FILE" ]; then
            run_eval ${TASK}
        else
            echo "[CACHED]" | tee -a "$PROGRESS"
            print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"
        fi
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] [TRAIN] ${TASK}" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=${EXPERIMENT} model=${MODEL} data_split=${DATA_SPLIT} \
        trainer=SIBL task_name=${TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_accumulation_steps=32 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.num_train_epochs=3 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        "${ATTN_ARGS[@]}" \
        2>&1 | tee /tmp/train_${TASK}.log

    local STATUS=$?
    [ $STATUS -ne 0 ] && echo "[WARN] ${TASK} failed (exit $STATUS)" | tee -a "$PROGRESS" && return $STATUS
    run_eval ${TASK}
}

# ─── B5: logit_margin + post_unlearn_inner_steps=50 ──────────────────────────
echo "=== [B5] logit_margin + post_inner_steps=50 ===" | tee -a "$PROGRESS"
run_sibl ablation_B5_logit_postinner unlearn/muse/ablation_B5_logit_postinner \
    || echo "[WARN] B5 failed" | tee -a "$PROGRESS"

# ─── B6: PDU squared margin loss ─────────────────────────────────────────────
echo "=== [B6] pdu (squared margin shift) ===" | tee -a "$PROGRESS"
run_sibl ablation_B6_pdu unlearn/muse/ablation_B6_pdu \
    || echo "[WARN] B6 failed" | tee -a "$PROGRESS"

# ─── B7: proportional outer LR (soft mask, ratio as LR scale) ────────────────
echo "=== [B7] proportional_outer_lr (ratio as LR scale, th_low=1.0) ===" | tee -a "$PROGRESS"
run_sibl ablation_B7_proportional_lr unlearn/muse/ablation_B7_proportional_lr \
    || echo "[WARN] B7 failed" | tee -a "$PROGRESS"

# ─── B8: retain_protection(layer31, 3x) + inner_repr_anchor(layer31) ──────────
echo "=== [B8] retain_protection(layer31, 3x) + inner_repr_anchor(layer31) ===" | tee -a "$PROGRESS"
run_sibl ablation_B8_retain_anchor unlearn/muse/ablation_B8_retain_anchor \
    || echo "[WARN] B8 failed" | tee -a "$PROGRESS"

# ─── B10: grad_ascent + gradproj(layer, rescale) ─────────────────────────────
echo "=== [B10] grad_ascent + gradproj(layer, rescale) ===" | tee -a "$PROGRESS"
run_sibl ablation_B10_grad_ascent unlearn/muse/ablation_B10_grad_ascent \
    || echo "[WARN] B10 failed" | tee -a "$PROGRESS"

echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] Ablation B DONE ===" | tee -a "$PROGRESS"
echo "Progress: $PROGRESS"
echo "Results: saves/unlearn/ablation_B*/evals/MUSE_EVAL.json"
