#!/bin/bash
# Run C4 then D series (C3/C5 skipped — logit_margin ceiling confirmed)
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
PROGRESS="/tmp/ablation_C4D_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting C4 -> D series" | tee "$PROGRESS"

print_metrics() {
    local EVAL_FILE=$1
    "${PYTHON_BIN}" -c "
import json
with open('$EVAL_FILE') as f: d = json.load(f)
def agg(d, k):
    v = d.get(k, {})
    return v.get('agg_value', v) if isinstance(v, dict) else v
fk=agg(d,'forget_knowmem_ROUGE'); rk=agg(d,'retain_knowmem_ROUGE')
fv=agg(d,'forget_verbmem_ROUGE'); ex=agg(d,'extraction_strength'); pl=agg(d,'privleak')
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
        echo "[SKIP] ${TASK}" | tee -a "$PROGRESS"
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
        trainer.args.num_train_epochs=3 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        "${ATTN_ARGS[@]}" \
        2>&1 | tee /tmp/train_${TASK}.log
    local STATUS=$?
    [ $STATUS -ne 0 ] && echo "[WARN] ${TASK} failed (exit $STATUS)" | tee -a "$PROGRESS" && return $STATUS
    run_eval ${TASK}
}

# ─── C4: Very aggressive (ε=0.5, K=3, η_θ=3e-4, ρ=0.2) + post_inner=100 ─────
echo "=== [C4] eps=0.5 K=3 eta=3e-4 rho=0.2 + post_inner=100 ===" | tee -a "$PROGRESS"
run_sibl ablation_C4_steep_decay unlearn/muse/ablation_C4_steep_decay \
    || echo "[WARN] C4 failed" | tee -a "$PROGRESS"

# ─── D1: NPO β=2.0 + relaxed ε=0.8 ──────────────────────────────────────────
echo "=== [D1] NPO beta=2.0 + eps=0.8 K=5 rho=0.3 ===" | tee -a "$PROGRESS"
run_sibl ablation_D1_npo_relaxed unlearn/muse/ablation_D1_npo_relaxed \
    || echo "[WARN] D1 failed" | tee -a "$PROGRESS"

# ─── D2: NPO + steering[5,6,7] + ε=0.8  (Exp8r repro) ───────────────────────
echo "=== [D2] NPO + steering[5,6,7] retain_match + eps=0.8 ===" | tee -a "$PROGRESS"
run_sibl ablation_D2_npo_steering unlearn/muse/ablation_D2_npo_steering \
    || echo "[WARN] D2 failed" | tee -a "$PROGRESS"

# ─── D3: logit_margin + steering[5,6,7] + ε=0.8 ─────────────────────────────
echo "=== [D3] logit_margin + steering[5,6,7] retain_match + eps=0.8 ===" | tee -a "$PROGRESS"
run_sibl ablation_D3_logit_steering_relaxed unlearn/muse/ablation_D3_logit_steering_relaxed \
    || echo "[WARN] D3 failed" | tee -a "$PROGRESS"

echo "=== [$(date '+%H:%M:%S')] C4+D DONE ===" | tee -a "$PROGRESS"
