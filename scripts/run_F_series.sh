#!/bin/bash
# F series: Strip-and-build approach
# F0: Bare NPO (T=10, no retain components) — establish min fk
# F1: F0 + steering
# F2: F0 + implicit
# F3: F0 + tighter epsilon
# F4: F0 + T=25 (1 full epoch)
# Each run is ~5 min. Add components only if rk improves without fk degrading >0.02
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/F_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting F series (strip-and-build)" | tee "$PROGRESS"
echo "Gold: fk<=0.328 rk>=0.560 | Best seen: fk=0.286 (broken pipeline)" | tee -a "$PROGRESS"
echo "Protocol: F0=bare forget baseline. Add components only if rk↑ without fk↑>0.02" | tee -a "$PROGRESS"

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
beat_fk = 'GOOD' if fk<=0.40 else 'BAD'
print(f'  fk={fk:.4f}[{beat_fk}] rk={rk:.4f} fv={fv:.4f} ex={ex:.4f} pl={pl:.2f}')
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

# ── F0: Bare NPO T=10, ε=0.95, no implicit, no steering ─────────────────────
echo "=== [F0] Bare NPO: T=10, eps=0.95, K=1, no implicit, no steering ===" | tee -a "$PROGRESS"
run_sibl ablation_F0_npo_bare unlearn/muse/ablation_F0_npo_bare \
    || echo "[WARN] F0 failed" | tee -a "$PROGRESS"

# ── F1: F0 + steering[5,6,7] ─────────────────────────────────────────────────
echo "=== [F1] F0 + steering[5,6,7] retain_match ===" | tee -a "$PROGRESS"
run_sibl ablation_F1_npo_steering unlearn/muse/ablation_F1_npo_steering \
    || echo "[WARN] F1 failed" | tee -a "$PROGRESS"

# ── F2: F0 + implicit correction ─────────────────────────────────────────────
echo "=== [F2] F0 + implicit (Neumann) ===" | tee -a "$PROGRESS"
run_sibl ablation_F2_npo_implicit unlearn/muse/ablation_F2_npo_implicit \
    || echo "[WARN] F2 failed" | tee -a "$PROGRESS"

# ── F3: F0 + tighter epsilon ─────────────────────────────────────────────────
echo "=== [F3] F0 + epsilon=0.70 (tighter retain constraint) ===" | tee -a "$PROGRESS"
run_sibl ablation_F3_npo_tight_eps unlearn/muse/ablation_F3_npo_tight_eps \
    || echo "[WARN] F3 failed" | tee -a "$PROGRESS"

# ── F4: F0 with T=25 (1 full epoch) ──────────────────────────────────────────
echo "=== [F4] F0 but T=25 (1 full epoch) — more steps better or worse? ===" | tee -a "$PROGRESS"
run_sibl ablation_F4_npo_25steps unlearn/muse/ablation_F4_npo_25steps \
    || echo "[WARN] F4 failed" | tee -a "$PROGRESS"

echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] F series DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "(After F series, run memorization scorer to guide trace-backed sample selection:)" | tee -a "$PROGRESS"
echo "  python scripts/score_forget_memorization.py --top_k 50 --hard_forget_path data/hard_forget_news.jsonl" | tee -a "$PROGRESS"
for task in ablation_F0_npo_bare ablation_F1_npo_steering ablation_F2_npo_implicit ablation_F3_npo_tight_eps ablation_F4_npo_25steps; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done

# Auto-chain G series
echo '[F->G] Launching G series...' | tee -a "$PROGRESS"
bash /datadrive/forked/open-unlearning/scripts/run_G_series.sh
