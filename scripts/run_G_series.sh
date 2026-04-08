#!/bin/bash
# G series: Build on F0 (fk=0.325 beats gold) — recover retain gradually
# G0: F0 + post_inner=100 (CE retain recovery after NPO)
# G1: F0 + weak steering coeff=5 (gentle retain anchor)
# G2: F0 + K=5 (more inner retain steps)
# G3: Best G + next component (defined after seeing G0/G1/G2 results)
# Rule: keep component only if rk↑ without fk crossing 0.340
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/G_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting G series (build on F3 anchor)" | tee "$PROGRESS"
echo "F3 anchor: fk=0.325 rk=0.316 eps=0.70 | Target: keep fk<0.340, push rk toward 0.560" | tee -a "$PROGRESS"
echo "F3 is anchor (strictly better than F0: same fk, rk +0.026 from tighter eps=0.70)" | tee -a "$PROGRESS"

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
fk_verdict = 'OK' if fk<=0.340 else 'HURT'
rk_delta = rk - 0.316
print(f'  fk={fk:.4f}[{fk_verdict}] rk={rk:.4f}({rk_delta:+.3f} vs F3) fv={fv:.4f} ex={ex:.4f}')
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

# ── G0: F0 + post_inner=100 ───────────────────────────────────────────────────
echo "=== [G0] F0 + post_inner=100 CE retain recovery ===" | tee -a "$PROGRESS"
run_sibl ablation_G0_npo_postinner unlearn/muse/ablation_G0_npo_postinner \
    || echo "[WARN] G0 failed" | tee -a "$PROGRESS"

# ── G1: F0 + weak steering coeff=5 ────────────────────────────────────────────
echo "=== [G1] F0 + steering coeff=5 (gentle, vs coeff=20 which hurt forget) ===" | tee -a "$PROGRESS"
run_sibl ablation_G1_npo_weak_steering unlearn/muse/ablation_G1_npo_weak_steering \
    || echo "[WARN] G1 failed" | tee -a "$PROGRESS"

# ── G2: F0 + K=5 inner steps ─────────────────────────────────────────────────
echo "=== [G2] F0 + K=5 inner retain steps (vs K=1) ===" | tee -a "$PROGRESS"
run_sibl ablation_G2_npo_more_inner unlearn/muse/ablation_G2_npo_more_inner \
    || echo "[WARN] G2 failed" | tee -a "$PROGRESS"

echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] G0/G1/G2 DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"

# ── G3: F3 + post_inner=100 + neuron bitmap mask (post_inner_retain_only) ────
echo "=== [G3] F3 + masked post_inner=100 (freeze forget neurons during CE recovery) ===" | tee -a "$PROGRESS"
run_sibl ablation_G3_npo_masked_postinner unlearn/muse/ablation_G3_npo_masked_postinner \
    || echo "[WARN] G3 failed" | tee -a "$PROGRESS"

echo "" | tee -a "$PROGRESS"
echo "=== FULL SUMMARY (F3 anchor: fk=0.325, rk=0.316) ===" | tee -a "$PROGRESS"
for task in ablation_F3_npo_tight_eps ablation_G0_npo_postinner ablation_G1_npo_weak_steering ablation_G2_npo_more_inner ablation_G3_npo_masked_postinner; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done

# ── Auto-chain: memorization scoring ─────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "[G->mem] Running memorization scorer on forget set..." | tee -a "$PROGRESS"
LOCAL_LLAMA="/datadrive/caches/huggingface/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
"${PYTHON_BIN}" scripts/score_forget_memorization.py \
    --base_model "${LOCAL_LLAMA}" \
    --top_k 50 \
    --hard_forget_path data/hard_forget_news.jsonl \
    2>&1 | tee "${LOG_DIR}/memorization_scoring.log" \
    && echo "[G->mem] Scoring done. Top-50 saved to data/hard_forget_news.jsonl" | tee -a "$PROGRESS" \
    || echo "[WARN] Memorization scoring failed — check ${LOG_DIR}/memorization_scoring.log" | tee -a "$PROGRESS"

# ── Auto-chain: H series (created after G results become available) ───────────
H_SCRIPT="/datadrive/forked/open-unlearning/scripts/run_H_series.sh"
if [ -f "$H_SCRIPT" ]; then
    echo "[G->H] Launching H series..." | tee -a "$PROGRESS"
    bash "$H_SCRIPT"
else
    echo "[G->H] H series script not yet created — stopping here." | tee -a "$PROGRESS"
    echo "       Create run_H_series.sh once G results are reviewed." | tee -a "$PROGRESS"
fi
