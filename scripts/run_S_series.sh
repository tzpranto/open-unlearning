#!/bin/bash
# S series: Small-batch NPO — recreating 8r's accidental frontier-breaking success
# KEY INSIGHT: 8r (1 sample/step, K=10 inner) gave fk=0.371, rk=0.417 — ABOVE CE frontier.
# The CE frontier predicts rk=0.374 at fk=0.371. 8r got rk=0.417 (+0.043 above).
# Why: smaller batch → more targeted NPO → inner loop precisely compensates each perturbation.
# With accum=32 (G1): inner loop tries to fix averaged damage from 32 samples — too hard.
#
# Experiments:
#   S0: accum=4, K=1, ρ=0.1 (200 steps/epoch — 8x more frequent inner corrections)
#   S1: accum=1, K=3, ρ=0.01 (800 steps — closest to 8r: 1 sample + strong inner)
#   S2: accum=4, K=1 + inverted inner + full outer (S0 + N4 architecture)
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/S_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting S series (small-batch NPO — frontier breaking)" | tee "$PROGRESS"
echo "Frontier: rk ≈ 0.55*fk + 0.17 | 8r ref: (0.371, 0.417) | G1: (0.274, 0.327)" | tee -a "$PROGRESS"

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
# Check if above CE frontier: predicted_rk = 0.55*fk + 0.17
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
    local TASK=$1 EXPERIMENT=$2 ACCUM=$3
    local TASK_DIR="saves/unlearn/${TASK}"
    if compgen -G "${TASK_DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] ${TASK}: weights exist" | tee -a "$PROGRESS"
        local EVAL_FILE="${TASK_DIR}/evals/MUSE_EVAL.json"
        if [ ! -f "$EVAL_FILE" ]; then run_eval ${TASK}
        else echo "[CACHED]" | tee -a "$PROGRESS"; print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"; fi
        return 0
    fi
    echo "[$(date '+%H:%M:%S')] [TRAIN] ${TASK} (accum=${ACCUM})" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=${EXPERIMENT} model=${MODEL} data_split=${DATA_SPLIT} \
        trainer=SIBL task_name=${TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_accumulation_steps=${ACCUM} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        "${ATTN_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/train_${TASK}.log"
    local STATUS=$?
    [ $STATUS -ne 0 ] && echo "[WARN] ${TASK} failed (exit $STATUS)" | tee -a "$PROGRESS" && return $STATUS
    run_eval ${TASK}
}

# ── S0: Small batch (accum=4), G1 base ──────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [S0] accum=4, K=1, G1 base (200 steps/epoch) ===" | tee -a "$PROGRESS"
echo "  8x more frequent inner corrections than G1's 25 steps" | tee -a "$PROGRESS"
run_sibl ablation_S0_smallbatch unlearn/muse/ablation_S0_smallbatch 4 \
    || echo "[WARN] S0 failed" | tee -a "$PROGRESS"

# ── S1: Single sample (accum=1), K=3 inner ──────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [S1] accum=1, K=3, ρ=0.01 (800 steps — closest to 8r) ===" | tee -a "$PROGRESS"
echo "  1 forget sample per step + 3 inner corrections" | tee -a "$PROGRESS"
run_sibl ablation_S1_single_sample unlearn/muse/ablation_S1_single_sample 1 \
    || echo "[WARN] S1 failed" | tee -a "$PROGRESS"

# ── S2: Small batch + inverted inner + full outer ───────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [S2] accum=4 + inverted inner mask + full outer ===" | tee -a "$PROGRESS"
echo "  S0 + N4 architecture (inner on retain neurons only)" | tee -a "$PROGRESS"
run_sibl ablation_S2_smallbatch_invinner unlearn/muse/ablation_S2_smallbatch_invinner 4 \
    || echo "[WARN] S2 failed" | tee -a "$PROGRESS"

# ── Summary ──────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] S series DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "  CE frontier: rk ≈ 0.55*fk + 0.17 | 8r: (0.371, 0.417) ABOVE" | tee -a "$PROGRESS"
echo "  G1 (accum=32): fk=0.274 rk=0.327 (ON frontier)" | tee -a "$PROGRESS"
for task in ablation_S0_smallbatch ablation_S1_single_sample ablation_S2_smallbatch_invinner; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done
