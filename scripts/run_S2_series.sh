#!/bin/bash
# S2 series: Small-batch NPO with EARLY STOPPING
# KEY INSIGHT from S0/S1: Small-batch dynamics are healthy early but collapse later.
# S1 at step 120: L_ret=0.84 (near target), L_fgt=0.05 (strong forget) — GREAT.
# S1 at step 500: L_ret=7.0, collapsed — too many steps.
#
# 8r processed only T=10 samples. S1 processed 813. Fix: limit total samples.
#
# Experiments (all accum=1, K=3, ρ=0.01 like S1):
#   S3: 25 steps (25 samples — minimal, like pruned 8r)
#   S4: 50 steps (50 samples — 5x 8r)
#   S5: 100 steps (100 samples — sweet spot candidate)
#   S6: 200 steps (200 samples — S1 was healthy until ~400)
#   S7: accum=4, 25 steps, K=1, ρ=0.1 (S0 config capped at 25 — = G1 but small-batch)
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/S2_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting S2 series (small-batch + early stopping)" | tee "$PROGRESS"

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

run_sibl_earlystop() {
    local TASK=$1 EXPERIMENT=$2 ACCUM=$3 STOP=$4 K_VAL=${5:-3} RHO_VAL=${6:-0.01}
    local TASK_DIR="saves/unlearn/${TASK}"
    if compgen -G "${TASK_DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] ${TASK}: weights exist" | tee -a "$PROGRESS"
        local EVAL_FILE="${TASK_DIR}/evals/MUSE_EVAL.json"
        if [ ! -f "$EVAL_FILE" ]; then run_eval ${TASK}
        else echo "[CACHED]" | tee -a "$PROGRESS"; print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"; fi
        return 0
    fi
    echo "[$(date '+%H:%M:%S')] [TRAIN] ${TASK} (accum=${ACCUM}, stop=${STOP}, K=${K_VAL}, ρ=${RHO_VAL})" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=${EXPERIMENT} model=${MODEL} data_split=${DATA_SPLIT} \
        trainer=SIBL task_name=${TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_accumulation_steps=${ACCUM} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        trainer.method_args.debug_stop_after_outer=${STOP} \
        trainer.method_args.K=${K_VAL} \
        trainer.method_args.rho=${RHO_VAL} \
        "${ATTN_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/train_${TASK}.log"
    local STATUS=$?
    [ $STATUS -ne 0 ] && echo "[WARN] ${TASK} failed (exit $STATUS)" | tee -a "$PROGRESS" && return $STATUS
    run_eval ${TASK}
}

# ── S3: 25 steps, accum=1, K=3 (25 samples total) ──────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [S3] 25 steps × accum=1 × K=3 (25 samples, minimal) ===" | tee -a "$PROGRESS"
run_sibl_earlystop ablation_S3_25steps unlearn/muse/ablation_S1_single_sample 1 24 3 0.01 \
    || echo "[WARN] S3 failed" | tee -a "$PROGRESS"

# ── S4: 50 steps, accum=1, K=3 (50 samples total) ──────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [S4] 50 steps × accum=1 × K=3 (50 samples, 5x 8r) ===" | tee -a "$PROGRESS"
run_sibl_earlystop ablation_S4_50steps unlearn/muse/ablation_S1_single_sample 1 49 3 0.01 \
    || echo "[WARN] S4 failed" | tee -a "$PROGRESS"

# ── S5: 100 steps, accum=1, K=3 (100 samples) ──────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [S5] 100 steps × accum=1 × K=3 (100 samples) ===" | tee -a "$PROGRESS"
run_sibl_earlystop ablation_S5_100steps unlearn/muse/ablation_S1_single_sample 1 99 3 0.01 \
    || echo "[WARN] S5 failed" | tee -a "$PROGRESS"

# ── S6: 200 steps, accum=1, K=3 (200 samples — where S1 was still stable) ──
echo "" | tee -a "$PROGRESS"
echo "=== [S6] 200 steps × accum=1 × K=3 (200 samples) ===" | tee -a "$PROGRESS"
run_sibl_earlystop ablation_S6_200steps unlearn/muse/ablation_S1_single_sample 1 199 3 0.01 \
    || echo "[WARN] S6 failed" | tee -a "$PROGRESS"

# ── S7: G1-equivalent but accum=4 (100 samples total, 25 steps) ─────────────
echo "" | tee -a "$PROGRESS"
echo "=== [S7] 25 steps × accum=4 × K=1 (100 samples, G1-like with small batch) ===" | tee -a "$PROGRESS"
run_sibl_earlystop ablation_S7_g1_smallbatch unlearn/muse/ablation_S0_smallbatch 4 24 1 0.1 \
    || echo "[WARN] S7 failed" | tee -a "$PROGRESS"

# ── Summary ──────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] S2 series DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "  CE frontier: rk ≈ 0.55*fk + 0.17 | G1: (0.274, 0.327) ON" | tee -a "$PROGRESS"
for task in ablation_S3_25steps ablation_S4_50steps ablation_S5_100steps ablation_S6_200steps ablation_S7_g1_smallbatch; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done
