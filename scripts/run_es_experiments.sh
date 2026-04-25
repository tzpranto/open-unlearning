#!/bin/bash
# LoRA-BiAL-ES experiments: forget01 LR comparison
# Phase 1: Two LRs on forget01, eval ES-best + final for each
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

CSV="results/tofu_es_results.csv"
if [[ ! -f "$CSV" ]]; then
    echo "model,split,task,stop_step,MU,FQ,ES,fgt_Prob,fgt_ROUGE,HM" > "$CSV"
fi

extract_metrics() {
    local eval_json=$1
    python3 << PYEOF
import json, sys
from statistics import harmonic_mean
try:
    d = json.load(open("$eval_json"))
except:
    print("ERROR"); sys.exit(1)
def val(x):
    return x['agg_value'] if isinstance(x, dict) else x
mu = val(d.get('model_utility', 0))
fq = val(d.get('forget_quality', d.get('mia_min_k', 0)))
es = val(d.get('extraction_strength', 0))
fp = val(d.get('fgt_Q_A_Prob', d.get('forget_Q_A_Prob', 0)))
fr = val(d.get('fgt_Q_A_ROUGE', d.get('forget_Q_A_ROUGE', 0)))
vals = [mu, 1-fp, 1-fr]
hm = harmonic_mean(vals) if all(v > 0 for v in vals) else 0.0
print(f"{mu:.4f},{fq:.4f},{es:.4f},{fp:.4f},{fr:.4f},{hm:.4f}")
PYEOF
}

get_fired_step() {
    local logfile=$1
    grep "AUTO-STOP: criterion fired" "$logfile" 2>/dev/null | grep -oP 'step \K[0-9]+' | head -1
}

run_and_eval() {
    local MODEL=$1 SPLIT=$2 RETAIN=$3 HOLDOUT=$4 TASK=$5 LR=$6 T=$7 EMULT=$8
    local MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
    local RETAIN_LOGS="saves/eval/tofu_${MODEL}_${RETAIN}/TOFU_EVAL.json"
    local OUTDIR="saves/unlearn/${TASK}"
    local LOGFILE="${OUTDIR}/train.log"

    # Skip if already done
    if grep -q "^${MODEL},${SPLIT},${TASK}," "$CSV" 2>/dev/null; then
        echo "[SKIP] $TASK already in CSV"
        return
    fi

    echo "========================================"
    echo "[TRAIN] $TASK (T=$T, lr=$LR, emult=$EMULT)"
    echo "========================================"

    mkdir -p "$OUTDIR"
    python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/lora_bial_1b.yaml \
        task_name="$TASK" \
        trainer=LoRABiALES \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
        forget_split=${SPLIT} retain_split=${RETAIN} holdout_split=${HOLDOUT} \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        trainer.method_args.checkpoint_every_epoch=false \
        trainer.method_args.eta_theta=${LR} trainer.method_args.T=${T} \
        trainer.method_args.epsilon=99.0 \
        trainer.method_args.epsilon_multiplier=${EMULT} \
        retain_logs_path="$RETAIN_LOGS" 2>&1 | tee "$LOGFILE"

    FIRED=$(get_fired_step "$LOGFILE")
    echo "[INFO] Auto-stop fired at step: ${FIRED:-NONE}"

    # Eval auto-stop-best
    BEST_DIR="${OUTDIR}/auto-stop-best"
    if [[ -d "$BEST_DIR" ]]; then
        echo "[EVAL] auto-stop-best (fired@${FIRED})"
        python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${SPLIT} holdout_split=${HOLDOUT} \
            model=${MODEL} \
            task_name="${TASK}_best" \
            model.model_args.pretrained_model_name_or_path="$BEST_DIR" \
            paths.output_dir="${BEST_DIR}/evals" \
            retain_logs_path="$RETAIN_LOGS"

        metrics=$(extract_metrics "${BEST_DIR}/evals/TOFU_EVAL.json")
        if [[ "$metrics" != "ERROR" ]]; then
            echo "${MODEL},${SPLIT},${TASK},${FIRED:-0},${metrics}" >> "$CSV"
            echo "[RESULT] $TASK auto-stop@${FIRED}: $metrics"
        fi
    fi

    # Eval final model too
    echo "[EVAL] final T=${T}"
    python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=${SPLIT} holdout_split=${HOLDOUT} \
        model=${MODEL} \
        task_name="${TASK}_final" \
        model.model_args.pretrained_model_name_or_path="${OUTDIR}" \
        paths.output_dir="${OUTDIR}/evals" \
        retain_logs_path="$RETAIN_LOGS"

    metrics=$(extract_metrics "${OUTDIR}/evals/TOFU_EVAL.json")
    if [[ "$metrics" != "ERROR" ]]; then
        echo "${MODEL},${SPLIT},${TASK}_final,${T},${metrics}" >> "$CSV"
        echo "[RESULT] $TASK final@${T}: $metrics"
    fi

    echo "[DONE] $TASK"
}

# ═══════════════════════════════════════════════════════════════
# Experiment 1: Forget01 — lr=2e-5, T=150
# ═══════════════════════════════════════════════════════════════
echo ""
echo "############ EXP 1: forget01 lr=2e-5 T=150 ############"
run_and_eval Llama-3.2-1B-Instruct forget01 retain99 holdout01 \
    "tofu_1b_01_es_lr2e5" 2e-5 150 0.85

# ═══════════════════════════════════════════════════════════════
# Experiment 2: Forget01 — lr=3e-5, T=150
# ═══════════════════════════════════════════════════════════════
echo ""
echo "############ EXP 2: forget01 lr=3e-5 T=150 ############"
run_and_eval Llama-3.2-1B-Instruct forget01 retain99 holdout01 \
    "tofu_1b_01_es_lr3e5" 3e-5 150 0.85

echo ""
echo "PHASE 1 DONE"
echo "Results: $CSV"
cat "$CSV"
