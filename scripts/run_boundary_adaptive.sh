#!/bin/bash
# Boundary + Adaptive experiments on TOFU forget01
# Phase A: High LR boundary tests (5e-5, 7e-5, 1e-4) with T=250 safety cap
# Phase B: Adaptive trainer starting from lr=5e-5 (lets calibration adjust)
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

CSV="docs/results/tofu_boundary_adaptive.csv"
if [[ ! -f "$CSV" ]]; then
    echo "model,split,task,stop_step,MU,FQ,ES,fgt_Prob,fgt_ROUGE,HM" > "$CSV"
fi

MODEL="Llama-3.2-1B-Instruct"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
RETAIN_LOGS="saves/eval/tofu_${MODEL}_retain99/TOFU_EVAL.json"

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

run_fixed() {
    local TASK=$1 LR=$2 T=$3 EMULT=$4
    local OUTDIR="saves/unlearn/${TASK}"
    local LOGFILE="${OUTDIR}/train.log"

    if grep -q "^${MODEL},forget01,${TASK}," "$CSV" 2>/dev/null; then
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
        forget_split=forget01 retain_split=retain99 holdout_split=holdout01 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        trainer.method_args.checkpoint_every_epoch=false \
        trainer.method_args.eta_theta=${LR} trainer.method_args.T=${T} \
        trainer.method_args.epsilon=99.0 \
        trainer.method_args.epsilon_multiplier=${EMULT} \
        retain_logs_path="$RETAIN_LOGS" 2>&1 | tee "$LOGFILE"

    # Eval final model
    echo "[EVAL] $TASK final"
    python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=forget01 holdout_split=holdout01 \
        model=${MODEL} \
        task_name="${TASK}" \
        model.model_args.pretrained_model_name_or_path="${OUTDIR}" \
        paths.output_dir="${OUTDIR}/evals" \
        retain_logs_path="$RETAIN_LOGS"

    metrics=$(extract_metrics "${OUTDIR}/evals/TOFU_EVAL.json")
    if [[ "$metrics" != "ERROR" ]]; then
        # Get ES fired step if any
        fired=$(grep "AUTO-STOP: criterion fired" "$LOGFILE" 2>/dev/null | grep -oP 'step \K[0-9]+' | head -1)
        stop=${fired:-$T}
        echo "${MODEL},forget01,${TASK},${stop},${metrics}" >> "$CSV"
        echo "[RESULT] $TASK: stop=$stop $metrics"
    fi

    # Also eval auto-stop-best if it exists
    BEST_DIR="${OUTDIR}/auto-stop-best"
    if [[ -d "$BEST_DIR" ]]; then
        if ! grep -q "^${MODEL},forget01,${TASK}_es," "$CSV" 2>/dev/null; then
            echo "[EVAL] $TASK auto-stop-best"
            python src/eval.py \
                experiment=eval/tofu/default.yaml \
                forget_split=forget01 holdout_split=holdout01 \
                model=${MODEL} \
                task_name="${TASK}_es" \
                model.model_args.pretrained_model_name_or_path="$BEST_DIR" \
                paths.output_dir="${BEST_DIR}/evals" \
                retain_logs_path="$RETAIN_LOGS"

            metrics=$(extract_metrics "${BEST_DIR}/evals/TOFU_EVAL.json")
            if [[ "$metrics" != "ERROR" ]]; then
                echo "${MODEL},forget01,${TASK}_es,${fired:-0},${metrics}" >> "$CSV"
                echo "[RESULT] $TASK ES@${fired}: $metrics"
            fi
        fi
    fi

    echo "[DONE] $TASK"
    echo ""
}

run_adaptive() {
    local TASK=$1 LR=$2 T=$3 EMULT=$4
    local OUTDIR="saves/unlearn/${TASK}"
    local LOGFILE="${OUTDIR}/train.log"

    if grep -q "^${MODEL},forget01,${TASK}," "$CSV" 2>/dev/null; then
        echo "[SKIP] $TASK already in CSV"
        return
    fi

    echo "========================================"
    echo "[TRAIN-ADAPTIVE] $TASK (start_lr=$LR, T_cap=$T, emult=$EMULT)"
    echo "========================================"

    mkdir -p "$OUTDIR"
    python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/lora_bial_1b.yaml \
        task_name="$TASK" \
        trainer=LoRABiALAdaptive \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
        forget_split=forget01 retain_split=retain99 holdout_split=holdout01 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        trainer.method_args.checkpoint_every_epoch=false \
        trainer.method_args.eta_theta=${LR} trainer.method_args.T=${T} \
        trainer.method_args.epsilon=99.0 \
        trainer.method_args.epsilon_multiplier=${EMULT} \
        retain_logs_path="$RETAIN_LOGS" 2>&1 | tee "$LOGFILE"

    # Eval final model
    echo "[EVAL] $TASK final"
    python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=forget01 holdout_split=holdout01 \
        model=${MODEL} \
        task_name="${TASK}" \
        model.model_args.pretrained_model_name_or_path="${OUTDIR}" \
        paths.output_dir="${OUTDIR}/evals" \
        retain_logs_path="$RETAIN_LOGS"

    metrics=$(extract_metrics "${OUTDIR}/evals/TOFU_EVAL.json")
    if [[ "$metrics" != "ERROR" ]]; then
        # Get convergence step
        conv_step=$(grep "CONVERGED at step" "$LOGFILE" 2>/dev/null | grep -oP 'step \K[0-9]+' | head -1)
        stop=${conv_step:-$T}
        echo "${MODEL},forget01,${TASK},${stop},${metrics}" >> "$CSV"
        echo "[RESULT] $TASK: stop=$stop $metrics"
    fi

    # Also eval converged-best if it exists
    CONV_DIR="${OUTDIR}/converged-best"
    if [[ -d "$CONV_DIR" ]]; then
        if ! grep -q "^${MODEL},forget01,${TASK}_conv," "$CSV" 2>/dev/null; then
            echo "[EVAL] $TASK converged-best"
            python src/eval.py \
                experiment=eval/tofu/default.yaml \
                forget_split=forget01 holdout_split=holdout01 \
                model=${MODEL} \
                task_name="${TASK}_conv" \
                model.model_args.pretrained_model_name_or_path="$CONV_DIR" \
                paths.output_dir="${CONV_DIR}/evals" \
                retain_logs_path="$RETAIN_LOGS"

            metrics=$(extract_metrics "${CONV_DIR}/evals/TOFU_EVAL.json")
            if [[ "$metrics" != "ERROR" ]]; then
                echo "${MODEL},forget01,${TASK}_conv,${conv_step:-0},${metrics}" >> "$CSV"
                echo "[RESULT] $TASK conv@${conv_step}: $metrics"
            fi
        fi
    fi

    echo "[DONE] $TASK"
    echo ""
}

# ═══════════════════════════════════════════════════════════════
# Phase A: Boundary experiments — find divergence LR
# ═══════════════════════════════════════════════════════════════
echo "############################################"
echo "# Phase A: Boundary LR tests (fixed ES)   #"
echo "############################################"

run_fixed "tofu_1b_01_lr5e5" 5e-5 250 0.85
run_fixed "tofu_1b_01_lr7e5" 7e-5 250 0.85
run_fixed "tofu_1b_01_lr1e4" 1e-4 250 0.85

# ═══════════════════════════════════════════════════════════════
# Phase B: Adaptive trainer — start from 5e-5, let it calibrate
# ═══════════════════════════════════════════════════════════════
echo "############################################"
echo "# Phase B: Adaptive trainer                #"
echo "############################################"

run_adaptive "tofu_1b_01_adaptive" 5e-5 250 0.85

echo ""
echo "ALL DONE"
echo "Results: $CSV"
cat "$CSV"
