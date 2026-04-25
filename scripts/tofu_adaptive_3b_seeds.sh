#!/bin/bash
# LoRA-BiAL-Adaptive multi-seed runs — 5 seeds × 3 splits (3B)
# Same protocol as 1B: 5 seeds, results to CSV with mean±std
# Usage: nohup bash scripts/tofu_adaptive_3b_seeds.sh > saves/unlearn/adaptive_3b_seeds.log 2>&1 &
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

# ── Config ──────────────────────────────────────────────────
SEEDS=(42 123 456 789 1337)
MODEL="Llama-3.2-3B-Instruct"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
CSV="results/tofu_adaptive_3b_seeds.csv"

SPLITS=(
    "forget01,retain99,holdout01"
    "forget05,retain95,holdout05"
    "forget10,retain90,holdout10"
)

# ── CSV setup ───────────────────────────────────────────────
mkdir -p saves/unlearn
if [[ ! -f "$CSV" ]]; then
    echo "model,split,seed,stop_step,MU,FQ,ES,fgt_Prob,fgt_ROUGE,HM" > "$CSV"
fi

is_done() {
    local split=$1 seed=$2
    grep -q "^${MODEL},${split},${seed}," "$CSV" 2>/dev/null
}

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

# ── Retain logs per split ──────────────────────────────────
get_retain_logs() {
    local split=$1
    case "$split" in
        forget01) echo "saves/eval/tofu_${MODEL}_retain99/TOFU_EVAL.json" ;;
        forget05) echo "saves/eval/tofu_${MODEL}_retain95/TOFU_EVAL.json" ;;
        forget10) echo "saves/eval/tofu_${MODEL}_retain90/TOFU_EVAL.json" ;;
    esac
}

# ── T per split ────────────────────────────────────────────
get_T() {
    local split=$1
    case "$split" in
        forget01) echo 250 ;;
        forget05) echo 250 ;;
        forget10) echo 500 ;;
    esac
}

# ── Main loop ──────────────────────────────────────────────
total=0; skip=0; fail=0; done_count=0

for split_cfg in "${SPLITS[@]}"; do
    IFS=',' read -r SPLIT RETAIN HOLDOUT <<< "$split_cfg"
    RETAIN_LOGS=$(get_retain_logs "$SPLIT")
    T=$(get_T "$SPLIT")

    echo "================================================================"
    echo "[SPLIT] $MODEL / $SPLIT (T=$T)"
    echo "================================================================"

    if [[ ! -f "$RETAIN_LOGS" ]]; then
        echo "[ERROR] Retain logs missing: $RETAIN_LOGS — skipping"
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        total=$((total + 1))

        if is_done "$SPLIT" "$seed"; then
            echo "[SKIP] $SPLIT s$seed already in CSV"
            skip=$((skip + 1))
            continue
        fi

        task_name="adaptive_${MODEL}_${SPLIT}_s${seed}"
        outdir="saves/unlearn/${task_name}"

        echo "────────────────────────────────────────"
        echo "[RUN] $task_name"
        echo "────────────────────────────────────────"

        mkdir -p "$outdir"

        # TRAIN
        echo "[TRAIN] $(date '+%H:%M:%S') $task_name"
        if ! python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/lora_bial_3b.yaml \
            task_name="$task_name" \
            trainer=LoRABiALAdaptive \
            model=${MODEL} \
            model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
            forget_split=${SPLIT} retain_split=${RETAIN} holdout_split=${HOLDOUT} \
            trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
            trainer.method_args.checkpoint_every_epoch=false \
            trainer.method_args.eta_theta=5e-5 trainer.method_args.T=${T} \
            trainer.method_args.epsilon=99.0 \
            trainer.method_args.epsilon_multiplier=0.85 \
            trainer.args.seed=${seed} \
            retain_logs_path="$RETAIN_LOGS" 2>&1 | tee "${outdir}/train.log" ; then
            echo "[TRAIN FAILED] $task_name"
            fail=$((fail + 1))
            rm -rf "$outdir" 2>/dev/null
            continue
        fi

        # EVAL
        echo "[EVAL] $(date '+%H:%M:%S') $task_name"
        if ! python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${SPLIT} holdout_split=${HOLDOUT} \
            model=${MODEL} \
            task_name="${task_name}" \
            model.model_args.pretrained_model_name_or_path="${outdir}" \
            paths.output_dir="${outdir}/evals" \
            retain_logs_path="$RETAIN_LOGS" ; then
            echo "[EVAL FAILED] $task_name"
            fail=$((fail + 1))
            continue
        fi

        # Extract metrics
        eval_json="${outdir}/evals/TOFU_EVAL.json"
        if [[ -f "$eval_json" ]]; then
            metrics=$(extract_metrics "$eval_json")
            if [[ "$metrics" != "ERROR" ]]; then
                conv_step=$(grep "CONVERGED at step" "${outdir}/train.log" 2>/dev/null | grep -oP 'step \K[0-9]+' | head -1)
                stop=${conv_step:-$T}
                echo "${MODEL},${SPLIT},${seed},${stop},${metrics}" >> "$CSV"
                echo "[RESULT] $task_name: stop=$stop $metrics"
                done_count=$((done_count + 1))
            else
                echo "[EXTRACT FAILED] $task_name"
                fail=$((fail + 1))
            fi
        fi

        # Clean up model weights to save disk (keep evals + train.log)
        rm -f "${outdir}/model.safetensors" "${outdir}/training_args.bin" \
              "${outdir}/trainer_state.json" 2>/dev/null
        rm -rf "${outdir}/converged-best" 2>/dev/null

        echo "[DONE] $task_name"
        echo ""
    done
done

echo ""
echo "========================================"
echo "ALL DONE: total=$total skip=$skip done=$done_count fail=$fail"
echo "Results: $CSV"
echo "========================================"
cat "$CSV"
