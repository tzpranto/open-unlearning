#!/bin/bash
# TOFU baselines — bs=32, 5 seeds, robust to restarts
# Saves results to CSV. Cleans model weights after eval to save disk.
# Usage: nohup bash scripts/tofu_baselines.sh > saves/unlearn/baselines.log 2>&1 &
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

# ── Config ──────────────────────────────────────────────────
SEEDS=(42 123 456 789 1337)
TRAINERS="GradAscent GradDiff NPO SimNPO RMU BLURNPO PDU"
CSV="docs/results/tofu_baselines.csv"
LOCKFILE="saves/unlearn/.baselines.lock"

PDU_ARGS="trainer.method_args.alpha=100 trainer.method_args.retain_loss_eps=0.3 trainer.method_args.dual_step_size=5 trainer.method_args.dual_warmup_epochs=5"

# Queue: model,split,retain,holdout
QUEUE=(
    "Llama-3.2-1B-Instruct,forget01,retain99,holdout01"
    "Llama-3.2-1B-Instruct,forget05,retain95,holdout05"
    "Llama-3.2-1B-Instruct,forget10,retain90,holdout10"
    "Llama-3.2-3B-Instruct,forget01,retain99,holdout01"
    "Llama-3.2-3B-Instruct,forget05,retain95,holdout05"
    "Llama-3.2-3B-Instruct,forget10,retain90,holdout10"
)

# ── Batch size per model (eff_bs=32 for all) ────────────────
get_bs_accum() {
    local model=$1
    case "$model" in
        Llama-3.2-1B-Instruct) echo "8 4" ;;   # 8*4=32
        Llama-3.2-3B-Instruct) echo "4 8" ;;   # 4*8=32
        *) echo "4 8" ;;
    esac
}

# ── CSV setup ───────────────────────────────────────────────
mkdir -p saves/unlearn
if [[ ! -f "$CSV" ]]; then
    echo "model,split,method,seed,MU,FQ,ES,fgt_Prob,fgt_ROUGE,HM" > "$CSV"
fi

# ── Helper: check if run already in CSV ─────────────────────
is_done() {
    local model=$1 split=$2 method=$3 seed=$4
    grep -q "^${model},${split},${method},${seed}," "$CSV" 2>/dev/null
}

# ── Helper: extract metrics from eval JSON ──────────────────
extract_metrics() {
    local eval_json=$1
    python3 << PYEOF
import json, sys
from statistics import harmonic_mean
try:
    d = json.load(open("$eval_json"))
except:
    print("ERROR")
    sys.exit(1)

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

# ── Main loop ───────────────────────────────────────────────
total_runs=0
skip_runs=0
fail_runs=0

for queue_item in "${QUEUE[@]}"; do
    IFS=',' read -r MODEL SPLIT RETAIN HOLDOUT <<< "$queue_item"
    MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
    RETAIN_LOGS="saves/eval/tofu_${MODEL}_${RETAIN}/TOFU_EVAL.json"
    read -r BSZ ACCUM <<< "$(get_bs_accum "$MODEL")"

    echo "================================================================"
    echo " ${MODEL} / ${SPLIT} (bs=${BSZ}x${ACCUM}=32)"
    echo "================================================================"

    # Verify retain logs exist
    if [[ ! -f "$RETAIN_LOGS" ]]; then
        echo "[ERROR] Retain logs missing: $RETAIN_LOGS — skipping $MODEL/$SPLIT"
        continue
    fi

    for trainer in $TRAINERS; do
        for seed in "${SEEDS[@]}"; do
            total_runs=$((total_runs + 1))

            # Skip if already done
            if is_done "$MODEL" "$SPLIT" "$trainer" "$seed"; then
                skip_runs=$((skip_runs + 1))
                continue
            fi

            task_name="bs32_${MODEL}_${SPLIT}_${trainer}_s${seed}"
            outdir="saves/unlearn/${task_name}"

            echo "────────────────────────────────────────"
            echo "[RUN] $task_name"
            echo "────────────────────────────────────────"

            # Extra args
            extra_args=""
            [[ "$trainer" == "PDU" ]] && extra_args="$PDU_ARGS"

            # TRAIN
            echo "[TRAIN] $(date '+%H:%M:%S') $task_name"
            if ! CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
                experiment=unlearn/tofu/default.yaml \
                trainer=${trainer} \
                task_name=${task_name} \
                model=${MODEL} \
                forget_split=${SPLIT} \
                retain_split=${RETAIN} \
                model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
                retain_logs_path=${RETAIN_LOGS} \
                trainer.args.per_device_train_batch_size=${BSZ} \
                trainer.args.gradient_accumulation_steps=${ACCUM} \
                trainer.args.gradient_checkpointing=true \
                trainer.args.eval_strategy=no \
                trainer.args.do_eval=false \
                trainer.args.eval_on_start=false \
                trainer.args.seed=${seed} \
                ${extra_args} ; then
                echo "[TRAIN FAILED] $task_name"
                fail_runs=$((fail_runs + 1))
                rm -rf "$outdir" 2>/dev/null
                continue
            fi

            # EVAL
            echo "[EVAL] $(date '+%H:%M:%S') $task_name"
            if ! CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                experiment=eval/tofu/default.yaml \
                forget_split=${SPLIT} \
                holdout_split=${HOLDOUT} \
                model=${MODEL} \
                task_name=${task_name} \
                model.model_args.pretrained_model_name_or_path=${outdir} \
                paths.output_dir=${outdir}/evals \
                retain_logs_path=${RETAIN_LOGS} ; then
                echo "[EVAL FAILED] $task_name"
                fail_runs=$((fail_runs + 1))
                rm -rf "$outdir" 2>/dev/null
                continue
            fi

            # Extract metrics and append to CSV
            eval_json="${outdir}/evals/TOFU_EVAL.json"
            if [[ -f "$eval_json" ]]; then
                metrics=$(extract_metrics "$eval_json")
                if [[ "$metrics" != "ERROR" ]]; then
                    echo "${MODEL},${SPLIT},${trainer},${seed},${metrics}" >> "$CSV"
                    echo "[RESULT] $task_name: $metrics"
                else
                    echo "[EXTRACT FAILED] $task_name"
                    fail_runs=$((fail_runs + 1))
                fi
            else
                echo "[NO EVAL JSON] $task_name"
                fail_runs=$((fail_runs + 1))
            fi

            # Clean model weights — keep only evals/
            rm -rf "$outdir"/pytorch_model* "$outdir"/model* "$outdir"/checkpoint-* \
                   "$outdir"/adapter_model* "$outdir"/optimizer* "$outdir"/scheduler* \
                   "$outdir"/training_args* "$outdir"/config.json "$outdir"/generation_config* \
                   "$outdir"/tokenizer* "$outdir"/special_tokens* "$outdir"/added_tokens* 2>/dev/null
            # Remove safetensors
            find "$outdir" -maxdepth 1 -name "*.safetensors" -delete 2>/dev/null
            find "$outdir" -maxdepth 1 -name "*.bin" -delete 2>/dev/null

            echo "[CLEANED] $task_name — kept evals only"
        done
    done

    echo ""
    echo "[BLOCK DONE] ${MODEL}/${SPLIT}: total=$total_runs skip=$skip_runs fail=$fail_runs"
    echo ""
done

echo "================================================================"
echo " ALL DONE: total=$total_runs skip=$skip_runs fail=$fail_runs"
echo " Results: $CSV"
echo "================================================================"
