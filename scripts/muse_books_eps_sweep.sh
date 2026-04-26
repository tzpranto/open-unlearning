#!/bin/bash
# MUSE Books: LoRA-BiAL-Adaptive epsilon sweep
# Runs 3 epsilon_multiplier values with T=250, seed=42
# Usage: nohup bash scripts/muse_books_eps_sweep.sh > saves/unlearn/eps_sweep.log 2>&1 &
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

DATA_SPLIT="Books"
MODEL="Llama-2-7b-hf"
SEED=42
T=250
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"

EPS_VALUES=(1.15 1.25 1.5)

echo "================================================================"
echo " LoRA-BiAL-Adaptive eps sweep — started $(date)"
echo " T=${T}, eps_values=${EPS_VALUES[*]}, seed=${SEED}"
echo "================================================================"

if [[ ! -f "$RETAIN_LOGS" ]]; then
    echo "[ERROR] Retain logs missing: $RETAIN_LOGS"
    exit 1
fi

for EPS in "${EPS_VALUES[@]}"; do
    # Use underscore for directory name (1.15 -> 1_15)
    EPS_TAG=$(echo "$EPS" | tr '.' '_')
    task_name="muse_${MODEL}_${DATA_SPLIT}_adaptive_eps${EPS_TAG}_T${T}_s${SEED}"
    outdir="saves/unlearn/${task_name}"

    if [[ -f "$outdir/evals/MUSE_SUMMARY.json" ]]; then
        echo "[SKIP] $task_name — eval exists"
        cat "$outdir/evals/MUSE_SUMMARY.json"
        echo ""
        continue
    fi

    mkdir -p "$outdir"

    echo "────────────────────────────────────────"
    echo "[TRAIN] $(date '+%H:%M:%S') $task_name (eps_mul=${EPS}, T=${T})"
    echo "────────────────────────────────────────"

    if ! CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/lora_bial_adaptive_books.yaml \
        task_name=${task_name} \
        retain_logs_path=${RETAIN_LOGS} \
        trainer.method_args.epsilon_multiplier=${EPS} \
        trainer.method_args.T=${T} \
        trainer.args.seed=${SEED} 2>&1 | tee "${outdir}/train.log" ; then
        echo "[TRAIN FAILED] $task_name"
        continue
    fi

    echo "[EVAL] $(date '+%H:%M:%S') $task_name"
    if ! CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${task_name} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${outdir} \
        paths.output_dir=${outdir}/evals \
        retain_logs_path=${RETAIN_LOGS} ; then
        echo "[EVAL FAILED] $task_name"
        continue
    fi

    echo "[DONE] $task_name"
    cat "$outdir/evals/MUSE_SUMMARY.json" 2>/dev/null || true

    # Clean model weights, keep evals + history
    find "$outdir" -maxdepth 1 -name "*.safetensors" -delete 2>/dev/null
    find "$outdir" -maxdepth 1 -name "*.bin" -delete 2>/dev/null
    rm -f "$outdir"/config.json "$outdir"/generation_config* \
          "$outdir"/tokenizer* "$outdir"/special_tokens* "$outdir"/added_tokens* 2>/dev/null
    rm -rf "$outdir"/checkpoint-* "$outdir"/converged-best 2>/dev/null
    echo "[CLEANED] $task_name"
    echo ""
done

echo ""
echo "================================================================"
echo " SWEEP SUMMARY"
echo "================================================================"
for EPS in "${EPS_VALUES[@]}"; do
    EPS_TAG=$(echo "$EPS" | tr '.' '_')
    task_name="muse_${MODEL}_${DATA_SPLIT}_adaptive_eps${EPS_TAG}_T${T}_s${SEED}"
    outdir="saves/unlearn/${task_name}"
    echo "--- eps=${EPS} ---"
    if [[ -f "$outdir/evals/MUSE_SUMMARY.json" ]]; then
        cat "$outdir/evals/MUSE_SUMMARY.json"
    else
        echo "  NO RESULTS"
    fi
    echo ""
done

echo "ALL DONE — $(date)"
