#!/bin/bash
# MUSE Books: baselines (1 seed) + LoRA-BiAL-Adaptive (1 seed)
# Usage: nohup bash scripts/muse_books_run.sh > saves/unlearn/muse_books_run.log 2>&1 &
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

DATA_SPLIT="Books"
MODEL="Llama-2-7b-hf"
BSZ=2
ACCUM=8
SEED=42
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"

# PDU paper params (arXiv:2506.05314 Table 5)
PDU_ARGS="trainer.method_args.alpha=50 trainer.method_args.retain_loss_eps=1.5 trainer.method_args.dual_step_size=1 trainer.method_args.dual_warmup_epochs=3"

trainers=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "SimNPO"
    "RMU"
    "BLURNPO"
    "PDU"
)

echo "================================================================"
echo " MUSE Books — started $(date)"
echo "================================================================"

if [[ ! -f "$RETAIN_LOGS" ]]; then
    echo "[ERROR] Retain logs missing: $RETAIN_LOGS"
    exit 1
fi

# ══════════════════════════════════════════════════════════════
# PHASE 1: Baselines (7 methods × 1 seed)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: Baselines — 7 methods, seed=$SEED                ║"
echo "╚══════════════════════════════════════════════════════════════╝"

for trainer in "${trainers[@]}"; do
    task_name="muse_${MODEL}_${DATA_SPLIT}_${trainer}_s${SEED}"
    outdir="saves/unlearn/${task_name}"

    if [[ -f "$outdir/evals/MUSE_EVAL.json" ]]; then
        echo "[SKIP] $task_name — eval exists"
        continue
    fi

    extra_args=""
    [[ "$trainer" == "PDU" ]] && extra_args="$PDU_ARGS"

    mkdir -p "$outdir"

    echo "────────────────────────────────────────"
    echo "[TRAIN] $(date '+%H:%M:%S') $task_name (bs=${BSZ}x${ACCUM})"
    echo "────────────────────────────────────────"

    if ! python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default.yaml \
        model=${MODEL} \
        data_split=${DATA_SPLIT} \
        trainer=${trainer} \
        task_name=${task_name} \
        retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=${BSZ} \
        trainer.args.gradient_accumulation_steps=${ACCUM} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.seed=${SEED} \
        ${extra_args} 2>&1 | tee "${outdir}/train.log" ; then
        echo "[TRAIN FAILED] $task_name"
        continue
    fi

    echo "[EVAL] $(date '+%H:%M:%S') $task_name"
    if ! python src/eval.py \
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

    # Clean model weights, keep evals
    find "$outdir" -maxdepth 1 -name "*.safetensors" -delete 2>/dev/null
    find "$outdir" -maxdepth 1 -name "*.bin" -delete 2>/dev/null
    rm -f "$outdir"/config.json "$outdir"/generation_config* \
          "$outdir"/tokenizer* "$outdir"/special_tokens* "$outdir"/added_tokens* 2>/dev/null
    rm -rf "$outdir"/checkpoint-* 2>/dev/null
    echo "[CLEANED] $task_name"
    echo ""
done

# ══════════════════════════════════════════════════════════════
# PHASE 2: LoRA-BiAL-Adaptive (1 seed)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: LoRA-BiAL-Adaptive, seed=$SEED                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"

task_name="muse_${MODEL}_${DATA_SPLIT}_adaptive_s${SEED}"
outdir="saves/unlearn/${task_name}"

if [[ -f "$outdir/evals/MUSE_EVAL.json" ]]; then
    echo "[SKIP] $task_name — eval exists"
else
    mkdir -p "$outdir"

    echo "[TRAIN] $(date '+%H:%M:%S') $task_name"
    if python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/lora_bial_adaptive_books.yaml \
        task_name=${task_name} \
        retain_logs_path=${RETAIN_LOGS} \
        trainer.args.seed=${SEED} 2>&1 | tee "${outdir}/train.log" ; then

        echo "[EVAL] $(date '+%H:%M:%S') $task_name"
        python src/eval.py \
            experiment=eval/muse/default.yaml \
            data_split=${DATA_SPLIT} \
            task_name=${task_name} \
            model=${MODEL} \
            model.model_args.pretrained_model_name_or_path=${outdir} \
            paths.output_dir=${outdir}/evals \
            retain_logs_path=${RETAIN_LOGS}

        echo "[DONE] $task_name"
        cat "$outdir/evals/MUSE_SUMMARY.json" 2>/dev/null || true
    else
        echo "[TRAIN FAILED] $task_name"
    fi
fi

echo ""
echo "================================================================"
echo " ALL DONE — $(date)"
echo "================================================================"
