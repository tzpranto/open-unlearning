#!/bin/bash
# MUSE baselines — single GPU
# Usage: bash scripts/muse_baselines.sh
# Configure DATA_SPLIT (News/Books), batch size, and trainers below.
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

# ── Config ──────────────────────────────────────────────────
DATA_SPLIT="Books"   # "News" or "Books"
MODEL="Llama-2-7b-hf"
BSZ=2
ACCUM=8

trainers=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "SimNPO"
    "RMU"
    "BLURNPO"
    "PDU"
)

# PDU needs extra args
# PDU paper params for MUSE (arXiv:2506.05314 Table 5)
PDU_ARGS="trainer.method_args.alpha=50 trainer.method_args.retain_loss_eps=1.5 trainer.method_args.dual_step_size=1 trainer.method_args.dual_warmup_epochs=3"
# ────────────────────────────────────────────────────────────

RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"

for trainer in "${trainers[@]}"; do
    task_name="muse_${MODEL}_${DATA_SPLIT}_${trainer}"
    outdir="saves/unlearn/${task_name}"

    if [[ -f "$outdir/evals/MUSE_SUMMARY.json" ]]; then
        echo "[SKIP] $task_name"
        cat "$outdir/evals/MUSE_SUMMARY.json"
        continue
    fi

    extra_args=""
    [[ "$trainer" == "PDU" ]] && extra_args="$PDU_ARGS"

    echo "[TRAIN] $task_name (bs=${BSZ}x${ACCUM})"
    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
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
        ${extra_args}

    echo "[EVAL] $task_name"
    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${task_name} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${outdir} \
        paths.output_dir=${outdir}/evals \
        retain_logs_path=${RETAIN_LOGS}

    echo "[DONE] $task_name"
    cat "$outdir/evals/MUSE_SUMMARY.json" 2>/dev/null || true
    rm -rf "$outdir"/checkpoint-* 2>/dev/null
done

echo "ALL DONE"
