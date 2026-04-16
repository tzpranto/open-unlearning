#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   TOFU Forget 5%: All baselines
#   GradAscent, NPO, GradDiff, SimNPO, RMU, BLURNPO
#   Model: Llama-2-7b-chat-hf (fine-tuned on TOFU)
#   Single GPU, bs=12, 10 epochs (standard defaults)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

MODEL="Llama-2-7b-chat-hf"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
SAVES="saves/unlearn"
PROGRESS="$SAVES/_tofu_baselines_progress.log"

FORGET_SPLIT="forget05"
HOLDOUT_SPLIT="holdout05"
RETAIN_SPLIT="retain95"
RETAIN_LOGS="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"

BSZ=12
ACCUM=1
EPOCHS=10
LR=1e-5

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

run_baseline() {
    local trainer="$1"; shift
    local experiment="$1"; shift
    local task_name="tofu_${MODEL}_${FORGET_SPLIT}_${trainer}"
    local outdir="$SAVES/$task_name"

    if [[ -d "$outdir/evals" ]]; then
        log "[SKIP] $task_name — already done"
        return
    fi
    rm -rf "$outdir" 2>/dev/null

    log "[TRAIN] $task_name (trainer=$trainer, epochs=$EPOCHS, bs=${BSZ}x${ACCUM})"
    python src/train.py --config-name=unlearn.yaml \
        experiment="$experiment" \
        trainer="$trainer" \
        task_name="$task_name" \
        model="$MODEL" \
        forget_split="$FORGET_SPLIT" \
        retain_split="$RETAIN_SPLIT" \
        model.model_args.pretrained_model_name_or_path="$MODEL_PATH" \
        retain_logs_path="$RETAIN_LOGS" \
        trainer.args.per_device_train_batch_size=$BSZ \
        trainer.args.gradient_accumulation_steps=$ACCUM \
        trainer.args.num_train_epochs=$EPOCHS \
        trainer.args.learning_rate=$LR \
        trainer.args.gradient_checkpointing=true \
        "$@" 2>&1 | tail -20

    log "[EVAL] $task_name"
    python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split="$FORGET_SPLIT" \
        holdout_split="$HOLDOUT_SPLIT" \
        model="$MODEL" \
        task_name="$task_name" \
        model.model_args.pretrained_model_name_or_path="$outdir" \
        paths.output_dir="$outdir/evals" \
        retain_logs_path="$RETAIN_LOGS" \
        question_key="question" 2>&1 | tail -10

    log "[DONE] $task_name"

    if [[ -f "$outdir/evals/TOFU_SUMMARY.json" ]]; then
        log "[RESULT] $task_name:"
        cat "$outdir/evals/TOFU_SUMMARY.json" | tee -a "$PROGRESS"
        echo "" | tee -a "$PROGRESS"
    fi
}

log "═══════════════════════════════════════════════════════════════"
log "  TOFU Baselines — ${FORGET_SPLIT} — ${MODEL}"
log "  bs=${BSZ}x${ACCUM}, ${EPOCHS} epochs, lr=${LR}"
log "═══════════════════════════════════════════════════════════════"

# 1. GradAscent
run_baseline GradAscent unlearn/tofu/default.yaml

# 2. NPO
run_baseline NPO unlearn/tofu/default.yaml

# 3. GradDiff
run_baseline GradDiff unlearn/tofu/default.yaml

# 4. SimNPO
run_baseline SimNPO unlearn/tofu/default.yaml

# 5. RMU
run_baseline RMU unlearn/tofu/default.yaml

# 6. BLURNPO (needs bs=1 with higher accum per its config default)
run_baseline BLUR_NPO unlearn/tofu/default.yaml \
    trainer.args.per_device_train_batch_size=1 \
    trainer.args.gradient_accumulation_steps=12

log "═══════════════════════════════════════════════════════════════"
log "  All TOFU baselines done!"
log "═══════════════════════════════════════════════════════════════"
