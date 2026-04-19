#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   TOFU All Baselines: 7 methods × 3 splits (1%, 5%, 10%)
#   Model: Llama-2-7b-chat-hf (fine-tuned on TOFU)
#   Single GPU, restart-safe (skip if evals/ dir exists)
#
#   Methods: GradAscent, GradDiff, NPO, SimNPO, RMU, BLURNPO, PDU
#   Splits:  forget01/retain99, forget05/retain95, forget10/retain90
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="/datadrive/forked/open-unlearning"
cd "$BASE"

MODEL="Llama-2-7b-chat-hf"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
SAVES="saves/unlearn"
PROGRESS="$SAVES/_tofu_all_baselines_progress.log"

EPOCHS=10
LR=1e-5

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

run_baseline() {
    local trainer="$1"; shift
    local forget_split="$1"; shift
    local retain_split="$1"; shift
    local holdout_split="$1"; shift
    local bsz="$1"; shift
    local accum="$1"; shift
    local task_name="tofu_${MODEL}_${forget_split}_${trainer}"
    local outdir="$SAVES/$task_name"
    local retain_logs="saves/eval/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json"

    # Skip if already evaluated
    if [[ -f "$outdir/evals/TOFU_SUMMARY.json" ]]; then
        log "[SKIP] $task_name — already done"
        return 0
    fi

    # Skip training if model already saved (eval-only retry)
    if compgen -G "$outdir/model-*.safetensors" > /dev/null 2>&1; then
        log "[SKIP-TRAIN] $task_name — model exists, running eval only"
    else
        log "[TRAIN] $task_name (trainer=$trainer, split=$forget_split, bs=${bsz}x${accum}, eff_bs=$((bsz * accum)))"
        python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        trainer="$trainer" \
        task_name="$task_name" \
        model="$MODEL" \
        forget_split="$forget_split" \
        retain_split="$retain_split" \
        model.model_args.pretrained_model_name_or_path="$MODEL_PATH" \
        retain_logs_path="$retain_logs" \
        trainer.args.per_device_train_batch_size=$bsz \
        trainer.args.gradient_accumulation_steps=$accum \
        trainer.args.num_train_epochs=$EPOCHS \
        trainer.args.learning_rate=$LR \
        trainer.args.gradient_checkpointing=true \
        "$@" 2>&1 | tail -20
    fi

    log "[EVAL] $task_name"
    python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split="$forget_split" \
        holdout_split="$holdout_split" \
        model="$MODEL" \
        task_name="$task_name" \
        model.model_args.pretrained_model_name_or_path="$outdir" \
        paths.output_dir="$outdir/evals" \
        retain_logs_path="$retain_logs" 2>&1 | tail -10

    log "[DONE] $task_name"

    if [[ -f "$outdir/evals/TOFU_SUMMARY.json" ]]; then
        log "[RESULT] $task_name:"
        cat "$outdir/evals/TOFU_SUMMARY.json" | tee -a "$PROGRESS"
        echo "" | tee -a "$PROGRESS"
    fi
}

# ═══════════════════════════════════════════════════════════════
# Splits config
# ═══════════════════════════════════════════════════════════════
SPLITS=(
    "forget01 retain99 holdout01"
    "forget05 retain95 holdout05"
    "forget10 retain90 holdout10"
)

log "═══════════════════════════════════════════════════════════════"
log "  TOFU All Baselines — ${MODEL}"
log "  7 methods × 3 splits, ${EPOCHS} epochs, lr=${LR}"
log "═══════════════════════════════════════════════════════════════"

for split_cfg in "${SPLITS[@]}"; do
    forget_split=$(echo $split_cfg | cut -d' ' -f1)
    retain_split=$(echo $split_cfg | cut -d' ' -f2)
    holdout_split=$(echo $split_cfg | cut -d' ' -f3)

    log "────────────────────────────────────────────"
    log "  Split: $forget_split / $retain_split"
    log "────────────────────────────────────────────"

    # 1. GradAscent — bs=12, accum=1 (eff_bs=12, standard)
    run_baseline GradAscent "$forget_split" "$retain_split" "$holdout_split" 12 1

    # 2. GradDiff — bs=12, accum=1 (eff_bs=12, standard)
    run_baseline GradDiff "$forget_split" "$retain_split" "$holdout_split" 12 1

    # 3. NPO — bs=12, accum=1 (eff_bs=12, standard)
    run_baseline NPO "$forget_split" "$retain_split" "$holdout_split" 12 1

    # 4. SimNPO — bs=12, accum=1 (eff_bs=12, standard)
    run_baseline SimNPO "$forget_split" "$retain_split" "$holdout_split" 12 1

    # 5. RMU — bs=12, accum=1 (eff_bs=12, standard)
    run_baseline RMU "$forget_split" "$retain_split" "$holdout_split" 12 1

    # 6. BLURNPO — needs bs=1 with higher accum
    run_baseline BLUR_NPO "$forget_split" "$retain_split" "$holdout_split" 1 12

    # 7. PDU — paper params: alpha=100, eps=0.3, dual_step_size=5, warmup=5
    run_baseline PDU "$forget_split" "$retain_split" "$holdout_split" 4 3 \
        trainer.method_args.alpha=100 \
        trainer.method_args.retain_loss_eps=0.3 \
        trainer.method_args.dual_step_size=5 \
        trainer.method_args.dual_warmup_epochs=5

done

log "═══════════════════════════════════════════════════════════════"
log "  All TOFU baselines done! (7 methods × 3 splits)"
log "═══════════════════════════════════════════════════════════════"
