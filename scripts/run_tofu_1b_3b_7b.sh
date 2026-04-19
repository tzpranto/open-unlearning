#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   TOFU All Baselines: 7 methods × 3 splits × 3 models
#   Order: finish 7B forget01 → all 1B → all 3B → 7B forget05/10
#   Single GPU, restart-safe (skip if TOFU_SUMMARY.json exists)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="/datadrive/forked/open-unlearning"
cd "$BASE"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_tofu_master_progress.log"
EPOCHS=10
LR=1e-5

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

run_baseline() {
    local model_name="$1"; shift
    local model_path="$1"; shift
    local trainer="$1"; shift
    local forget_split="$1"; shift
    local retain_split="$1"; shift
    local holdout_split="$1"; shift
    local bsz="$1"; shift
    local accum="$1"; shift
    local task_name="tofu_${model_name}_${forget_split}_${trainer}"
    local outdir="$SAVES/$task_name"
    local retain_logs="saves/eval/tofu_${model_name}_${retain_split}/TOFU_EVAL.json"

    if [[ -f "$outdir/evals/TOFU_SUMMARY.json" ]]; then
        log "[SKIP] $task_name — already done"
        return 0
    fi

    if compgen -G "$outdir/model-*.safetensors" > /dev/null 2>&1 || [[ -f "$outdir/model.safetensors" ]]; then
        log "[SKIP-TRAIN] $task_name — model exists, running eval only"
    else
        log "[TRAIN] $task_name (trainer=$trainer, model=$model_name, split=$forget_split, bs=${bsz}x${accum})"
        python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        trainer="$trainer" \
        task_name="$task_name" \
        model="$model_name" \
        forget_split="$forget_split" \
        retain_split="$retain_split" \
        model.model_args.pretrained_model_name_or_path="$model_path" \
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
        model="$model_name" \
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

run_all_methods() {
    local model_name="$1"
    local model_path="$2"
    local forget_split="$3"
    local retain_split="$4"
    local holdout_split="$5"

    log "────────────────────────────────────────────"
    log "  ${model_name} — ${forget_split} / ${retain_split}"
    log "────────────────────────────────────────────"

    run_baseline "$model_name" "$model_path" GradAscent "$forget_split" "$retain_split" "$holdout_split" 8 4
    run_baseline "$model_name" "$model_path" GradDiff "$forget_split" "$retain_split" "$holdout_split" 8 4
    run_baseline "$model_name" "$model_path" NPO "$forget_split" "$retain_split" "$holdout_split" 8 4
    run_baseline "$model_name" "$model_path" SimNPO "$forget_split" "$retain_split" "$holdout_split" 8 4
    run_baseline "$model_name" "$model_path" RMU "$forget_split" "$retain_split" "$holdout_split" 8 4
    run_baseline "$model_name" "$model_path" BLURNPO "$forget_split" "$retain_split" "$holdout_split" 1 32
    run_baseline "$model_name" "$model_path" PDU "$forget_split" "$retain_split" "$holdout_split" 8 4 \
        trainer.method_args.alpha=100 \
        trainer.method_args.retain_loss_eps=0.3 \
        trainer.method_args.dual_step_size=5 \
        trainer.method_args.dual_warmup_epochs=5
}

SPLITS=(
    "forget01 retain99 holdout01"
    "forget05 retain95 holdout05"
    "forget10 retain90 holdout10"
)

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Finish 7B forget01
# ═══════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════"
log "  PHASE 1: Finish 7B forget01"
log "═══════════════════════════════════════════════════════════════"
run_all_methods "Llama-2-7b-chat-hf" "open-unlearning/tofu_Llama-2-7b-chat-hf_full" "forget01" "retain99" "holdout01"

# ═══════════════════════════════════════════════════════════════
# PHASE 2: All 1B (3 splits)
# ═══════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════"
log "  PHASE 2: Llama-3.2-1B-Instruct — all splits"
log "═══════════════════════════════════════════════════════════════"
for split_cfg in "${SPLITS[@]}"; do
    forget_split=$(echo $split_cfg | cut -d' ' -f1)
    retain_split=$(echo $split_cfg | cut -d' ' -f2)
    holdout_split=$(echo $split_cfg | cut -d' ' -f3)
    run_all_methods "Llama-3.2-1B-Instruct" "open-unlearning/tofu_Llama-3.2-1B-Instruct_full" "$forget_split" "$retain_split" "$holdout_split"
done

# ═══════════════════════════════════════════════════════════════
# PHASE 3: All 3B (3 splits)
# ═══════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════"
log "  PHASE 3: Llama-3.2-3B-Instruct — all splits"
log "═══════════════════════════════════════════════════════════════"
for split_cfg in "${SPLITS[@]}"; do
    forget_split=$(echo $split_cfg | cut -d' ' -f1)
    retain_split=$(echo $split_cfg | cut -d' ' -f2)
    holdout_split=$(echo $split_cfg | cut -d' ' -f3)
    run_all_methods "Llama-3.2-3B-Instruct" "open-unlearning/tofu_Llama-3.2-3B-Instruct_full" "$forget_split" "$retain_split" "$holdout_split"
done

# ═══════════════════════════════════════════════════════════════
# PHASE 4: 7B forget05 + forget10
# ═══════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════"
log "  PHASE 4: 7B remaining splits (forget05, forget10)"
log "═══════════════════════════════════════════════════════════════"
for split_cfg in "forget05 retain95 holdout05" "forget10 retain90 holdout10"; do
    forget_split=$(echo $split_cfg | cut -d' ' -f1)
    retain_split=$(echo $split_cfg | cut -d' ' -f2)
    holdout_split=$(echo $split_cfg | cut -d' ' -f3)
    run_all_methods "Llama-2-7b-chat-hf" "open-unlearning/tofu_Llama-2-7b-chat-hf_full" "$forget_split" "$retain_split" "$holdout_split"
done

log "═══════════════════════════════════════════════════════════════"
log "  ALL TOFU BASELINES COMPLETE (3 models × 7 methods × 3 splits = 63 runs)"
log "═══════════════════════════════════════════════════════════════"
