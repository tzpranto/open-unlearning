#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   TOFU Baselines: 7 methods × forget01 × (1B, 3B)
#   Effective batch size: 16
#   Single GPU, restart-safe (skip if TOFU_SUMMARY.json exists)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="/datadrive/forked/open-unlearning"
cd "$BASE"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_tofu_1b3b_bs16_progress.log"
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
    local task_name="tofu_${model_name}_${forget_split}_${trainer}_bs16"
    local outdir="$SAVES/$task_name"
    local retain_logs="saves/eval/tofu_${model_name}_${retain_split}/TOFU_EVAL.json"

    if [[ -f "$outdir/evals/TOFU_SUMMARY.json" ]]; then
        log "[SKIP] $task_name — already done"
        return 0
    fi

    if compgen -G "$outdir/model-*.safetensors" > /dev/null 2>&1 || [[ -f "$outdir/model.safetensors" ]]; then
        log "[SKIP-TRAIN] $task_name — model exists, running eval only"
    else
        log "[TRAIN] $task_name (trainer=$trainer, model=$model_name, split=$forget_split, bs=${bsz}x${accum}, eff_bs=$((bsz * accum)))"
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
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
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
    local bsz="$6"
    local accum="$7"

    log "────────────────────────────────────────────"
    log "  ${model_name} — ${forget_split}/${retain_split} — eff_bs=$((bsz * accum))"
    log "────────────────────────────────────────────"

    run_baseline "$model_name" "$model_path" GradAscent "$forget_split" "$retain_split" "$holdout_split" $bsz $accum
    run_baseline "$model_name" "$model_path" GradDiff "$forget_split" "$retain_split" "$holdout_split" $bsz $accum
    run_baseline "$model_name" "$model_path" NPO "$forget_split" "$retain_split" "$holdout_split" $bsz $accum
    run_baseline "$model_name" "$model_path" SimNPO "$forget_split" "$retain_split" "$holdout_split" $bsz $accum
    run_baseline "$model_name" "$model_path" RMU "$forget_split" "$retain_split" "$holdout_split" $bsz $accum
    run_baseline "$model_name" "$model_path" BLURNPO "$forget_split" "$retain_split" "$holdout_split" 1 16
    run_baseline "$model_name" "$model_path" PDU "$forget_split" "$retain_split" "$holdout_split" $bsz $accum \
        trainer.method_args.alpha=100 \
        trainer.method_args.retain_loss_eps=0.3 \
        trainer.method_args.dual_step_size=5 \
        trainer.method_args.dual_warmup_epochs=5
}

log "═══════════════════════════════════════════════════════════════"
log "  TOFU Baselines — 1B + 3B — eff_bs=16 — forget01"
log "═══════════════════════════════════════════════════════════════"

# 1B: micro_bs=8, accum=2 → eff_bs=16
run_all_methods "Llama-3.2-1B-Instruct" "open-unlearning/tofu_Llama-3.2-1B-Instruct_full" \
    "forget01" "retain99" "holdout01" 8 2

# 3B: micro_bs=4, accum=4 → eff_bs=16
run_all_methods "Llama-3.2-3B-Instruct" "open-unlearning/tofu_Llama-3.2-3B-Instruct_full" \
    "forget01" "retain99" "holdout01" 4 4

log "═══════════════════════════════════════════════════════════════"
log "  ALL DONE: 7 methods × 2 models = 14 runs"
log "═══════════════════════════════════════════════════════════════"
