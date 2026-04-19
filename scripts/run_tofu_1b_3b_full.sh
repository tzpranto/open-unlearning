#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   TOFU Full Pipeline: Gold/Target sanity + 7 baselines
#   Models: Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct
#   Splits: forget01/retain99, forget05/retain95, forget10/retain90
#   Single GPU, restart-safe (skip if evals exist)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="/datadrive/forked/open-unlearning"
cd "$BASE"

SAVES_EVAL="saves/eval"
SAVES_UNLEARN="saves/unlearn"
PROGRESS="$SAVES_UNLEARN/_tofu_1b_3b_progress.log"

EPOCHS=10
LR=1e-5

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Gold/Target Sanity Evals
# Re-run eval calc on existing saved evals for both models
# ═══════════════════════════════════════════════════════════════

run_sanity_eval() {
    local model_short="$1"  # 1B or 3B
    local model_name="Llama-3.2-${model_short}-Instruct"
    local model_path="open-unlearning/tofu_${model_name}_full"
    local forget_split="$2"
    local holdout_split="$3"
    local retain_split="$4"
    local eval_type="$5"  # "full" or "retainXX"

    local task_name="tofu_${model_name}_${eval_type}"
    local eval_dir

    if [[ "$eval_type" == "full" ]]; then
        eval_dir="$SAVES_EVAL/$task_name/evals_${forget_split}"
        local eval_model_path="$model_path"
    else
        eval_dir="$SAVES_EVAL/$task_name"
        local eval_model_path="open-unlearning/${task_name}"
    fi

    local summary_file="$eval_dir/TOFU_SUMMARY.json"

    # For sanity re-eval, we want to re-run even if exists
    # But skip if we already did it THIS run (check a marker)
    local marker="$eval_dir/.sanity_done_$(date +%Y%m%d)"
    if [[ -f "$marker" ]]; then
        log "[SKIP-SANITY] $task_name ($forget_split) — already done today"
        return 0
    fi

    log "[SANITY-EVAL] $task_name ($forget_split)"

    local retain_logs="$SAVES_EVAL/tofu_${model_name}_${retain_split}/TOFU_EVAL.json"

    if [[ "$eval_type" == "full" ]]; then
        python src/eval.py \
            experiment=eval/tofu/default.yaml \
            model="$model_name" \
            forget_split="$forget_split" \
            holdout_split="$holdout_split" \
            task_name="$task_name" \
            model.model_args.pretrained_model_name_or_path="$model_path" \
            paths.output_dir="$eval_dir" \
            retain_logs_path="$retain_logs" \
            eval.tofu.overwrite=true 2>&1 | tail -5
    else
        python src/eval.py \
            experiment=eval/tofu/default.yaml \
            model="$model_name" \
            forget_split="$forget_split" \
            holdout_split="$holdout_split" \
            task_name="$task_name" \
            model.model_args.pretrained_model_name_or_path="$eval_model_path" \
            paths.output_dir="$eval_dir" \
            retain_logs_path="$retain_logs" \
            eval.tofu.overwrite=true 2>&1 | tail -5
    fi

    if [[ -f "$summary_file" ]]; then
        touch "$marker"
        log "[SANITY-OK] $task_name ($forget_split):"
        python3 -c "import json; d=json.load(open('$summary_file')); print(f'  FQ={d.get(\"forget_quality\",\"N/A\"):.4f}  MU={d.get(\"model_utility\",\"N/A\"):.4f}  ES={d.get(\"extraction_strength\",\"N/A\"):.4f}')" 2>/dev/null || cat "$summary_file"
    else
        log "[SANITY-FAIL] $task_name ($forget_split) — no summary produced"
    fi
}

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Baseline Training + Eval
# ═══════════════════════════════════════════════════════════════

run_baseline() {
    local model_short="$1"; shift  # 1B or 3B
    local trainer="$1"; shift
    local forget_split="$1"; shift
    local retain_split="$1"; shift
    local holdout_split="$1"; shift
    local bsz="$1"; shift
    local accum="$1"; shift
    # remaining args are extra overrides

    local model_name="Llama-3.2-${model_short}-Instruct"
    local model_path="open-unlearning/tofu_${model_name}_full"
    local task_name="tofu_${model_name}_${forget_split}_${trainer}"
    local outdir="$SAVES_UNLEARN/$task_name"
    local retain_logs="$SAVES_EVAL/tofu_${model_name}_${retain_split}/TOFU_EVAL.json"

    # Skip if already evaluated
    if [[ -f "$outdir/evals/TOFU_SUMMARY.json" ]]; then
        log "[SKIP] $task_name — already done"
        return 0
    fi

    # Resume from checkpoint if exists
    local resume_arg=""
    if [[ -d "$outdir" ]]; then
        local latest_ckpt
        latest_ckpt=$(ls -d "$outdir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1) || true
        if [[ -n "$latest_ckpt" ]]; then
            resume_arg="trainer.args.resume_from_checkpoint=$latest_ckpt"
            log "[RESUME] $task_name from $latest_ckpt"
        fi
    fi

    log "[TRAIN] $task_name (trainer=$trainer, split=$forget_split, bs=${bsz}x${accum}, eff_bs=$((bsz * accum)))"
    python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        trainer="$trainer" \
        model="$model_name" \
        task_name="$task_name" \
        forget_split="$forget_split" \
        retain_split="$retain_split" \
        model.model_args.pretrained_model_name_or_path="$model_path" \
        retain_logs_path="$retain_logs" \
        trainer.args.per_device_train_batch_size=$bsz \
        trainer.args.gradient_accumulation_steps=$accum \
        trainer.args.num_train_epochs=$EPOCHS \
        trainer.args.learning_rate=$LR \
        trainer.args.gradient_checkpointing=true \
        $resume_arg \
        "$@" 2>&1 | tail -20

    log "[EVAL] $task_name"
    python src/eval.py \
        experiment=eval/tofu/default.yaml \
        model="$model_name" \
        forget_split="$forget_split" \
        holdout_split="$holdout_split" \
        task_name="$task_name" \
        model.model_args.pretrained_model_name_or_path="$outdir" \
        paths.output_dir="$outdir/evals" \
        retain_logs_path="$retain_logs" 2>&1 | tail -10

    if [[ -f "$outdir/evals/TOFU_SUMMARY.json" ]]; then
        log "[DONE] $task_name"
        python3 -c "import json; d=json.load(open('$outdir/evals/TOFU_SUMMARY.json')); print(f'  FQ={d.get(\"forget_quality\",\"N/A\"):.4f}  MU={d.get(\"model_utility\",\"N/A\"):.4f}  ES={d.get(\"extraction_strength\",\"N/A\"):.4f}')" 2>&1 | tee -a "$PROGRESS"
    else
        log "[WARN] $task_name — eval produced no summary"
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

MODELS=("1B" "3B")

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Sanity evals
# ═══════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════"
log "  TOFU Full Pipeline — 1B + 3B"
log "  Phase 1: Gold/Target sanity evals"
log "═══════════════════════════════════════════════════════════════"

for model_short in "${MODELS[@]}"; do
    model_name="Llama-3.2-${model_short}-Instruct"
    log "── Model: $model_name ──"

    for split_cfg in "${SPLITS[@]}"; do
        forget_split=$(echo $split_cfg | cut -d' ' -f1)
        retain_split=$(echo $split_cfg | cut -d' ' -f2)
        holdout_split=$(echo $split_cfg | cut -d' ' -f3)

        # Target (full model, no unlearning)
        run_sanity_eval "$model_short" "$forget_split" "$holdout_split" "$retain_split" "full"

        # Gold (retrain on retain set only)
        run_sanity_eval "$model_short" "$forget_split" "$holdout_split" "$retain_split" "$retain_split"
    done
done

log "═══════════════════════════════════════════════════════════════"
log "  Phase 1 COMPLETE — sanity evals done"
log "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Baselines
# ═══════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════"
log "  Phase 2: 7 baselines × 3 splits × 2 models"
log "═══════════════════════════════════════════════════════════════"

for model_short in "${MODELS[@]}"; do
    model_name="Llama-3.2-${model_short}-Instruct"
    log "══════════════════════════════════════════"
    log "  Model: $model_name"
    log "══════════════════════════════════════════"

    for split_cfg in "${SPLITS[@]}"; do
        forget_split=$(echo $split_cfg | cut -d' ' -f1)
        retain_split=$(echo $split_cfg | cut -d' ' -f2)
        holdout_split=$(echo $split_cfg | cut -d' ' -f3)

        log "────────────────────────────────────────────"
        log "  Split: $forget_split / $retain_split"
        log "────────────────────────────────────────────"

        # 1. GradAscent — bs=12, accum=1
        run_baseline "$model_short" GradAscent "$forget_split" "$retain_split" "$holdout_split" 12 1

        # 2. GradDiff — bs=12, accum=1
        run_baseline "$model_short" GradDiff "$forget_split" "$retain_split" "$holdout_split" 12 1

        # 3. NPO — bs=12, accum=1
        run_baseline "$model_short" NPO "$forget_split" "$retain_split" "$holdout_split" 12 1

        # 4. SimNPO — bs=12, accum=1
        run_baseline "$model_short" SimNPO "$forget_split" "$retain_split" "$holdout_split" 12 1

        # 5. RMU — bs=12, accum=1
        run_baseline "$model_short" RMU "$forget_split" "$retain_split" "$holdout_split" 12 1

        # 6. BLURNPO — needs bs=1 with higher accum
        run_baseline "$model_short" BLURNPO "$forget_split" "$retain_split" "$holdout_split" 1 12

        # 7. PDU — paper params
        run_baseline "$model_short" PDU "$forget_split" "$retain_split" "$holdout_split" 4 3 \
            trainer.method_args.alpha=100 \
            trainer.method_args.retain_loss_eps=0.3 \
            trainer.method_args.dual_step_size=5 \
            trainer.method_args.dual_warmup_epochs=5
    done
done

log "═══════════════════════════════════════════════════════════════"
log "  ALL DONE — TOFU 1B + 3B baselines complete"
log "═══════════════════════════════════════════════════════════════"
