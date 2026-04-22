#!/bin/bash
# Queue: 7B baselines → 7B LoRA-BiAL tuning → MUSE Books ours report
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="/datadrive/forked/open-unlearning"
cd "$BASE"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_next_queue_progress.log"
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
        log "[TRAIN] $task_name (bs=${bsz}x${accum})"
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

run_lora_bial() {
    local config="$1"; shift
    local model_name="$1"; shift
    local task="$1"; shift
    local retain_logs="$1"; shift
    local outdir="$SAVES/$task"

    if [[ -f "$outdir/evals/TOFU_SUMMARY.json" ]]; then
        log "[SKIP] $task — already done"
        cat "$outdir/evals/TOFU_SUMMARY.json" | tee -a "$PROGRESS"
        return 0
    fi

    log "[TRAIN] $task"
    python src/train.py --config-name=unlearn.yaml \
        experiment="$config" \
        task_name="$task" \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        retain_logs_path="$retain_logs" \
        "$@"

    log "[EVAL] $task"
    local forget_split="forget01"
    local holdout_split="holdout01"
    python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split="$forget_split" \
        holdout_split="$holdout_split" \
        model="$model_name" \
        task_name="$task" \
        model.model_args.pretrained_model_name_or_path="$outdir" \
        paths.output_dir="$outdir/evals" \
        retain_logs_path="$retain_logs"

    log "[RESULT] $task:"
    cat "$outdir/evals/TOFU_SUMMARY.json" | tee -a "$PROGRESS"
    echo "" | tee -a "$PROGRESS"
}

# ═══════════════════════════════════════════════════════════════
# PHASE 1: 7B baselines (forget01, eff_bs=16)
# 7B: micro_bs=2, accum=8 → eff_bs=16
# ═══════════════════════════════════════════════════════════════
MODEL_7B="Llama-2-7b-chat-hf"
MODEL_7B_PATH="open-unlearning/tofu_Llama-2-7b-chat-hf_full"
RETAIN_7B="saves/eval/tofu_Llama-2-7b-chat-hf_retain99/TOFU_EVAL.json"

log "═══════════════════════════════════════════════════════════════"
log "  PHASE 1: 7B baselines — forget01 — eff_bs=16"
log "═══════════════════════════════════════════════════════════════"

run_baseline "$MODEL_7B" "$MODEL_7B_PATH" GradAscent forget01 retain99 holdout01 2 8
run_baseline "$MODEL_7B" "$MODEL_7B_PATH" GradDiff forget01 retain99 holdout01 2 8
run_baseline "$MODEL_7B" "$MODEL_7B_PATH" NPO forget01 retain99 holdout01 2 8
run_baseline "$MODEL_7B" "$MODEL_7B_PATH" SimNPO forget01 retain99 holdout01 2 8
run_baseline "$MODEL_7B" "$MODEL_7B_PATH" RMU forget01 retain99 holdout01 2 8
run_baseline "$MODEL_7B" "$MODEL_7B_PATH" BLURNPO forget01 retain99 holdout01 1 16
run_baseline "$MODEL_7B" "$MODEL_7B_PATH" PDU forget01 retain99 holdout01 2 8 \
    trainer.method_args.alpha=100 \
    trainer.method_args.retain_loss_eps=0.3 \
    trainer.method_args.dual_step_size=5 \
    trainer.method_args.dual_warmup_epochs=5

# ═══════════════════════════════════════════════════════════════
# PHASE 2: 7B LoRA-BiAL tuning
# ═══════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════"
log "  PHASE 2: 7B LoRA-BiAL tuning"
log "═══════════════════════════════════════════════════════════════"

run_lora_bial "unlearn/tofu/lora_bial_7b.yaml" "$MODEL_7B" \
    "tofu_Llama-2-7b-chat-hf_forget01_LoRABiAL_lr2e5" "$RETAIN_7B" \
    trainer.method_args.eta_theta=2e-5 trainer.method_args.T=100

run_lora_bial "unlearn/tofu/lora_bial_7b.yaml" "$MODEL_7B" \
    "tofu_Llama-2-7b-chat-hf_forget01_LoRABiAL_lr3e5" "$RETAIN_7B" \
    trainer.method_args.eta_theta=3e-5 trainer.method_args.T=100

run_lora_bial "unlearn/tofu/lora_bial_7b.yaml" "$MODEL_7B" \
    "tofu_Llama-2-7b-chat-hf_forget01_LoRABiAL_lr1e5" "$RETAIN_7B" \
    trainer.method_args.eta_theta=1e-5 trainer.method_args.T=100

# ═══════════════════════════════════════════════════════════════
# PHASE 3: MUSE Books — eval our champion (exp_18)
# ═══════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════"
log "  PHASE 3: MUSE Books — report our result"
log "═══════════════════════════════════════════════════════════════"

MUSE_BOOKS_DIR="saves/unlearn/muse_books_exp_18"
if [[ -d "$MUSE_BOOKS_DIR" ]]; then
    if [[ -f "$MUSE_BOOKS_DIR/evals/MUSE_EVAL.json" ]]; then
        log "[SKIP] MUSE Books exp_18 already evaluated"
    else
        log "[EVAL] MUSE Books exp_18"
        mkdir -p "$MUSE_BOOKS_DIR/evals"
        python src/eval.py \
            experiment=eval/muse/default.yaml \
            data_split=Books \
            model=Llama-2-7b-hf \
            task_name=muse_books_exp_18 \
            model.model_args.pretrained_model_name_or_path="$MUSE_BOOKS_DIR" \
            paths.output_dir="$MUSE_BOOKS_DIR/evals" \
            retain_logs_path=saves/eval/muse_Llama-2-7b-hf_Books_retrain/MUSE_EVAL.json 2>&1 | tail -10
    fi
    log "[RESULT] MUSE Books exp_18:"
    cat "$MUSE_BOOKS_DIR/evals/MUSE_SUMMARY.json" 2>/dev/null || echo "no summary found"
else
    log "[WARN] MUSE Books exp_18 dir not found at $MUSE_BOOKS_DIR"
fi

log "═══════════════════════════════════════════════════════════════"
log "  ALL DONE"
log "═══════════════════════════════════════════════════════════════"
