#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   TOFU: All baselines + LoRA-BiAL × 3 models × forget01
#   eff_bs=32 (matching official repro setup)
#   Single GPU, restart-safe
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="/datadrive/forked/open-unlearning"
cd "$BASE"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_tofu_bs32_progress.log"
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
        log "[TRAIN] $task_name (bs=${bsz}x${accum}, eff_bs=$((bsz * accum)))"
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

    # Clean checkpoints to save disk
    rm -rf "$outdir"/checkpoint-* 2>/dev/null
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

    log "[TRAIN] $task (LoRA-BiAL)"
    python src/train.py --config-name=unlearn.yaml \
        experiment="$config" \
        task_name="$task" \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.method_args.checkpoint_every_epoch=false \
        retain_logs_path="$retain_logs" \
        "$@"

    log "[EVAL] $task"
    python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=forget01 \
        holdout_split=holdout01 \
        model="$model_name" \
        task_name="$task" \
        model.model_args.pretrained_model_name_or_path="$outdir" \
        paths.output_dir="$outdir/evals" \
        retain_logs_path="$retain_logs"

    log "[RESULT] $task:"
    cat "$outdir/evals/TOFU_SUMMARY.json" | tee -a "$PROGRESS"
    echo "" | tee -a "$PROGRESS"

    # Clean checkpoints
    rm -rf "$outdir"/checkpoint-* 2>/dev/null
}

run_all_baselines() {
    local model_name="$1"
    local model_path="$2"
    local bsz="$3"
    local accum="$4"
    local blur_accum="$5"

    log "────────────────────────────────────────────"
    log "  ${model_name} — forget01 — eff_bs=$((bsz * accum))"
    log "────────────────────────────────────────────"

    run_baseline "$model_name" "$model_path" GradAscent forget01 retain99 holdout01 $bsz $accum
    run_baseline "$model_name" "$model_path" GradDiff forget01 retain99 holdout01 $bsz $accum
    run_baseline "$model_name" "$model_path" NPO forget01 retain99 holdout01 $bsz $accum
    run_baseline "$model_name" "$model_path" SimNPO forget01 retain99 holdout01 $bsz $accum
    run_baseline "$model_name" "$model_path" RMU forget01 retain99 holdout01 $bsz $accum
    run_baseline "$model_name" "$model_path" BLURNPO forget01 retain99 holdout01 1 $blur_accum
    run_baseline "$model_name" "$model_path" PDU forget01 retain99 holdout01 $bsz $accum \
        trainer.method_args.alpha=100 \
        trainer.method_args.retain_loss_eps=0.3 \
        trainer.method_args.dual_step_size=5 \
        trainer.method_args.dual_warmup_epochs=5
}

log "═══════════════════════════════════════════════════════════════"
log "  TOFU All — eff_bs=32 — forget01"
log "═══════════════════════════════════════════════════════════════"

RETAIN_1B="saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json"
RETAIN_3B="saves/eval/tofu_Llama-3.2-3B-Instruct_retain99/TOFU_EVAL.json"
RETAIN_7B="saves/eval/tofu_Llama-2-7b-chat-hf_retain99/TOFU_EVAL.json"

# ═══════════════════════════════════════════════════════════════
# 1B: bs=8, accum=4 → eff_bs=32
# ═══════════════════════════════════════════════════════════════
run_all_baselines "Llama-3.2-1B-Instruct" "open-unlearning/tofu_Llama-3.2-1B-Instruct_full" 8 4 32

# LoRA-BiAL 1B: sweep eta_theta
run_lora_bial "unlearn/tofu/lora_bial_1b.yaml" "Llama-3.2-1B-Instruct" \
    "tofu_Llama-3.2-1B-Instruct_forget01_LoRABiAL_lr2e5" "$RETAIN_1B" \
    trainer.method_args.eta_theta=2e-5 trainer.method_args.T=100

run_lora_bial "unlearn/tofu/lora_bial_1b.yaml" "Llama-3.2-1B-Instruct" \
    "tofu_Llama-3.2-1B-Instruct_forget01_LoRABiAL_lr3e5" "$RETAIN_1B" \
    trainer.method_args.eta_theta=3e-5 trainer.method_args.T=100

run_lora_bial "unlearn/tofu/lora_bial_1b.yaml" "Llama-3.2-1B-Instruct" \
    "tofu_Llama-3.2-1B-Instruct_forget01_LoRABiAL_lr1e5" "$RETAIN_1B" \
    trainer.method_args.eta_theta=1e-5 trainer.method_args.T=100

# ═══════════════════════════════════════════════════════════════
# 3B: bs=4, accum=8 → eff_bs=32
# ═══════════════════════════════════════════════════════════════
run_all_baselines "Llama-3.2-3B-Instruct" "open-unlearning/tofu_Llama-3.2-3B-Instruct_full" 4 8 32

# LoRA-BiAL 3B: sweep eta_theta
run_lora_bial "unlearn/tofu/lora_bial_3b.yaml" "Llama-3.2-3B-Instruct" \
    "tofu_Llama-3.2-3B-Instruct_forget01_LoRABiAL_lr2e5" "$RETAIN_3B" \
    trainer.method_args.eta_theta=2e-5 trainer.method_args.T=100

run_lora_bial "unlearn/tofu/lora_bial_3b.yaml" "Llama-3.2-3B-Instruct" \
    "tofu_Llama-3.2-3B-Instruct_forget01_LoRABiAL_lr3e5" "$RETAIN_3B" \
    trainer.method_args.eta_theta=3e-5 trainer.method_args.T=100

run_lora_bial "unlearn/tofu/lora_bial_3b.yaml" "Llama-3.2-3B-Instruct" \
    "tofu_Llama-3.2-3B-Instruct_forget01_LoRABiAL_lr1e5" "$RETAIN_3B" \
    trainer.method_args.eta_theta=1e-5 trainer.method_args.T=100

# ═══════════════════════════════════════════════════════════════
# 7B: bs=2, accum=16 → eff_bs=32
# ═══════════════════════════════════════════════════════════════
run_all_baselines "Llama-2-7b-chat-hf" "open-unlearning/tofu_Llama-2-7b-chat-hf_full" 2 16 32

# LoRA-BiAL 7B: sweep eta_theta
run_lora_bial "unlearn/tofu/lora_bial_7b.yaml" "Llama-2-7b-chat-hf" \
    "tofu_Llama-2-7b-chat-hf_forget01_LoRABiAL_lr2e5" "$RETAIN_7B" \
    trainer.method_args.eta_theta=2e-5 trainer.method_args.T=100

run_lora_bial "unlearn/tofu/lora_bial_7b.yaml" "Llama-2-7b-chat-hf" \
    "tofu_Llama-2-7b-chat-hf_forget01_LoRABiAL_lr3e5" "$RETAIN_7B" \
    trainer.method_args.eta_theta=3e-5 trainer.method_args.T=100

run_lora_bial "unlearn/tofu/lora_bial_7b.yaml" "Llama-2-7b-chat-hf" \
    "tofu_Llama-2-7b-chat-hf_forget01_LoRABiAL_lr1e5" "$RETAIN_7B" \
    trainer.method_args.eta_theta=1e-5 trainer.method_args.T=100

log "═══════════════════════════════════════════════════════════════"
log "  ALL DONE: 3 models × (7 baselines + 3 LoRA-BiAL) = 30 runs"
log "═══════════════════════════════════════════════════════════════"
