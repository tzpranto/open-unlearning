#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   TOFU: LoRA-Implicit — batch=8, K=5, implicit from step 0
#   Step 1: Basic version, then trace analysis
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_tofu_progress.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

COMMON=(
    trainer.args.per_device_train_batch_size=8
    trainer.args.gradient_accumulation_steps=1
    trainer.method_args.forget_loss_type=logit_margin
    trainer.method_args.epsilon=0.05
    trainer.method_args.lambda_max=20.0
    trainer.method_args.K=5
    trainer.method_args.use_implicit=true
    trainer.method_args.implicit_warmup_steps=0
    trainer.method_args.eta_theta=5e-5
)

run_exp() {
    local name="$1"; shift
    local forget_split="$1"; shift
    local retain_split="$1"; shift
    local holdout_split="$1"; shift
    local outdir="$SAVES/$name"

    if [[ -d "$outdir/evals" ]]; then
        log "[SKIP] $name — done"
        return
    fi
    rm -rf "$outdir" 2>/dev/null

    log "[TRAIN] $name"
    python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/lora_implicit \
        forget_split="$forget_split" \
        retain_split="$retain_split" \
        holdout_split="$holdout_split" \
        task_name="$name" \
        "${COMMON[@]}" \
        "$@" 2>&1 | tail -5

    log "[DONE] $name"
}

log "═══════════════════════════════════════════════════════════════"
log "  TOFU — batch=8, K=5, implicit@0, logit_margin"
log "═══════════════════════════════════════════════════════════════"

RETAIN_LOGS_01="saves/eval/tofu_Llama-2-7b-chat-hf_retain99/TOFU_EVAL.json"
RETAIN_LOGS_05="saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json"

# ─── Basic versions ──────────────────────────────────────────
# forget01: 40/8=5 steps/ep, 10ep=50 steps, K=5 -> 250 inner
run_exp tofu_f01_basic forget01 retain99 holdout01 \
    retain_logs_path="$RETAIN_LOGS_01" \
    trainer.args.num_train_epochs=10

# forget05: 200/8=25 steps/ep, 3ep=75 steps, K=5 -> 375 inner
run_exp tofu_f05_basic forget05 retain95 holdout05 \
    retain_logs_path="$RETAIN_LOGS_05" \
    trainer.args.num_train_epochs=3

log "═══════════════════════════════════════════════════════════════"
log "  Basic runs done — check traces before continuing"
log "═══════════════════════════════════════════════════════════════"
