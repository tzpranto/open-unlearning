#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   TOFU Forget 1%: LoRA-Implicit v2 — Push FQ harder
#   Llama-3.2-1B-Instruct
#
#   Key changes from v1:
#     - K=3 (3 inner retain steps = much stronger MU protection)
#     - logit_margin throughout (no NPO)
#     - No implicit correction
#     - Lower LR (2e-4) for finer grain
#     - Dense eval every 2 steps to find sweet spot
#     - More epochs (10) since LR is lower
#
#   40/12 ≈ 4 steps/epoch, 10 epochs = 40 steps
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_tofu_1b_v2_progress.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

RETAIN_LOGS="saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json"

COMMON=(
    trainer.args.per_device_train_batch_size=12
    trainer.args.gradient_accumulation_steps=1
    trainer.args.gradient_checkpointing=true
    trainer.method_args.checkpoint_every_epoch=true
)

run_exp() {
    local name="$1"; shift
    local outdir="$SAVES/$name"

    if [[ -d "$outdir/evals" ]]; then
        log "[SKIP] $name — done"
        return
    fi
    rm -rf "$outdir" 2>/dev/null

    log "[TRAIN] $name"
    python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/lora_implicit_1b \
        forget_split=forget01 \
        retain_split=retain99 \
        holdout_split=holdout01 \
        task_name="$name" \
        retain_logs_path="$RETAIN_LOGS" \
        "${COMMON[@]}" \
        "$@" 2>&1 | tail -80

    log "[DONE TRAIN] $name"

    if [[ -f "$outdir/evals/TOFU_SUMMARY.json" ]]; then
        log "[RESULT] $name:"
        cat "$outdir/evals/TOFU_SUMMARY.json" | tee -a "$PROGRESS"
        echo "" | tee -a "$PROGRESS"
    fi
}

log "═══════════════════════════════════════════════════════════════"
log "  TOFU forget01 — 1B — v2: K=3 logit_margin, no implicit"
log "═══════════════════════════════════════════════════════════════"

# ─── Phase 1: logit_margin, K=3 retain protection, no implicit ──
# 4 steps/epoch × 10 epochs = 40 steps
# Dense eval: every 2 steps for first 16, then every 4
run_exp tofu_1b_v2_p1 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=2e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.05 \
    trainer.method_args.lambda_max=5.0 \
    trainer.method_args.use_implicit=false \
    trainer.method_args.adaptive_lr=false \
    trainer.method_args.eval_at_steps=[2,4,6,8,10,12,14,16,20,24,28,32,36,40] \
    trainer.args.num_train_epochs=10

# ─── Phase 2: logit_margin bilevel from best checkpoint ──────────
# Will be run after inspecting Phase 1 results
# Phase 2: fresh LoRA on best p1 checkpoint, K=5, lower LR
# (placeholder — update model path after Phase 1)

log "═══════════════════════════════════════════════════════════════"
log "  Phase 1 done — check intermediate evals"
log "═══════════════════════════════════════════════════════════════"
