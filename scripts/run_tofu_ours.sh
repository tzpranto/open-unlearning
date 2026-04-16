#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   TOFU Forget 5%: LoRA-Implicit (our method)
#   Two-phase approach:
#     Phase 1: logit_margin to break cold start (high LR)
#     Phase 2: NPO on phase1 checkpoint (bilevel with implicit)
#
#   forget05: 200/12≈17 steps/epoch
#   Intermediate checkpoints at epoch boundaries
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_tofu_ours_progress.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

RETAIN_LOGS_05="saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json"

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
        experiment=unlearn/tofu/lora_implicit \
        forget_split=forget05 \
        retain_split=retain95 \
        holdout_split=holdout05 \
        task_name="$name" \
        retain_logs_path="$RETAIN_LOGS_05" \
        "${COMMON[@]}" \
        "$@" 2>&1 | tail -30

    log "[DONE] $name"

    if [[ -f "$outdir/evals/TOFU_SUMMARY.json" ]]; then
        log "[RESULT] $name:"
        cat "$outdir/evals/TOFU_SUMMARY.json" | tee -a "$PROGRESS"
        echo "" | tee -a "$PROGRESS"
    fi
}

log "═══════════════════════════════════════════════════════════════"
log "  TOFU forget05 — LoRA-Implicit (our method)"
log "═══════════════════════════════════════════════════════════════"

# ─── Phase 1: logit_margin cold-start breaker ────────────────
# Margin ~24 on TOFU → strong gradient from step 0
# K=1 (inner loop useless when retain CE=0.001)
# High LR to actually move the parameters
# 2 epochs = ~34 steps, checkpoint each epoch
run_exp tofu_f05_phase1_lm \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.K=1 \
    trainer.method_args.eta_theta=5e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.05 \
    trainer.method_args.lambda_max=5.0 \
    trainer.method_args.use_implicit=false \
    trainer.method_args.adaptive_lr=false \
    trainer.method_args.eval_at_steps=[17,34] \
    trainer.args.num_train_epochs=2

# ─── Phase 2: NPO on phase1 checkpoint ───────────────────────
# Load phase1 merged model, wrap with fresh LoRA
# Now model≠ref → NPO has gradient signal
# K=3, implicit correction ON (retain CE should be > 0.001 now)
run_exp tofu_f05_phase2_npo \
    model.model_args.pretrained_model_name_or_path="$SAVES/tofu_f05_phase1_lm" \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=4.0 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.05 \
    trainer.method_args.lambda_max=5.0 \
    trainer.method_args.use_implicit=true \
    trainer.method_args.implicit_warmup_steps=10 \
    trainer.method_args.adaptive_lr=false \
    trainer.method_args.eval_at_steps=[17,34,51] \
    trainer.args.num_train_epochs=3

# ─── Ablation: NPO from scratch (for comparison) ────────────
# Direct NPO without phase 1 warmup — expect cold start
run_exp tofu_f05_npo_direct \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=4.0 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.05 \
    trainer.method_args.lambda_max=5.0 \
    trainer.method_args.use_implicit=true \
    trainer.method_args.implicit_warmup_steps=10 \
    trainer.method_args.adaptive_lr=false \
    trainer.method_args.eval_at_steps=[17,34,51,68] \
    trainer.args.num_train_epochs=5

log "═══════════════════════════════════════════════════════════════"
log "  All LoRA-Implicit experiments done!"
log "═══════════════════════════════════════════════════════════════"
