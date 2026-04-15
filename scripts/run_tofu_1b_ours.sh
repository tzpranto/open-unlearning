#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   TOFU Forget 1%: LoRA-Implicit on Llama-3.2-1B-Instruct
#   Two-phase: logit_margin cold-start + NPO bilevel
#   Comparable to PerTA paper Table 2
#
#   forget01: 40/12≈4 steps/epoch
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_tofu_1b_progress.log"

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
        "$@" 2>&1 | tail -50

    log "[DONE TRAIN] $name"

    if [[ -f "$outdir/evals/TOFU_SUMMARY.json" ]]; then
        log "[RESULT] $name:"
        cat "$outdir/evals/TOFU_SUMMARY.json" | tee -a "$PROGRESS"
        echo "" | tee -a "$PROGRESS"
    fi
}

log "═══════════════════════════════════════════════════════════════"
log "  TOFU forget01 — Llama-3.2-1B-Instruct — 2-phase"
log "═══════════════════════════════════════════════════════════════"

# ─── Phase 1: logit_margin cold-start breaker ────────────────
# 40/12≈4 steps/epoch, 5 epochs = 20 steps
# K=1 (inner useless at start), high LR
run_exp tofu_1b_f01_phase1 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.K=1 \
    trainer.method_args.eta_theta=5e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.05 \
    trainer.method_args.lambda_max=5.0 \
    trainer.method_args.use_implicit=false \
    trainer.method_args.adaptive_lr=false \
    trainer.method_args.eval_at_steps=[4,8,12,16,20] \
    trainer.args.num_train_epochs=5

# ─── Phase 2: NPO bilevel on phase1 checkpoint ───────────────
# Fresh LoRA on phase1 model, NPO ref = phase1 model
# K=3, implicit ON, 5 epochs = 20 steps
run_exp tofu_1b_f01_phase2 \
    model.model_args.pretrained_model_name_or_path="$SAVES/tofu_1b_f01_phase1" \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=4.0 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.05 \
    trainer.method_args.lambda_max=5.0 \
    trainer.method_args.use_implicit=true \
    trainer.method_args.implicit_warmup_steps=5 \
    trainer.method_args.adaptive_lr=false \
    trainer.method_args.eval_at_steps=[4,8,12,16,20] \
    trainer.args.num_train_epochs=5

log "═══════════════════════════════════════════════════════════════"
log "  Done!"
log "═══════════════════════════════════════════════════════════════"
