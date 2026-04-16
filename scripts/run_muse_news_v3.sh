#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   MUSE News v3: Graduated Schedule
#
#   Phase A: AGGRESSIVE FORGET — 15 steps
#     K=0 (no inner), no ALM, logit_margin, LR=2e-5
#     Pure outer descent on forget manifold
#
#   Phase B: BILEVEL + ALM — 45 steps
#     K=2, ALM on (eps=0.50), logit_margin
#     LR=5e-6, tightening the leash
#     Fresh LoRA on Phase A merged model
#
#   Total: ~60 steps = 30% of an epoch
#   Goal: break cold start then stabilize
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_muse_news_v3_progress.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json"

run_exp() {
    local name="$1"; shift
    local outdir="$SAVES/$name"

    if [[ -d "$outdir/checkpoint-0/evals" ]]; then
        log "[SKIP] $name — already has eval results"
        return
    fi
    # Clean previous partial run
    rm -rf "$outdir" 2>/dev/null

    log "[TRAIN] $name"
    python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/lora_implicit \
        data_split=News \
        task_name="$name" \
        retain_logs_path="$RETAIN_LOGS" \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.gradient_accumulation_steps=1 \
        trainer.args.gradient_checkpointing=true \
        "$@" 2>&1 | tee -a "$PROGRESS"

    log "[DONE] $name"
    for f in "$outdir"/checkpoint-*/evals/MUSE_SUMMARY.json "$outdir"/evals/MUSE_SUMMARY.json; do
        if [[ -f "$f" ]]; then
            log "[RESULT] $name:"
            cat "$f" | tee -a "$PROGRESS"
            echo "" | tee -a "$PROGRESS"
        fi
    done
}

log "═══════════════════════════════════════════════════════════════"
log "  MUSE News v3 — Graduated Schedule"
log "═══════════════════════════════════════════════════════════════"

# ─── Phase A: AGGRESSIVE FORGET — 15 steps ─────────────────────
# K=0 (no inner retain), no ALM, logit_margin
# LR=2e-5 (calibrated: L_ret should hit ~0.6 in 15 steps, recoverable)
# T=15 (fixed step count, not epoch-based)
run_exp muse_news_v3_phA \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.K=0 \
    trainer.method_args.T=15 \
    trainer.method_args.eta_theta=2e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=100.0 \
    trainer.method_args.lambda_init=0.0 \
    trainer.method_args.lambda_max=0.0 \
    trainer.method_args.rho=0.0 \
    trainer.method_args.use_implicit=false \
    trainer.method_args.adaptive_lr=false \
    trainer.method_args.checkpoint_every_epoch=false \
    trainer.method_args.eval_at_steps=[] \
    trainer.args.num_train_epochs=1

log "═══════════════════════════════════════════════════════════════"
log "  Phase A done — starting Phase B (bilevel + ALM)"
log "═══════════════════════════════════════════════════════════════"

# ─── Phase B: BILEVEL + ALM — 45 steps ─────────────────────────
# Fresh LoRA on Phase A merged model
# K=2 inner retain, ALM on (eps=0.50, lam_init=0.5, max=3.0)
# LR=5e-6 (4x drop — in delicate regime)
# T=45 (fixed steps)
PA_CKPT="$SAVES/muse_news_v3_phA"

run_exp muse_news_v3_phB \
    model.model_args.pretrained_model_name_or_path="$PA_CKPT" \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.K=2 \
    trainer.method_args.T=45 \
    trainer.method_args.eta_theta=5e-6 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.50 \
    trainer.method_args.lambda_init=0.5 \
    trainer.method_args.lambda_max=3.0 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.use_implicit=false \
    trainer.method_args.adaptive_lr=false \
    trainer.method_args.checkpoint_every_epoch=false \
    trainer.method_args.eval_at_steps=[] \
    trainer.args.num_train_epochs=1

log "═══════════════════════════════════════════════════════════════"
log "  v3 Done!"
log "═══════════════════════════════════════════════════════════════"
