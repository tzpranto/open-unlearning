#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   MUSE News v2: LoRA-Implicit — bilevel from the start
#   Llama-2-7b-hf, max_len=1024
#
#   Phase 1: GENTLE BILEVEL (PerTA proxy)
#     K=3 inner retain, logit_margin forget, LR=3e-5, 1 epoch
#     ALM OFF (lambda=0, max=0) — inner loop protects retain
#     Goal: lower forget without nuking retain
#
#   Phase 2: BILEVEL + ALM (recover retain)
#     K=3 inner retain, logit_margin forget, LR=3e-5, 3 epochs
#     ALM ON (eps=0.50, lambda_max=5.0) — actively recover retain
#     Fresh LoRA on P1 merged model, eval after each epoch
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_muse_news_v2_progress.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json"

run_exp() {
    local name="$1"; shift
    local outdir="$SAVES/$name"

    if [[ -d "$outdir/checkpoint-0/evals" ]]; then
        log "[SKIP] $name — already has eval results"
        return
    fi

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
    # Print eval results if they exist
    for f in "$outdir"/checkpoint-*/evals/MUSE_SUMMARY.json; do
        if [[ -f "$f" ]]; then
            log "[RESULT] $(basename $(dirname $(dirname $f))):"
            cat "$f" | tee -a "$PROGRESS"
            echo "" | tee -a "$PROGRESS"
        fi
    done
}

log "═══════════════════════════════════════════════════════════════"
log "  MUSE News v2 — Bilevel from the start"
log "═══════════════════════════════════════════════════════════════"

# ─── Phase 1: GENTLE BILEVEL — 1 epoch ─────────────────────────
# K=3 inner retain, no ALM, logit_margin forget
# LR=3e-5 (proven outer_lr from LoRA-BiAL Ze0)
# This acts as a PerTA proxy: lowers forget gently, retain protected by inner loop
run_exp muse_news_v2_p1 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=3e-5 \
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
log "  Phase 1 done — starting Phase 2 (bilevel + ALM)"
log "═══════════════════════════════════════════════════════════════"

# ─── Phase 2: BILEVEL + ALM — 3 epochs from P1 ────────────────
# Fresh LoRA on P1 merged model
# K=3 inner retain, ALM on (eps=0.50, lambda_max=5.0)
# Eval after each epoch via checkpoint_every_epoch
P1_CKPT="$SAVES/muse_news_v2_p1"

run_exp muse_news_v2_p2 \
    model.model_args.pretrained_model_name_or_path="$P1_CKPT" \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.50 \
    trainer.method_args.lambda_init=1.0 \
    trainer.method_args.lambda_max=5.0 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.use_implicit=false \
    trainer.method_args.adaptive_lr=false \
    trainer.method_args.checkpoint_every_epoch=true \
    trainer.method_args.eval_at_steps=[] \
    trainer.args.num_train_epochs=3

log "═══════════════════════════════════════════════════════════════"
log "  All done!"
log "═══════════════════════════════════════════════════════════════"
