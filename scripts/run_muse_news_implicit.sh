#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   MUSE News: LoRA-Implicit — logit_margin answer-masked
#   Llama-2-7b-hf, max_len=1024
#
#   Dataset: ~3273 forget chunks, ~1603 retain chunks (at 1024)
#   Dataloader pairs them → 204 steps/epoch (limited by smaller set)
#
#   Phase 1: PURE FORGET — logit_margin only, no retain, 1 epoch
#            K=0 (no inner), ALM disabled (lambda=0)
#            Breaks cold start aggressively
#
#   Phase 2: BILEVEL — logit_margin + retain recovery, 3 epochs
#            K=1, ALM on, eval after each epoch
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_muse_news_implicit_progress.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json"

run_exp() {
    local name="$1"; shift
    local outdir="$SAVES/$name"

    if [[ -d "$outdir/checkpoint-0/evals" ]]; then
        log "[SKIP] $name — done"
        return
    fi
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
        "$@" 2>&1 | tail -100

    log "[DONE] $name"
    for f in "$outdir"/checkpoint-*/evals/MUSE_SUMMARY.json; do
        if [[ -f "$f" ]]; then
            log "[RESULT] $(basename $(dirname $(dirname $f))):"
            cat "$f" | tee -a "$PROGRESS"
            echo "" | tee -a "$PROGRESS"
        fi
    done
}

log "═══════════════════════════════════════════════════════════════"
log "  MUSE News — LoRA-Implicit"
log "═══════════════════════════════════════════════════════════════"

# ─── Phase 1: PURE FORGET — 1 epoch ─────────────────────────────
# K=0 (no inner retain), ALM disabled (lambda=0, max=0, eps=100)
# Outer = pure logit_margin on forget data
# eta_theta=1e-5 (gentle — 7B model, 203 steps, 40M LoRA params)
run_exp muse_news_imp_p1 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.K=0 \
    trainer.method_args.eta_theta=1e-5 \
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
log "  Phase 1 done — starting Phase 2 (bilevel)"
log "═══════════════════════════════════════════════════════════════"

# ─── Phase 2: BILEVEL — 3 epochs from P1 ────────────────────────
# Fresh LoRA on P1 merged model
# K=1 inner retain, ALM on, logit_margin forget
# eta_theta=3e-5 (gentler), eval after each epoch
P1_CKPT="$SAVES/muse_news_imp_p1"

run_exp muse_news_imp_p2 \
    model.model_args.pretrained_model_name_or_path="$P1_CKPT" \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.K=1 \
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
log "  Done!"
log "═══════════════════════════════════════════════════════════════"
