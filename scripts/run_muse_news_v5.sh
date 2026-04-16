#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   MUSE News v5: NPO Bilevel — Exp8r-inspired sample-efficient
#
#   Key insight: NPO pure-forget (K=0) with LoRA is broken —
#   the LoRA has no anchor to define a forgetting direction.
#   Exp8r achieved fk=0.371 with K=3 bilevel NPO beta=2 and
#   accidentally only ~10 samples. NPO saturates fast (5-8 steps),
#   then remaining steps = free retain recovery.
#
#   We test 3 configs at different step budgets:
#   Exp A: T=25 steps (matching Ze0's sweet spot)
#   Exp B: T=50 steps (25% of epoch)
#   Exp C: T=80 steps (~40% of epoch)
#
#   All use: K=3, NPO beta=2, LR=3e-5, adaptive_lr, no ALM cap
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_muse_news_v5_progress.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json"

run_exp() {
    local name="$1"; shift
    local outdir="$SAVES/$name"

    if [[ -d "$outdir/checkpoint-0/evals" ]]; then
        log "[SKIP] $name — already has eval results"
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
log "  MUSE News v5 — NPO Bilevel (Exp8r-style, sample-efficient)"
log "═══════════════════════════════════════════════════════════════"

# ─── Exp A: T=25 steps ──────────────────────────────────────────
run_exp muse_news_v5_t25 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.K=3 \
    trainer.method_args.T=25 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.method_args.lambda_max=0.0 \
    trainer.method_args.use_implicit=false \
    trainer.method_args.adaptive_lr=true \
    trainer.method_args.checkpoint_every_epoch=false \
    trainer.method_args.eval_at_steps=[] \
    trainer.args.num_train_epochs=1

# ─── Exp B: T=50 steps ──────────────────────────────────────────
run_exp muse_news_v5_t50 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.K=3 \
    trainer.method_args.T=50 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.method_args.lambda_max=0.0 \
    trainer.method_args.use_implicit=false \
    trainer.method_args.adaptive_lr=true \
    trainer.method_args.checkpoint_every_epoch=false \
    trainer.method_args.eval_at_steps=[] \
    trainer.args.num_train_epochs=1

# ─── Exp C: T=80 steps ──────────────────────────────────────────
run_exp muse_news_v5_t80 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.K=3 \
    trainer.method_args.T=80 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.method_args.lambda_max=0.0 \
    trainer.method_args.use_implicit=false \
    trainer.method_args.adaptive_lr=true \
    trainer.method_args.checkpoint_every_epoch=false \
    trainer.method_args.eval_at_steps=[] \
    trainer.args.num_train_epochs=1

log "═══════════════════════════════════════════════════════════════"
log "  v5 Done!"
log "═══════════════════════════════════════════════════════════════"
