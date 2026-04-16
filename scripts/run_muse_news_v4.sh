#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   MUSE News v4: Graduated Schedule with NPO
#
#   Key insight: logit_margin doesn't target QA-measurable forgetting.
#   NPO directly reduces P(correct tokens) which is what forget_knowmem
#   measures. MUSE CE(forget)~0.36 at init — NOT cold-started like TOFU.
#   NPO will have strong gradient signal from the start.
#
#   Phase A: AGGRESSIVE NPO FORGET — 30 steps
#     K=0 (no inner), no ALM, NPO beta=2.0, LR=3e-5
#     From Exp8r: NPO beta=2.0 was optimal for FD-HVP
#     30 steps = ~15% of epoch, matching Exp8r's accidental sample count
#
#   Phase B: BILEVEL + ALM — 50 steps
#     K=2, ALM on, NPO beta=4.0 (high beta → saturates → free retain recovery)
#     Fresh LoRA on Phase A merged model
#     LR=3e-5 (proven outer_lr from Ze0)
#
#   Also running a variant with layer freezing:
#   Phase A_layered: same but freeze_outer_layers=[0,1,4,8,9] (high-cosine layers)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_muse_news_v4_progress.log"

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
log "  MUSE News v4 — NPO Graduated Schedule"
log "═══════════════════════════════════════════════════════════════"

# ─── Exp 1: Phase A — 30 steps NPO pure forget ──────────────────
run_exp muse_news_v4_phA \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.K=0 \
    trainer.method_args.T=30 \
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
log "  Phase A done — starting Phase B (bilevel + ALM, NPO beta=4)"
log "═══════════════════════════════════════════════════════════════"

# ─── Exp 2: Phase B — 50 steps bilevel from Phase A ─────────────
# NPO beta=4 (saturates fast → free retain recovery, proven in Ze0)
PA_CKPT="$SAVES/muse_news_v4_phA"

run_exp muse_news_v4_phB \
    model.model_args.pretrained_model_name_or_path="$PA_CKPT" \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=4.0 \
    trainer.method_args.K=2 \
    trainer.method_args.T=50 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.50 \
    trainer.method_args.lambda_init=0.5 \
    trainer.method_args.lambda_max=5.0 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.use_implicit=false \
    trainer.method_args.adaptive_lr=false \
    trainer.method_args.checkpoint_every_epoch=false \
    trainer.method_args.eval_at_steps=[] \
    trainer.args.num_train_epochs=1

# ─── Exp 3: Phase A with layer freezing — 30 steps ──────────────
# Same as Exp 1 but freeze layers 0,1,4,8,9 (high gradient cosine)
# during the outer (forget) step. Should reduce retain damage.
log "═══════════════════════════════════════════════════════════════"
log "  Exp 3: Phase A with layer freezing"
log "═══════════════════════════════════════════════════════════════"

run_exp muse_news_v4_phA_layered \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.K=0 \
    trainer.method_args.T=30 \
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
    'trainer.method_args.freeze_outer_layers=[0,1,4,8,9]' \
    trainer.args.num_train_epochs=1

# ─── Exp 4: Phase B from layered Phase A ────────────────────────
log "═══════════════════════════════════════════════════════════════"
log "  Exp 4: Phase B from layered Phase A"
log "═══════════════════════════════════════════════════════════════"

PA_LAYERED_CKPT="$SAVES/muse_news_v4_phA_layered"

run_exp muse_news_v4_phB_layered \
    model.model_args.pretrained_model_name_or_path="$PA_LAYERED_CKPT" \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=4.0 \
    trainer.method_args.K=2 \
    trainer.method_args.T=50 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.50 \
    trainer.method_args.lambda_init=0.5 \
    trainer.method_args.lambda_max=5.0 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.use_implicit=false \
    trainer.method_args.adaptive_lr=false \
    trainer.method_args.checkpoint_every_epoch=false \
    trainer.method_args.eval_at_steps=[] \
    trainer.args.num_train_epochs=1

log "═══════════════════════════════════════════════════════════════"
log "  v4 Done!"
log "═══════════════════════════════════════════════════════════════"
