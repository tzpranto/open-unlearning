#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   MUSE News v6: Layer-Surgical NPO Bilevel
#
#   Key insight from 1000+ experiments: gradient bilevel is trapped
#   on a linear frontier due to 85% neuron overlap. PerTA breaks it
#   via weight surgery. Can we break it with layer-selective training?
#
#   Layer cosine diagnostic shows:
#     DANGER (cos>0.3): layers 0,1,4,8,9 — high entanglement
#     SAFEST (cos<0.13): layers 17-25 — lowest entanglement
#
#   Strategy: During outer (forget) step, FREEZE high-cosine layers.
#   This reduces effective neuron overlap for forget updates, potentially
#   changing the slope of the Pareto frontier.
#
#   Exp A: Freeze danger layers [0,1,4,8,9] during outer (forget) step
#   Exp B: Only update safe layers [14-25] during outer step (aggressive)
#   Exp C: Exp A + higher LR (5e-5) since fewer params updated
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_muse_news_v6_progress.log"

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
log "  MUSE News v6 — Layer-Surgical NPO Bilevel"
log "═══════════════════════════════════════════════════════════════"

# ─── Exp A: Freeze danger layers during outer step, T=80 ────────
# Inner loop (retain) = all layers. Outer (forget) = skip 0,1,4,8,9
run_exp muse_news_v6_freeze5 \
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
    'trainer.method_args.freeze_outer_layers=[0,1,4,8,9]' \
    trainer.args.num_train_epochs=1

# ─── Exp B: Only update safe layers [14-25] during outer step ───
# Freeze everything except 14-25 during forget
run_exp muse_news_v6_safe_only \
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
    'trainer.method_args.freeze_outer_layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,26,27,28,29,30,31]' \
    trainer.args.num_train_epochs=1

# ─── Exp C: Freeze danger + higher LR ──────────────────────────
run_exp muse_news_v6_freeze5_hiLR \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.K=3 \
    trainer.method_args.T=80 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.method_args.lambda_max=0.0 \
    trainer.method_args.use_implicit=false \
    trainer.method_args.adaptive_lr=true \
    trainer.method_args.checkpoint_every_epoch=false \
    trainer.method_args.eval_at_steps=[] \
    'trainer.method_args.freeze_outer_layers=[0,1,4,8,9]' \
    trainer.args.num_train_epochs=1

log "═══════════════════════════════════════════════════════════════"
log "  v6 Done!"
log "═══════════════════════════════════════════════════════════════"
