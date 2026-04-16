#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   MUSE News v7: GA Bilevel — Gradient Ascent with Retain Protection
#
#   GA (negative CE) directly reduces P(correct tokens) on forget data.
#   GradDiff baseline achieved fk=0.330 (near gold) but rk=0.247.
#   Our bilevel framework should protect retain better via K inner steps.
#
#   GA is volatile — need careful LR and clipping.
#   ga_clip=1.0 limits gradient norm (prevents explosion).
#
#   Exp A: K=3, T=25, LR=1e-5, ga_clip=1.0 (conservative)
#   Exp B: K=3, T=50, LR=1e-5, ga_clip=1.0
#   Exp C: K=3, T=25, LR=3e-5, ga_clip=0.5 (tighter clip, higher LR)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

SAVES="saves/unlearn"
PROGRESS="$SAVES/_muse_news_v7_progress.log"

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
log "  MUSE News v7 — GA Bilevel"
log "═══════════════════════════════════════════════════════════════"

# ─── Exp A: Conservative GA ─────────────────────────────────────
run_exp muse_news_v7_ga_t25 \
    trainer.method_args.forget_loss_type=ga \
    trainer.method_args.ga_clip=1.0 \
    trainer.method_args.K=3 \
    trainer.method_args.T=25 \
    trainer.method_args.eta_theta=1e-5 \
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

# ─── Exp B: GA with more steps ──────────────────────────────────
run_exp muse_news_v7_ga_t50 \
    trainer.method_args.forget_loss_type=ga \
    trainer.method_args.ga_clip=1.0 \
    trainer.method_args.K=3 \
    trainer.method_args.T=50 \
    trainer.method_args.eta_theta=1e-5 \
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

# ─── Exp C: Tighter clip, higher LR ────────────────────────────
run_exp muse_news_v7_ga_clip05 \
    trainer.method_args.forget_loss_type=ga \
    trainer.method_args.ga_clip=0.5 \
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

log "═══════════════════════════════════════════════════════════════"
log "  v7 Done!"
log "═══════════════════════════════════════════════════════════════"
