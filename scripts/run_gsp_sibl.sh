#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   GSP-SIBL: Gradient Subspace Partitioned Bilevel — MUSE News
#   Bilevel with GSP-weighted sampling, logit_margin, no PerTA
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SAVES="saves/unlearn"
PROGRESS="$SAVES/_gsp_sibl_progress.log"
RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

run_exp() {
    local name="$1"; shift
    local outdir="$SAVES/$name"

    # Skip if already done
    if [[ -f "$outdir/evals/MUSE_SUMMARY.json" ]]; then
        log "[SKIP] $name — done"
        return
    fi

    log "[TRAIN] $name"
    python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/gsp_sibl \
        task_name="$name" \
        "$@" 2>&1 | tail -5

    log "[EVAL] $name"
    python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=News \
        task_name="$name" \
        model=Llama-2-7b-hf \
        model.model_args.pretrained_model_name_or_path="$outdir" \
        model.model_args.attn_implementation=sdpa \
        paths.output_dir="$outdir/evals" \
        retain_logs_path="$RETAIN_LOGS" 2>&1 | tail -3

    log "[DONE] $name"
}

log "═══════════════════════════════════════════════════════════════"
log "  GSP-SIBL: Bilevel with GSP-weighted sampling — MUSE News"
log "═══════════════════════════════════════════════════════════════"

# ─── Sweep A: GSP strength (α, β) ─────────────────────────
# A1: Baseline — no GSP weighting (α=0, β=0)
run_exp gsp_baseline \
    trainer.method_args.gsp_alpha=0.0 \
    trainer.method_args.gsp_beta=0.0 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=5e-5 \
    args.num_train_epochs=2

# A2: Moderate GSP (α=5, β=5) — needs high α because E_f has low spread
run_exp gsp_a5b5 \
    trainer.method_args.gsp_alpha=5.0 \
    trainer.method_args.gsp_beta=5.0 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=5e-5 \
    args.num_train_epochs=2

# A3: Strong GSP (α=20, β=20) — amplify the small differences
run_exp gsp_a20b20 \
    trainer.method_args.gsp_alpha=20.0 \
    trainer.method_args.gsp_beta=20.0 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=5e-5 \
    args.num_train_epochs=2

# ─── Sweep B: K and LR ────────────────────────────────────
# B1: K=5, aggressive outer LR
run_exp gsp_k5_lr4 \
    trainer.method_args.gsp_alpha=5.0 \
    trainer.method_args.gsp_beta=5.0 \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=1e-4 \
    args.num_train_epochs=2

# B2: K=5 + recovery
run_exp gsp_k5_rec2 \
    trainer.method_args.gsp_alpha=5.0 \
    trainer.method_args.gsp_beta=5.0 \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.recovery_epochs=2 \
    args.num_train_epochs=2

# B3: 3 epochs, high LR
run_exp gsp_3ep_lr4 \
    trainer.method_args.gsp_alpha=5.0 \
    trainer.method_args.gsp_beta=5.0 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=1e-4 \
    args.num_train_epochs=3

log "═══════════════════════════════════════════════════════════════"
log "  All GSP-SIBL experiments done!"
log "═══════════════════════════════════════════════════════════════"
