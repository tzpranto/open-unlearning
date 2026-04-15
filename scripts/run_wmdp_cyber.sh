#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   WMDP Cyber: GSP-SIBL + LoRA-Implicit unlearning
#   Model: Zephyr-7b-beta
#   Eval: wmdp_cyber accuracy + MMLU
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SAVES="saves/unlearn"
PROGRESS="$SAVES/_wmdp_cyber_progress.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

run_exp() {
    local name="$1"; shift
    local config="$1"; shift
    local outdir="$SAVES/$name"

    if [[ -d "$outdir/evals" ]]; then
        log "[SKIP] $name — done"
        return
    fi

    log "[TRAIN] $name"
    python src/train.py --config-name=unlearn.yaml \
        experiment="$config" \
        data_split=cyber \
        task_name="$name" \
        "$@" 2>&1 | tail -5

    log "[DONE] $name"
}

log "═══════════════════════════════════════════════════════════════"
log "  WMDP Cyber Unlearning — Zephyr-7b-beta"
log "═══════════════════════════════════════════════════════════════"

# ─── Step 0: Compute GSP signatures ────────────────────────
GSP_PATH="$SAVES/gsp_interference_wmdp_cyber.pt"
if [[ ! -f "$GSP_PATH" ]]; then
    log "[GSP] Computing WMDP cyber signatures..."
    python scripts/compute_gsp.py \
        --benchmark wmdp --data_split cyber \
        --max_length 512 2>&1 | tail -10
    log "[GSP] Done."
else
    log "[GSP] Cache exists: $GSP_PATH"
fi

# ─── LoRA-Implicit baselines (no GSP) ──────────────────────
# LI1: Basic bilevel, logit_margin, 2 epochs
run_exp wmdp_cy_imp_lm unlearn/wmdp/lora_implicit \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.args.num_train_epochs=2

# LI2: K=5, higher LR
run_exp wmdp_cy_imp_k5 unlearn/wmdp/lora_implicit \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.args.num_train_epochs=2

# ─── GSP-SIBL experiments ──────────────────────────────────
# GS1: Baseline (no weighting)
run_exp wmdp_cy_gsp_base unlearn/wmdp/gsp_sibl \
    trainer.method_args.gsp_alpha=0.0 \
    trainer.method_args.gsp_beta=0.0 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.args.num_train_epochs=2

# GS2: Moderate GSP
run_exp wmdp_cy_gsp_a5b5 unlearn/wmdp/gsp_sibl \
    trainer.method_args.gsp_alpha=5.0 \
    trainer.method_args.gsp_beta=5.0 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.args.num_train_epochs=2

# GS3: Strong GSP + K=5
run_exp wmdp_cy_gsp_a20_k5 unlearn/wmdp/gsp_sibl \
    trainer.method_args.gsp_alpha=20.0 \
    trainer.method_args.gsp_beta=20.0 \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.args.num_train_epochs=2

# GS4: GSP + recovery
run_exp wmdp_cy_gsp_rec unlearn/wmdp/gsp_sibl \
    trainer.method_args.gsp_alpha=5.0 \
    trainer.method_args.gsp_beta=5.0 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.recovery_epochs=2 \
    trainer.args.num_train_epochs=2

log "═══════════════════════════════════════════════════════════════"
log "  All WMDP cyber experiments done!"
log "═══════════════════════════════════════════════════════════════"
