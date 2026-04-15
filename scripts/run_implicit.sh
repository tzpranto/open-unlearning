#!/bin/bash
# LoRA-Implicit: Clean bilevel from target model
#
# Phase 1: Run both modes WITHOUT implicit (fast, ~3s/step)
# Phase 2: If results need retain recovery, add post-hoc inner steps
#
# Two bilevel modes:
#   forget_outer: inner=retain CE, outer=forget+ALM
#   forget_inner: inner=forget,    outer=retain+ALM
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="/datadrive/forked/open-unlearning"
SAVES="$BASE/saves/unlearn"
EVAL_SAVES="$BASE/saves/eval"
PYTHON="${CONDA_PREFIX:-/datadrive/conda/envs/unlearning}/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
PROGRESS="$BASE/saves/unlearn/_implicit_progress.log"

cd "$BASE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

is_done() {
    find "$SAVES/$1" -name "MUSE_EVAL.json" -print -quit 2>/dev/null | grep -q .
}

has_model() {
    compgen -G "$SAVES/$1/model*.safetensors" > /dev/null 2>&1 || \
    compgen -G "$SAVES/$1/model.safetensors" > /dev/null 2>&1
}

report_metrics() {
    local json
    json=$(find "$SAVES/$1" -name "MUSE_EVAL.json" -print -quit 2>/dev/null)
    if [ -z "$json" ]; then
        json=$(find "$EVAL_SAVES" -path "*${1}*" -name "MUSE_EVAL.json" -print -quit 2>/dev/null)
    fi
    if [ -n "$json" ] && [ -f "$json" ]; then
        "$PYTHON" -c "
import json
with open('$json') as f:
    d = json.load(f)
fk = d.get('forget_knowmem_ROUGE', {}).get('agg_value', -1)
rk = d.get('retain_knowmem_ROUGE', {}).get('agg_value', -1)
delta = rk - (0.55*fk + 0.17)
print(f'  fk={fk:.4f}  rk={rk:.4f}  delta={delta:+.4f}')
" 2>/dev/null || echo "  (parse failed)"
    fi
}

run_eval() {
    log "[EVAL] $1"
    "$PYTHON" src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${1} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${1} \
        model.model_args.attn_implementation=sdpa \
        paths.output_dir=saves/unlearn/${1}/evals \
        retain_logs_path=${RETAIN_LOGS} \
        2>&1 | tee /tmp/eval_imp_${1}.log
}

run_imp() {
    local name=$1; shift
    local overrides=("$@")

    if is_done "$name"; then
        log "[SKIP] $name — done"
        report_metrics "$name"
        return 0
    fi

    if [ -d "$SAVES/$name" ] && ! has_model "$name"; then
        rm -rf "$SAVES/$name"
    fi

    if has_model "$name"; then
        log "[SKIP-TRAIN] $name — model exists"
    else
        log "[TRAIN] $name"
        "$PYTHON" src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/lora_implicit \
            model=${MODEL} \
            data_split=${DATA_SPLIT} \
            task_name=${name} \
            retain_logs_path=${RETAIN_LOGS} \
            model.model_args.attn_implementation=sdpa \
            trainer.args.eval_strategy=no \
            trainer.args.do_eval=false \
            trainer.args.eval_on_start=false \
            trainer.args.gradient_checkpointing=true \
            "${overrides[@]}" \
            2>&1 | tee /tmp/train_imp_${name}.log
    fi

    run_eval "$name"
    report_metrics "$name"
    log "[DONE] $name"
}

# Common: no implicit, adaptive LR on
COMMON=(
    trainer.args.per_device_train_batch_size=2
    trainer.args.gradient_accumulation_steps=1
    trainer.method_args.lora_r=16
    trainer.method_args.lora_alpha=32
    trainer.method_args.epsilon=0.70
    trainer.method_args.rho=0.1
    trainer.method_args.lambda_init=1.0
    trainer.method_args.T=-1
    trainer.method_args.use_implicit=false
    trainer.method_args.adaptive_lr=true
    trainer.method_args.lr_schedule=constant
)

log "═══════════════════════════════════════════════════════════════"
log "  LoRA Bilevel — MUSE ${DATA_SPLIT} — NO implicit (Phase 1)"
log "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════
# A: forget_outer (inner=retain, outer=forget+ALM)
# ═══════════════════════════════════════════════════════════════
log "=== A: forget_outer ==="

# A0: NPO beta=4, K=3
run_imp "imp_fo_npo" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.bilevel_mode=forget_outer \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=4.0

# A1: NPO beta=2
run_imp "imp_fo_npo_b2" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.bilevel_mode=forget_outer \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0

# A2: logit_margin
run_imp "imp_fo_lm" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.bilevel_mode=forget_outer \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=logit_margin

# A3: logit_margin K=5
run_imp "imp_fo_lm_k5" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.bilevel_mode=forget_outer \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=logit_margin

# A4: logit_margin 2 epochs
run_imp "imp_fo_lm_2ep" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=2 \
    trainer.method_args.bilevel_mode=forget_outer \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=logit_margin

# ═══════════════════════════════════════════════════════════════
# B: forget_inner (inner=forget, outer=retain+ALM)
# ═══════════════════════════════════════════════════════════════
log "=== B: forget_inner ==="

# (NPO dropped — saturates from cold start, model=ref)

# B2: logit_margin
run_imp "imp_fi_lm" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.bilevel_mode=forget_inner \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=logit_margin

# B3: logit_margin K=5
run_imp "imp_fi_lm_k5" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.bilevel_mode=forget_inner \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=logit_margin

# B4: logit_margin 2 epochs
run_imp "imp_fi_lm_2ep" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=2 \
    trainer.method_args.bilevel_mode=forget_inner \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=logit_margin

# ═══════════════════════════════════════════════════════════════
# Phase 1 SUMMARY
# ═══════════════════════════════════════════════════════════════
log ""
log "═══════════════════════════════════════════════════════════════"
log "  Phase 1 Results — CE frontier: rk = 0.55*fk + 0.17"
log "  Target: fk=0.612, rk=0.544, delta=-0.162"
log "═══════════════════════════════════════════════════════════════"

log "--- A: forget_outer ---"
for n in imp_fo_npo imp_fo_npo_b2 imp_fo_lm imp_fo_lm_k5 imp_fo_lm_2ep; do
    log "$n:"; report_metrics "$n"
done

log "--- B: forget_inner ---"
for n in imp_fi_npo imp_fi_npo_b2 imp_fi_lm imp_fi_lm_k5 imp_fi_lm_2ep; do
    log "$n:"; report_metrics "$n"
done

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Aggressive — higher LR, more K, CG implicit, recovery
# Key fix: NO adaptive LR (it killed outer LR in Phase 1)
# ═══════════════════════════════════════════════════════════════
log ""
log "═══════════════════════════════════════════════════════════════"
log "  Phase 2: AGGRESSIVE — high LR, K=5-10, CG implicit, recovery"
log "═══════════════════════════════════════════════════════════════"

AGG=(
    trainer.args.per_device_train_batch_size=2
    trainer.args.gradient_accumulation_steps=1
    trainer.method_args.lora_r=16
    trainer.method_args.lora_alpha=32
    trainer.method_args.epsilon=0.70
    trainer.method_args.rho=0.1
    trainer.method_args.lambda_init=1.0
    trainer.method_args.T=-1
    trainer.method_args.adaptive_lr=false
    trainer.method_args.lr_schedule=constant
)

# C0: Aggressive baseline — higher outer LR, K=5, 3ep, no implicit
run_imp "imp_agg_lm_3ep" \
    "${AGG[@]}" \
    trainer.args.num_train_epochs=3 \
    trainer.method_args.bilevel_mode=forget_outer \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.use_implicit=false

# C1: Same + CG implicit (5 iters, mu=1.0, warmup=30)
run_imp "imp_agg_lm_3ep_cg" \
    "${AGG[@]}" \
    trainer.args.num_train_epochs=3 \
    trainer.method_args.bilevel_mode=forget_outer \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.use_implicit=true \
    trainer.method_args.neumann_steps=5 \
    trainer.method_args.neumann_mu=1.0 \
    trainer.method_args.implicit_warmup_steps=30

# C2: K=10 (heavy inner retain), 3ep, no implicit
run_imp "imp_agg_lm_k10_3ep" \
    "${AGG[@]}" \
    trainer.args.num_train_epochs=3 \
    trainer.method_args.bilevel_mode=forget_outer \
    trainer.method_args.K=10 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.use_implicit=false

# C3: K=10 + CG implicit
run_imp "imp_agg_lm_k10_3ep_cg" \
    "${AGG[@]}" \
    trainer.args.num_train_epochs=3 \
    trainer.method_args.bilevel_mode=forget_outer \
    trainer.method_args.K=10 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.use_implicit=true \
    trainer.method_args.neumann_steps=5 \
    trainer.method_args.neumann_mu=1.0 \
    trainer.method_args.implicit_warmup_steps=30

# C4: 3ep aggressive + 2ep retain recovery (the full pipeline)
run_imp "imp_agg_lm_3ep_rec2" \
    "${AGG[@]}" \
    trainer.args.num_train_epochs=3 \
    trainer.method_args.bilevel_mode=forget_outer \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.use_implicit=false \
    trainer.method_args.recovery_epochs=2 \
    trainer.method_args.recovery_lr=1e-5

# C5: K=10 + CG implicit + 2ep recovery
run_imp "imp_agg_lm_k10_cg_rec2" \
    "${AGG[@]}" \
    trainer.args.num_train_epochs=3 \
    trainer.method_args.bilevel_mode=forget_outer \
    trainer.method_args.K=10 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.use_implicit=true \
    trainer.method_args.neumann_steps=5 \
    trainer.method_args.neumann_mu=1.0 \
    trainer.method_args.implicit_warmup_steps=30 \
    trainer.method_args.recovery_epochs=2 \
    trainer.method_args.recovery_lr=1e-5

# C6: Even more aggressive: outer LR=1e-4, K=5, 3ep
run_imp "imp_agg_lm_lr4_3ep" \
    "${AGG[@]}" \
    trainer.args.num_train_epochs=3 \
    trainer.method_args.bilevel_mode=forget_outer \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.use_implicit=false

# ═══════════════════════════════════════════════════════════════
# FULL SUMMARY
# ═══════════════════════════════════════════════════════════════
log ""
log "═══════════════════════════════════════════════════════════════"
log "  Full Results — CE frontier: rk = 0.55*fk + 0.17"
log "═══════════════════════════════════════════════════════════════"

log "--- A: forget_outer ---"
for n in imp_fo_lm imp_fo_lm_k5 imp_fo_lm_2ep; do
    log "$n:"; report_metrics "$n"
done

log "--- B: forget_inner ---"
for n in imp_fi_lm imp_fi_lm_k5 imp_fi_lm_2ep; do
    log "$n:"; report_metrics "$n"
done

log "--- C: Phase 2 (aggressive) ---"
for n in imp_agg_lm_3ep imp_agg_lm_3ep_cg imp_agg_lm_k10_3ep imp_agg_lm_k10_3ep_cg imp_agg_lm_3ep_rec2 imp_agg_lm_k10_cg_rec2 imp_agg_lm_lr4_3ep; do
    log "$n:"; report_metrics "$n"
done

log "Done!"
