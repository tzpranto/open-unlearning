#!/bin/bash
# Z-pure: Pure LoRA-BiAL from target model (NO PerTA)
#
# Tests whether the Ze0 bilevel recipe works without PerTA initialization.
# ref = target model with LoRA disabled (zero overhead).
#
# Tracks:
#   1. Core test: Ze0 recipe variants from target
#   2. Rank ablation: r=4,8,16,32
#   3. Baselines: LoRA+NPO only, LoRA+GA only (proves bilevel matters)
#
# Restart-resistant: every step checks if output already exists.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="/datadrive/forked/open-unlearning"
SAVES="$BASE/saves/unlearn"
EVAL_SAVES="$BASE/saves/eval"
PYTHON="${CONDA_PREFIX:-/datadrive/conda/envs/unlearning}/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
PROGRESS="$BASE/saves/unlearn/_zpure_progress.log"

cd "$BASE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

is_done() {
    local name=$1
    find "$SAVES/$name" -name "MUSE_EVAL.json" -print -quit 2>/dev/null | grep -q .
}

has_model() {
    local name=$1
    compgen -G "$SAVES/$name/model*.safetensors" > /dev/null 2>&1 || \
    compgen -G "$SAVES/$name/model.safetensors" > /dev/null 2>&1
}

report_metrics() {
    local name=$1
    local json
    json=$(find "$SAVES/$name" -name "MUSE_EVAL.json" -print -quit 2>/dev/null)
    if [ -z "$json" ]; then
        json=$(find "$EVAL_SAVES" -path "*${name}*" -name "MUSE_EVAL.json" -print -quit 2>/dev/null)
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
" 2>/dev/null || echo "  (metrics parse failed)"
    fi
}

run_eval() {
    local name=$1
    log "[EVAL] $name"
    "$PYTHON" src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${name} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${name} \
        model.model_args.attn_implementation=sdpa \
        paths.output_dir=saves/unlearn/${name}/evals \
        retain_logs_path=${RETAIN_LOGS} \
        2>&1 | tee /tmp/eval_zpure_${name}.log
}

run_zpure() {
    local name=$1
    shift
    local overrides=("$@")

    if is_done "$name"; then
        log "[SKIP] $name — eval exists"
        report_metrics "$name"
        return 0
    fi

    if [ -d "$SAVES/$name" ] && ! has_model "$name"; then
        log "[CLEANUP] $name — removing incomplete directory"
        rm -rf "$SAVES/$name"
    fi

    if has_model "$name"; then
        log "[SKIP-TRAIN] $name — model exists, running eval only"
    else
        log "[TRAIN] $name"
        "$PYTHON" src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/zpure \
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
            2>&1 | tee /tmp/train_zpure_${name}.log
    fi

    run_eval "$name"
    report_metrics "$name"
    log "[DONE] $name"
}

# ═══════════════════════════════════════════════════════════════════
# Common overrides — shared across all Z-pure runs
# ═══════════════════════════════════════════════════════════════════
COMMON=(
    trainer.args.per_device_train_batch_size=2
    trainer.args.gradient_accumulation_steps=1
    trainer.method_args.lora_alpha=32
    trainer.method_args.forget_loss_type=npo
    trainer.method_args.epsilon=0.70
    trainer.method_args.rho=0.1
    trainer.method_args.lambda_init=1.0
    trainer.method_args.eta_theta=3e-5
    trainer.method_args.eta_in=2e-4
    trainer.method_args.T=-1
    trainer.method_args.lr_schedule=constant
)

log "═══════════════════════════════════════════════════════════════"
log "  Z-pure: Pure LoRA-BiAL (NO PerTA) — MUSE ${DATA_SPLIT}"
log "  ref = target with LoRA disabled (zero overhead)"
log "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════
# TRACK 1: Core test — Ze0 recipe from target model
# ═══════════════════════════════════════════════════════════════════
log "=== Track 1: Core tests ==="

# Z-pure-0: Ze0 recipe, 1 epoch, K=3, beta=4
run_zpure "zpure_0" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=3 \
    trainer.method_args.npo_beta=4.0

# Z-pure-2: beta=2.0 (slower NPO saturation — target has stronger forget signal)
run_zpure "zpure_2" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=3 \
    trainer.method_args.npo_beta=2.0

# Z-pure-3: K=5 (more inner correction — target rk starts at 0.544)
run_zpure "zpure_3" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=5 \
    trainer.method_args.npo_beta=4.0

# Z-pure-1: 2 epochs (more training since starting from target, harder)
run_zpure "zpure_1" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=2 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=3 \
    trainer.method_args.npo_beta=4.0

# ═══════════════════════════════════════════════════════════════════
# TRACK 3: Rank ablation (r=4,8,32 — r=16 is zpure_0)
# ═══════════════════════════════════════════════════════════════════
log "=== Track 3: Rank ablation ==="

# Z-pure-6: r=4 (very constrained, ~13.6M params)
run_zpure "zpure_6_r4" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=4 \
    trainer.method_args.lora_alpha=8 \
    trainer.method_args.K=3 \
    trainer.method_args.npo_beta=4.0

# Z-pure-7: r=8 (intermediate)
run_zpure "zpure_7_r8" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=8 \
    trainer.method_args.lora_alpha=16 \
    trainer.method_args.K=3 \
    trainer.method_args.npo_beta=4.0

# Z-pure-8: r=32 (more capacity — helps or hurts?)
run_zpure "zpure_8_r32" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=32 \
    trainer.method_args.lora_alpha=64 \
    trainer.method_args.K=3 \
    trainer.method_args.npo_beta=4.0

# ═══════════════════════════════════════════════════════════════════
# TRACK 1c: Logit margin bilevel (flatten toward uniform — never saturates)
# min(max_logit - mean_logit) on forget. Smooth, bounded, no reference model.
# ═══════════════════════════════════════════════════════════════════
log "=== Track 1c: Logit margin bilevel ==="

# Z-pure-lm0: logit margin bilevel, K=3, 1 epoch (core test)
run_zpure "zpure_lm0" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=3 \
    trainer.method_args.forget_loss_type=logit_margin

# Z-pure-lm1: logit margin, K=5
run_zpure "zpure_lm1_k5" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=5 \
    trainer.method_args.forget_loss_type=logit_margin

# Z-pure-lm2: logit margin, 2 epochs
run_zpure "zpure_lm2_2ep" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=2 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=3 \
    trainer.method_args.forget_loss_type=logit_margin

# Z-pure-lm3: logit margin, tighter epsilon
run_zpure "zpure_lm3_eps05" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=3 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.epsilon=0.50

# ═══════════════════════════════════════════════════════════════════
# TRACK 1b: GA bilevel (gradient ascent — no saturation unlike NPO)
# Inner=retain CE, Outer=GA+ALM. GA never saturates: -CE is unbounded.
# ═══════════════════════════════════════════════════════════════════
log "=== Track 1b: GA bilevel (no saturation) ==="

# Z-pure-ga0: GA bilevel, K=3, 1 epoch (core GA test)
run_zpure "zpure_ga0" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=3 \
    trainer.method_args.forget_loss_type=ga \
    trainer.method_args.ga_clip=1.0

# Z-pure-ga1: GA bilevel, K=5 (more inner correction)
run_zpure "zpure_ga1_k5" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=5 \
    trainer.method_args.forget_loss_type=ga \
    trainer.method_args.ga_clip=1.0

# Z-pure-ga2: GA bilevel, K=3, 2 epochs
run_zpure "zpure_ga2_2ep" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=2 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=3 \
    trainer.method_args.forget_loss_type=ga \
    trainer.method_args.ga_clip=1.0

# Z-pure-ga3: GA bilevel, K=3, tighter epsilon (more retain pressure)
run_zpure "zpure_ga3_eps05" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=3 \
    trainer.method_args.forget_loss_type=ga \
    trainer.method_args.ga_clip=1.0 \
    trainer.method_args.epsilon=0.50

# ═══════════════════════════════════════════════════════════════════
# TRACK 4: Baselines (proves bilevel structure matters, not just LoRA)
# ═══════════════════════════════════════════════════════════════════
log "=== Track 4: Baselines ==="

# Z-pure-9: K=0, NPO only — same outer NPO+ALM but no inner retain CE
run_zpure "zpure_9_npo_only" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=0 \
    trainer.method_args.npo_beta=4.0

# Z-pure-10: K=0, GA — gradient ascent + ALM, no bilevel (no inner loop)
run_zpure "zpure_10_ga_only" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=0 \
    trainer.method_args.forget_loss_type=ga

# ═══════════════════════════════════════════════════════════════════
# TRACK 2: FD-HVP Implicit correction (novelty claim #1)
# Applied to logit_margin bilevel — smooth, never saturates, ideal for implicit
# ═══════════════════════════════════════════════════════════════════
log "=== Track 2: FD-HVP Implicit ==="

# Z-pure-4: logit_margin bilevel + implicit (5 Neumann steps)
run_zpure "zpure_4_implicit" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=3 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.use_implicit=true \
    trainer.method_args.neumann_steps=5 \
    trainer.method_args.fd_hvp_eps=0.01 \
    trainer.method_args.neumann_mu=0.01

# Z-pure-5: implicit with 2 Neumann steps (cheaper, may suffice)
run_zpure "zpure_5_implicit_n2" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=3 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.use_implicit=true \
    trainer.method_args.neumann_steps=2 \
    trainer.method_args.fd_hvp_eps=0.01 \
    trainer.method_args.neumann_mu=0.01

# Z-pure-4c: implicit with CPU offload (if GPU OOM on zpure_4)
run_zpure "zpure_4c_implicit_cpu" \
    "${COMMON[@]}" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.K=3 \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.use_implicit=true \
    trainer.method_args.neumann_steps=5 \
    trainer.method_args.fd_hvp_eps=0.01 \
    trainer.method_args.neumann_mu=0.01 \
    trainer.method_args.implicit_offload_cpu=true

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
log ""
log "═══════════════════════════════════════════════════════════════"
log "  Z-pure Results Summary"
log "═══════════════════════════════════════════════════════════════"
log "References:"
log "  Ze0 (PerTA+LoRA, r=16, T=25): fk=0.289, rk=0.449, delta=+0.120"
log "  Target model:                  fk=0.612, rk=0.544, delta=-0.162"
log "  CE frontier:                   rk = 0.55*fk + 0.17"
log ""

log "--- Track 1: NPO bilevel (K=3, r=16) ---"
for name in zpure_0 zpure_2 zpure_3 zpure_1; do
    log "$name:"; report_metrics "$name"
done

log "--- Track 1c: Logit margin bilevel (flatten toward uniform) ---"
for name in zpure_lm0 zpure_lm1_k5 zpure_lm2_2ep zpure_lm3_eps05; do
    log "$name:"; report_metrics "$name"
done

log "--- Track 1b: GA bilevel ---"
for name in zpure_ga0 zpure_ga1_k5 zpure_ga2_2ep zpure_ga3_eps05; do
    log "$name:"; report_metrics "$name"
done

log "--- Track 3: Rank ablation (NPO, K=3, beta=4) ---"
for name in zpure_6_r4 zpure_7_r8 zpure_0 zpure_8_r32; do
    log "$name:"; report_metrics "$name"
done

log "--- Track 2: Logit margin + Implicit (FD-HVP Neumann) ---"
for name in zpure_4_implicit zpure_5_implicit_n2 zpure_4c_implicit_cpu; do
    log "$name:"; report_metrics "$name"
done

log "--- Track 4: Baselines (K=0, r=16) ---"
for name in zpure_9_npo_only zpure_10_ga_only; do
    log "$name:"; report_metrics "$name"
done

log ""
log "Success criterion: delta > 0 (above CE frontier)"
log "═══════════════════════════════════════════════════════════════"
log "Done."
