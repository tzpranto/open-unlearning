#!/bin/bash
# Zc-series: LoRA-BiAL Adam — NPO saturation exploitation
#
# Key insight from Z/Za/Zb: NPO saturation is a FEATURE.
# Fast saturation → outer loop stops → inner loop gets free retain recovery.
# Zb5 (NPO β=2.0 K=3) = δ=+0.089 (best). Now push further:
#
# Axis 1: K < 3 (even less forget leakage per outer step)
# Axis 2: β > 2 (faster saturation → more free inner steps)
# Axis 3: T > 25 (more total inner steps after saturation)
# Axis 4: Inner LR tuning (balance inner speed vs stability)
#
# Zc0: NPO β=2.0 K=2 (fewer inner steps)
# Zc1: NPO β=2.0 K=1 (minimum inner steps)
# Zc2: NPO β=4.0 K=3 (faster saturation + K=3 sweet spot)
# Zc3: NPO β=8.0 K=3 (very fast saturation)
# Zc4: NPO β=2.0 K=3 T=50 (extend Zb5 winner with more steps)
# Zc5: NPO β=4.0 K=3 T=50 (fast saturation + long training)
# Zc6: NPO β=2.0 K=3 eta_in=5e-4 (faster inner retain recovery)
# Zc7: NPO β=2.0 K=3 r4 (smaller LoRA — Zb6 showed r4 is good)
#
# Restart-resistant.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="/datadrive/forked/open-unlearning"
SAVES="$BASE/saves/unlearn"
PYTHON="${CONDA_PREFIX:-/datadrive/conda/envs/unlearning}/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
FISHER_CACHE="saves/unlearn/_perta_fisher_cache_News_n64.pt"
PROGRESS="$BASE/saves/unlearn/_Zc_series_progress.log"

cd "$BASE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

is_done() {
    local name=$1
    [ -f "$SAVES/$name/evals/MUSE_EVAL.json" ]
}

has_model() {
    local name=$1
    compgen -G "$SAVES/$name/model*.safetensors" > /dev/null 2>&1 || \
    compgen -G "$SAVES/$name/model.safetensors" > /dev/null 2>&1
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
        2>&1 | tee /tmp/eval_${name}.log
}

report_metrics() {
    local name=$1
    local json="$SAVES/$name/evals/MUSE_EVAL.json"
    if [ -f "$json" ]; then
        "$PYTHON" -c "
import json
with open('$json') as f:
    d = json.load(f)
fk = d.get('forget_knowmem_ROUGE', {}).get('agg_value', d.get('forget_knowmem_ROUGE', {}).get('mean', -1))
vm = d.get('forget_verbmem_ROUGE', {}).get('agg_value', d.get('forget_verbmem_ROUGE', {}).get('mean', -1))
rk = d.get('retain_knowmem_ROUGE', {}).get('agg_value', d.get('retain_knowmem_ROUGE', {}).get('mean', -1))
delta = rk - (0.55*fk + 0.17)
print(f'  fk={fk:.4f}  vm={vm:.4f}  rk={rk:.4f}  delta={delta:+.4f}')
" 2>/dev/null || echo "  (metrics parse failed)"
    fi
}

run_experiment() {
    local name=$1
    shift
    local overrides=("$@")

    if is_done "$name"; then
        log "[SKIP] $name — eval already exists"
        report_metrics "$name"
        return 0
    fi

    # Clean up incomplete runs
    if [ -d "$SAVES/$name" ] && ! has_model "$name"; then
        log "[CLEANUP] $name — removing incomplete directory"
        rm -rf "$SAVES/$name"
    fi

    if has_model "$name"; then
        log "[SKIP-TRAIN] $name — model exists, running eval only"
    else
        log "[TRAIN] $name"
        "$PYTHON" src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/lora_bial \
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
            2>&1 | tee /tmp/train_${name}.log
    fi

    run_eval "$name"
    report_metrics "$name"
    log "[DONE] $name"
}

# Common overrides
COMMON=(
    trainer.method_args.fisher_cache_path=${FISHER_CACHE}
    trainer.args.per_device_train_batch_size=2
    trainer.args.gradient_accumulation_steps=1
    trainer.method_args.perta_lambda=3.5
    trainer.method_args.lora_r=16
    trainer.method_args.lora_alpha=32
    trainer.method_args.forget_loss_type=npo
    trainer.method_args.epsilon=0.70
    trainer.method_args.rho=0.1
    trainer.method_args.lambda_init=1.0
)

log "=== Zc-series: NPO saturation exploitation ==="

# ═══════════════════════════════════════════════════════════════════
# Zc0: NPO β=2.0 K=2 (fewer inner steps)
# Zb5 showed K=3 > K=5. Is K=2 even better?
# Less forget leakage per outer step → more efficient correction
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zc0_npo_K2" \
    "${COMMON[@]}" \
    trainer.method_args.npo_beta=2.0 \
    trainer.args.num_train_epochs=25 \
    trainer.method_args.T=25 \
    trainer.method_args.K=2 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4

# ═══════════════════════════════════════════════════════════════════
# Zc1: NPO β=2.0 K=1 (minimum inner steps)
# Absolute minimum inner loop — 1 retain CE step per outer step
# Maximum correction frequency but minimum retain progress per step
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zc1_npo_K1" \
    "${COMMON[@]}" \
    trainer.method_args.npo_beta=2.0 \
    trainer.args.num_train_epochs=25 \
    trainer.method_args.T=25 \
    trainer.method_args.K=1 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4

# ═══════════════════════════════════════════════════════════════════
# Zc2: NPO β=4.0 K=3 (faster saturation)
# Higher β = steeper sigmoid = NPO saturates faster
# More free inner steps after saturation → better retain recovery
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zc2_npo_beta4_K3" \
    "${COMMON[@]}" \
    trainer.method_args.npo_beta=4.0 \
    trainer.args.num_train_epochs=25 \
    trainer.method_args.T=25 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4

# ═══════════════════════════════════════════════════════════════════
# Zc3: NPO β=8.0 K=3 (very fast saturation)
# Extreme: NPO might saturate in 2-3 outer steps
# Nearly all 25 steps are free inner retain recovery
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zc3_npo_beta8_K3" \
    "${COMMON[@]}" \
    trainer.method_args.npo_beta=8.0 \
    trainer.args.num_train_epochs=25 \
    trainer.method_args.T=25 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4

# ═══════════════════════════════════════════════════════════════════
# Zc4: NPO β=2.0 K=3 T=50 (extend Zb5 winner)
# More outer steps = more inner steps after NPO saturates
# If NPO saturates at step 10, T=50 gives 40*3=120 free inner steps
# vs T=25 giving 15*3=45 free inner steps
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zc4_npo_K3_T50" \
    "${COMMON[@]}" \
    trainer.method_args.npo_beta=2.0 \
    trainer.args.num_train_epochs=50 \
    trainer.method_args.T=50 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4

# ═══════════════════════════════════════════════════════════════════
# Zc5: NPO β=4.0 K=3 T=50 (fast saturation + long training)
# Combines fast saturation (more free steps) with more total steps
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zc5_npo_beta4_K3_T50" \
    "${COMMON[@]}" \
    trainer.method_args.npo_beta=4.0 \
    trainer.args.num_train_epochs=50 \
    trainer.method_args.T=50 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4

# ═══════════════════════════════════════════════════════════════════
# Zc6: NPO β=2.0 K=3 + faster inner LR
# Higher inner LR = more retain progress per step
# With K=3, each inner step matters more → faster LR may help
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zc6_npo_K3_fast_inner" \
    "${COMMON[@]}" \
    trainer.method_args.npo_beta=2.0 \
    trainer.args.num_train_epochs=25 \
    trainer.method_args.T=25 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=5e-4

# ═══════════════════════════════════════════════════════════════════
# Zc7: NPO β=2.0 K=3 r4 (smaller LoRA)
# r4 constrains updates more → less forget leakage
# Zb6 (KL r4) got rk=0.448 — test with NPO K=3
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zc7_npo_K3_r4" \
    "${COMMON[@]}" \
    trainer.method_args.npo_beta=2.0 \
    trainer.args.num_train_epochs=25 \
    trainer.method_args.T=25 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.lora_r=4 \
    trainer.method_args.lora_alpha=8

log "=== Zc-series complete ==="

log "=== Zc-series results ==="
for name in Zc0_npo_K2 Zc1_npo_K1 Zc2_npo_beta4_K3 Zc3_npo_beta8_K3 \
            Zc4_npo_K3_T50 Zc5_npo_beta4_K3_T50 Zc6_npo_K3_fast_inner Zc7_npo_K3_r4; do
    log "$name:"
    report_metrics "$name"
done
