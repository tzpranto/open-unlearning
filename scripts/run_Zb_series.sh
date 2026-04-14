#!/bin/bash
# Zb-series: LoRA-BiAL Adam — KL divergence + NPO-β tuning
#
# Za-series finding: Za0 (NPO K=5 Adam) = δ=+0.077 (best).
# Bottleneck: NPO saturates after ~10 outer steps → outer loop loses signal.
#
# This series explores:
# 1. KL divergence as forget loss — never saturates (main innovation)
# 2. Lower npo_beta — slower NPO saturation
# 3. K tuning around the K=3-5 sweet spot
# 4. LR tuning for inner/outer balance
#
# Zb0: KL K=5 (KL divergence instead of NPO — key test)
# Zb1: KL K=3 (KL + fewer inner steps)
# Zb2: KL K=5 + higher outer LR (more aggressive KL correction)
# Zb3: NPO β=0.5 K=5 (gentler NPO → slower saturation)
# Zb4: NPO β=1.0 K=5 (moderate NPO)
# Zb5: NPO β=2.0 K=3 (standard NPO + fewer inner steps)
# Zb6: KL K=5 + r4 (smaller LoRA + KL)
# Zb7: KL K=5 + PerTA 2.5 (moderate PerTA init + KL)
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
PROGRESS="$BASE/saves/unlearn/_Zb_series_progress.log"

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
    trainer.args.num_train_epochs=25
    trainer.args.gradient_accumulation_steps=1
)

log "=== Zb-series: KL divergence + NPO-β tuning ==="

# ═══════════════════════════════════════════════════════════════════
# Zb0: KL K=5 (KEY EXPERIMENT)
# KL(model || ref) on forget: never saturates, continuous gradient
# Should keep outer loop active for all 25 steps (vs NPO → dead by step 10)
# Compare: Za0 (NPO K=5) got δ=+0.077
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zb0_kl_K5" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=kl \
    trainer.method_args.T=25 \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0

# ═══════════════════════════════════════════════════════════════════
# Zb1: KL K=3 (fewer inner steps + KL)
# K=3 is the conservative end — less forget leakage per outer step
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zb1_kl_K3" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=kl \
    trainer.method_args.T=25 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0

# ═══════════════════════════════════════════════════════════════════
# Zb2: KL K=5 + higher outer LR
# Since KL doesn't saturate, stronger outer correction should help
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zb2_kl_K5_lr5e4" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=kl \
    trainer.method_args.T=25 \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=5e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0

# ═══════════════════════════════════════════════════════════════════
# Zb3: NPO β=0.5 K=5 (gentler NPO, slower saturation)
# β=0.5 makes logsigmoid curve gentler → slower approach to 0
# Same config as Za0 but with β=0.5 instead of 2.0
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zb3_npo_beta05_K5" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=0.5 \
    trainer.method_args.T=25 \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0

# ═══════════════════════════════════════════════════════════════════
# Zb4: NPO β=1.0 K=5 (moderate NPO)
# Between Za0's β=2.0 (too fast saturation) and Zb3's β=0.5
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zb4_npo_beta10_K5" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=1.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0

# ═══════════════════════════════════════════════════════════════════
# Zb5: NPO β=2.0 K=3 (standard NPO, fewer inner steps)
# If K=5 is too many with Adam, K=3 might be the sweet spot
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zb5_npo_K3" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0

# ═══════════════════════════════════════════════════════════════════
# Zb6: KL K=5 + r4 (smaller LoRA + KL)
# r4 constrains updates more → less forget leakage per inner step
# Combined with KL's non-saturating outer signal
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zb6_kl_K5_r4" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=4 \
    trainer.method_args.lora_alpha=8 \
    trainer.method_args.forget_loss_type=kl \
    trainer.method_args.T=25 \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0

# ═══════════════════════════════════════════════════════════════════
# Zb7: KL K=5 + PerTA 2.5 (moderate PerTA + KL)
# PerTA 3.5 is aggressive (fk=0.282 but rk=0.396 low)
# PerTA 2.5 should give more rk headroom for bilevel to work
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zb7_kl_K5_perta25" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=2.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=kl \
    trainer.method_args.T=25 \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0

log "=== Zb-series complete ==="

log "=== Zb-series results ==="
for name in Zb0_kl_K5 Zb1_kl_K3 Zb2_kl_K5_lr5e4 Zb3_npo_beta05_K5 \
            Zb4_npo_beta10_K5 Zb5_npo_K3 Zb6_kl_K5_r4 Zb7_kl_K5_perta25; do
    log "$name:"
    report_metrics "$name"
done
