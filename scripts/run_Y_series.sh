#!/bin/bash
# Y-series: Three-track exploration
#
# Track A: Bilevel from PerTA-masked th0.5 (rk=GOLD, push fk down)
#   ref_model = target (NOT deepcopy of init — avoids NPO saturation)
#   A1: G1-style (accum=32, K=1, T=25)
#   A2: A1 + warm ALM (λ_init=2.0, ρ=0.5)
#   A3: X1f recipe (accum=1, K=10, T=15)
#
# Track B: Bilevel from PerTA-masked th0.3 (best δ, balanced)
#   ref_model = target
#   B1: G1-style (accum=32, K=1, T=25)
#   B2: X1f recipe (accum=1, K=10, T=15)
#
# Track C: Pure bilevel from target with Fisher disjoint masks
#   Outer: NPO on w>0.3 (14% params). Inner: CE on w<=0.3 (86% params)
#   C1: accum=1, K=10, T=25
#   C2: C1 + implicit (FD-HVP)
#
# Restart-resistant: skip logic checks for eval JSON or model weights.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="/datadrive/forked/open-unlearning"
SAVES="$BASE/saves/unlearn"
PYTHON="${CONDA_PREFIX:-/datadrive/conda/envs/unlearning}/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
FISHER_CACHE="saves/unlearn/_perta_fisher_cache_News_n64.pt"
PROGRESS="$BASE/saves/unlearn/_Y_series_progress.log"

cd "$BASE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

is_done() {
    local name=$1
    [ -f "$SAVES/$name/evals/MUSE_EVAL.json" ]
}

has_model() {
    local name=$1
    compgen -G "$SAVES/$name/model-*.safetensors" > /dev/null 2>&1
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

    if has_model "$name"; then
        log "[SKIP-TRAIN] $name — model exists, running eval only"
    else
        log "[TRAIN] $name"
        "$PYTHON" src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/sibl \
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

# ═══════════════════════════════════════════════════════════════════════════════
# Track A: Bilevel from PerTA-masked th0.5 (rk=GOLD ceiling)
# Init: perta_masked_l1.5_th0.5 (fk=0.591, rk=0.552)
# Ref: target model (muse-bench/MUSE-News_target) — avoids NPO saturation
# ═══════════════════════════════════════════════════════════════════════════════
log "=== Track A: Bilevel from PerTA-masked th0.5 ==="

PERTA_TH05="saves/unlearn/perta_masked_l1.5_th0.5"
TARGET_REF="muse-bench/MUSE-News_target"

# A1: G1-style bilevel (accum=32, K=1, T=25, steering)
run_experiment "Y_A1_g1_from_th05" \
    model.model_args.pretrained_model_name_or_path=${PERTA_TH05} \
    trainer.method_args.ref_model_path=${TARGET_REF} \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=1 \
    trainer.method_args.eta_theta=2e-4 \
    trainer.method_args.eta_in=1e-4 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=0 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.use_sparsity=false \
    trainer.method_args.regularization_type=none \
    trainer.method_args.use_steering=true \
    trainer.method_args.steering_coeff=5.0 \
    trainer.method_args.steering_only=true \
    trainer.method_args.use_fisher_weighting=false \
    trainer.method_args.inner_contrastive=false \
    trainer.method_args.inner_repr_anchor=false \
    trainer.method_args.gradient_projection=false \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=32 \
    trainer.args.num_train_epochs=25

# A2: A1 + warm ALM (λ_init=2.0, ρ=0.5)
run_experiment "Y_A2_warm_alm_th05" \
    model.model_args.pretrained_model_name_or_path=${PERTA_TH05} \
    trainer.method_args.ref_model_path=${TARGET_REF} \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=1 \
    trainer.method_args.eta_theta=2e-4 \
    trainer.method_args.eta_in=1e-4 \
    trainer.method_args.rho=0.5 \
    trainer.method_args.lambda_init=2.0 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.use_sparsity=false \
    trainer.method_args.regularization_type=none \
    trainer.method_args.use_steering=true \
    trainer.method_args.steering_coeff=5.0 \
    trainer.method_args.steering_only=true \
    trainer.method_args.use_fisher_weighting=false \
    trainer.method_args.inner_contrastive=false \
    trainer.method_args.inner_repr_anchor=false \
    trainer.method_args.gradient_projection=false \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=32 \
    trainer.args.num_train_epochs=25

# A3: X1f recipe (accum=1, K=10, T=15, npo_beta=0.5)
run_experiment "Y_A3_x1f_from_th05" \
    model.model_args.pretrained_model_name_or_path=${PERTA_TH05} \
    trainer.method_args.ref_model_path=${TARGET_REF} \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=0.5 \
    trainer.method_args.T=15 \
    trainer.method_args.K=10 \
    trainer.method_args.eta_theta=2e-4 \
    trainer.method_args.eta_in=5e-4 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.use_sparsity=false \
    trainer.method_args.regularization_type=none \
    trainer.method_args.use_steering=false \
    trainer.method_args.use_fisher_weighting=false \
    trainer.method_args.inner_contrastive=false \
    trainer.method_args.inner_repr_anchor=false \
    trainer.method_args.gradient_projection=false \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.num_train_epochs=15

log "=== Track A complete ==="

# ═══════════════════════════════════════════════════════════════════════════════
# Track B: Bilevel from PerTA-masked th0.3 (best δ=+0.063)
# Init: perta_masked_l1.5_th0.3 (fk=0.510, rk=0.514)
# Ref: target model
# ═══════════════════════════════════════════════════════════════════════════════
log "=== Track B: Bilevel from PerTA-masked th0.3 ==="

PERTA_TH03="saves/unlearn/perta_masked_l1.5_th0.3"

# B1: G1-style (accum=32, K=1, T=25, steering)
run_experiment "Y_B1_g1_from_th03" \
    model.model_args.pretrained_model_name_or_path=${PERTA_TH03} \
    trainer.method_args.ref_model_path=${TARGET_REF} \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=1 \
    trainer.method_args.eta_theta=2e-4 \
    trainer.method_args.eta_in=1e-4 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=0 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.use_sparsity=false \
    trainer.method_args.regularization_type=none \
    trainer.method_args.use_steering=true \
    trainer.method_args.steering_coeff=5.0 \
    trainer.method_args.steering_only=true \
    trainer.method_args.use_fisher_weighting=false \
    trainer.method_args.inner_contrastive=false \
    trainer.method_args.inner_repr_anchor=false \
    trainer.method_args.gradient_projection=false \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=32 \
    trainer.args.num_train_epochs=25

# B2: X1f recipe (accum=1, K=10, T=15)
run_experiment "Y_B2_x1f_from_th03" \
    model.model_args.pretrained_model_name_or_path=${PERTA_TH03} \
    trainer.method_args.ref_model_path=${TARGET_REF} \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=0.5 \
    trainer.method_args.T=15 \
    trainer.method_args.K=10 \
    trainer.method_args.eta_theta=2e-4 \
    trainer.method_args.eta_in=5e-4 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.use_sparsity=false \
    trainer.method_args.regularization_type=none \
    trainer.method_args.use_steering=false \
    trainer.method_args.use_fisher_weighting=false \
    trainer.method_args.inner_contrastive=false \
    trainer.method_args.inner_repr_anchor=false \
    trainer.method_args.gradient_projection=false \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.num_train_epochs=15

log "=== Track B complete ==="

# ═══════════════════════════════════════════════════════════════════════════════
# Track C: Pure bilevel from target with Fisher disjoint masks
# Init: target model (muse-bench/MUSE-News_target)
# Outer mask: w > 0.3 (~14% params, forget-dominant for NPO)
# Inner mask: w <= 0.3 (~86% params, retain-dominant for CE)
# No ref_model_path needed — deepcopy(target) is correct for standard setup
# ═══════════════════════════════════════════════════════════════════════════════
log "=== Track C: Fisher disjoint mask bilevel ==="

# C1: accum=1, K=10, T=25, npo_beta=2.0, warm ALM
run_experiment "Y_C1_fisher_mask" \
    trainer.method_args.fisher_mask_path=${FISHER_CACHE} \
    trainer.method_args.fisher_mask_threshold=0.3 \
    trainer.method_args.fisher_mask_alpha=1.0 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=10 \
    trainer.method_args.eta_theta=2e-4 \
    trainer.method_args.eta_in=5e-4 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.use_sparsity=false \
    trainer.method_args.regularization_type=none \
    trainer.method_args.use_steering=true \
    trainer.method_args.steering_coeff=5.0 \
    trainer.method_args.steering_only=true \
    trainer.method_args.use_fisher_weighting=false \
    trainer.method_args.inner_contrastive=false \
    trainer.method_args.inner_repr_anchor=false \
    trainer.method_args.gradient_projection=false \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.num_train_epochs=25

# C2: C1 + FD-HVP implicit correction (full DS-BiAL)
run_experiment "Y_C2_fisher_implicit" \
    trainer.method_args.fisher_mask_path=${FISHER_CACHE} \
    trainer.method_args.fisher_mask_threshold=0.3 \
    trainer.method_args.fisher_mask_alpha=1.0 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=10 \
    trainer.method_args.eta_theta=2e-4 \
    trainer.method_args.eta_in=5e-4 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.use_sparsity=false \
    trainer.method_args.regularization_type=none \
    trainer.method_args.use_steering=true \
    trainer.method_args.steering_coeff=5.0 \
    trainer.method_args.steering_only=true \
    trainer.method_args.use_implicit=true \
    trainer.method_args.implicit_solver=neumann \
    trainer.method_args.implicit_blockwise=true \
    trainer.method_args.implicit_block_last_n_layers=2 \
    trainer.method_args.implicit_block_include_attn=true \
    trainer.method_args.implicit_block_include_mlp=true \
    trainer.method_args.neumann_variant=richardson \
    trainer.method_args.neumann_steps=2 \
    trainer.method_args.neumann_mu=1.0 \
    trainer.method_args.neumann_alpha_default=0.01 \
    trainer.method_args.neumann_use_probe_alpha=false \
    trainer.method_args.use_fisher_weighting=false \
    trainer.method_args.inner_contrastive=false \
    trainer.method_args.inner_repr_anchor=false \
    trainer.method_args.gradient_projection=false \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.num_train_epochs=25

log "=== Track C complete ==="

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
log "=== Y-series complete ==="
log "Results summary:"
for name in Y_A1_g1_from_th05 Y_A2_warm_alm_th05 Y_A3_x1f_from_th05 \
            Y_B1_g1_from_th03 Y_B2_x1f_from_th03 \
            Y_C1_fisher_mask Y_C2_fisher_implicit; do
    log "$name:"
    report_metrics "$name"
done
