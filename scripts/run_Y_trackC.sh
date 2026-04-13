#!/bin/bash
# Y-series Track C: Fisher disjoint mask bilevel (fix OOM by disabling steering)
#
# C1: accum=1, K=10, T=25, Fisher masks (w>0.3 outer, w<=0.3 inner)
# C2: C1 + FD-HVP implicit correction
# C3: C1 with accum=4 (larger effective batch, more stable gradients)
#
# Track A/B showed NPO with target-ref saturates from PerTA init.
# Track C avoids this by using standard target init + deepcopy ref.
# Fisher masks disentangle outer (forget params) from inner (retain params).
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
PROGRESS="$BASE/saves/unlearn/_Y_trackC_progress.log"

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

    # Clean up incomplete model (OOM crash recovery)
    if [ -d "$SAVES/$name" ] && ! has_model "$name"; then
        log "[CLEANUP] $name — removing incomplete directory"
        rm -rf "$SAVES/$name"
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
# Track C: Pure bilevel from target with Fisher disjoint masks
# Init: target model (muse-bench/MUSE-News_target)
# Outer mask: w > 0.3 (~14% params, forget-dominant for NPO)
# Inner mask: w <= 0.3 (~86% params, retain-dominant for CE)
# No steering (saves ~8GB VRAM — was marginal in prior experiments)
# ═══════════════════════════════════════════════════════════════════════════════
log "=== Track C: Fisher disjoint mask bilevel ==="

# Common overrides for Track C
COMMON_C=(
    trainer.method_args.fisher_mask_path=${FISHER_CACHE}
    trainer.method_args.fisher_mask_threshold=0.3
    trainer.method_args.fisher_mask_alpha=1.0
    trainer.method_args.forget_loss_type=npo
    trainer.method_args.npo_beta=2.0
    trainer.method_args.eta_theta=2e-4
    trainer.method_args.eta_in=5e-4
    trainer.method_args.rho=0.1
    trainer.method_args.lambda_init=1.0
    trainer.method_args.epsilon=0.70
    trainer.method_args.use_sparsity=false
    trainer.method_args.regularization_type=none
    trainer.method_args.use_steering=false
    trainer.method_args.use_fisher_weighting=false
    trainer.method_args.inner_contrastive=false
    trainer.method_args.inner_repr_anchor=false
    trainer.method_args.gradient_projection=false
    trainer.method_args.use_implicit=false
    trainer.args.per_device_train_batch_size=1
)

# C1: accum=1, K=10, T=25 (X1f recipe + Fisher masks)
run_experiment "Y_C1_fisher_mask" \
    "${COMMON_C[@]}" \
    trainer.method_args.T=25 \
    trainer.method_args.K=10 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.num_train_epochs=25

# C2: C1 + FD-HVP implicit correction (full DS-BiAL)
run_experiment "Y_C2_fisher_implicit" \
    "${COMMON_C[@]}" \
    trainer.method_args.T=25 \
    trainer.method_args.K=10 \
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
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.num_train_epochs=25

# C3: Fisher masks + accum=4 (larger batch, more stable)
run_experiment "Y_C3_fisher_accum4" \
    "${COMMON_C[@]}" \
    trainer.method_args.T=25 \
    trainer.method_args.K=10 \
    trainer.args.gradient_accumulation_steps=4 \
    trainer.args.num_train_epochs=25

log "=== Track C complete ==="

log "=== Track C results ==="
for name in Y_C1_fisher_mask Y_C2_fisher_implicit Y_C3_fisher_accum4; do
    log "$name:"
    report_metrics "$name"
done
