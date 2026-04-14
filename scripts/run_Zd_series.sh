#!/bin/bash
# Zd-series: Fine-tuning around Zc2 sweet spot (β=4.0, K=3, T=25)
#
# Zc2 = δ=+0.099 (fk=0.273, rk=0.419) — current champion.
# Now explore the local neighborhood:
#   Axis 1: β ∈ {3.0, 5.0, 6.0} (between β=2→+0.089 and β=8→+0.078)
#   Axis 2: K=4 (between K=3→+0.099 and K=5→+0.077)
#   Axis 3: ε tuning (constraint tightness)
#   Axis 4: outer LR (correction strength)
#   Axis 5: PerTA λ (initialization strength)
#
# Also: detect NPO saturation and stop outer loop to prevent stale momentum
# T=50 was destructive → T=25 is optimal.
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
PROGRESS="$BASE/saves/unlearn/_Zd_series_progress.log"

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

# Base config (Zc2 champion replicated)
BASE_COMMON=(
    trainer.method_args.fisher_cache_path=${FISHER_CACHE}
    trainer.args.per_device_train_batch_size=2
    trainer.args.gradient_accumulation_steps=1
    trainer.args.num_train_epochs=25
    trainer.method_args.perta_lambda=3.5
    trainer.method_args.lora_r=16
    trainer.method_args.lora_alpha=32
    trainer.method_args.forget_loss_type=npo
    trainer.method_args.T=25
    trainer.method_args.epsilon=0.70
    trainer.method_args.rho=0.1
    trainer.method_args.lambda_init=1.0
    trainer.method_args.eta_theta=1e-4
    trainer.method_args.eta_in=2e-4
)

log "=== Zd-series: fine-tuning around Zc2 (β=4, K=3) ==="

# ═══════════════════════════════════════════════════════════════════
# Zd0: β=3.0 K=3 (between β=2→+0.089 and β=4→+0.099)
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zd0_beta3_K3" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.npo_beta=3.0 \
    trainer.method_args.K=3

# ═══════════════════════════════════════════════════════════════════
# Zd1: β=5.0 K=3 (between β=4→+0.099 and β=8→+0.078)
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zd1_beta5_K3" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.npo_beta=5.0 \
    trainer.method_args.K=3

# ═══════════════════════════════════════════════════════════════════
# Zd2: β=6.0 K=3 (narrow the peak between β=4 and β=8)
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zd2_beta6_K3" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.npo_beta=6.0 \
    trainer.method_args.K=3

# ═══════════════════════════════════════════════════════════════════
# Zd3: β=4.0 K=4 (between K=3→+0.099 and K=5→+0.077)
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zd3_beta4_K4" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.npo_beta=4.0 \
    trainer.method_args.K=4

# ═══════════════════════════════════════════════════════════════════
# Zd4: β=4.0 K=3 ε=0.50 (tighter retain constraint)
# Tighter ε → ALM penalizes retain degradation more → stronger rk
# But may limit forget correction
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zd4_beta4_K3_eps050" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.npo_beta=4.0 \
    trainer.method_args.K=3 \
    trainer.method_args.epsilon=0.50

# ═══════════════════════════════════════════════════════════════════
# Zd5: β=4.0 K=3 ε=0.90 (looser retain constraint)
# Looser ε → more freedom for forget correction → lower fk?
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zd5_beta4_K3_eps090" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.npo_beta=4.0 \
    trainer.method_args.K=3 \
    trainer.method_args.epsilon=0.90

# ═══════════════════════════════════════════════════════════════════
# Zd6: β=4.0 K=3 outer_lr=5e-5 (gentler outer correction)
# Softer outer step → less damage to retain during initial correction
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zd6_beta4_K3_olr5e5" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.npo_beta=4.0 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=5e-5

# ═══════════════════════════════════════════════════════════════════
# Zd7: β=4.0 K=3 PerTA λ=3.0 (slightly less aggressive PerTA)
# λ=3.0 gives slightly higher fk but better rk starting point
# More rk headroom for bilevel to exploit
# ═══════════════════════════════════════════════════════════════════
run_experiment "Zd7_beta4_K3_perta30" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.npo_beta=4.0 \
    trainer.method_args.K=3 \
    trainer.method_args.perta_lambda=3.0

log "=== Zd-series complete ==="

log "=== Zd-series results ==="
for name in Zd0_beta3_K3 Zd1_beta5_K3 Zd2_beta6_K3 Zd3_beta4_K4 \
            Zd4_beta4_K3_eps050 Zd5_beta4_K3_eps090 Zd6_beta4_K3_olr5e5 Zd7_beta4_K3_perta30; do
    log "$name:"
    report_metrics "$name"
done
