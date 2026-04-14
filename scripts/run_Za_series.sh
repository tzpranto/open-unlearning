#!/bin/bash
# Za-series: LoRA-BiAL with Adam optimizers (replacing manual SGD)
#
# Z-series showed manual SGD is too slow — LoRA params barely moved.
# Adam provides adaptive per-parameter LR + momentum → much larger effective updates.
# Z3 (K=10) was best at δ=+0.075; all others ≈ δ=+0.06 (barely above PerTA baseline).
#
# Za0: PerTA 3.5 + r16 + NPO K=5 (Adam baseline, compare Z0)
# Za1: PerTA 3.5 + r16 + NPO K=10 (Adam + K=10, compare Z3 best)
# Za2: PerTA 3.5 + r16 + GA K=5 (Adam + GA, compare Z2)
# Za3: PerTA 3.5 + r16 + GA K=10 (Adam + GA strong inner)
# Za4: PerTA 3.5 + r16 + NPO K=10 + higher outer LR (push harder)
# Za5: PerTA 3.5 + r16 + GA K=10 + T=50 (Adam GA long, compare Z6)
# Za6: PerTA 3.5 + r4 + NPO K=10 (smaller LoRA + Adam)
# Za7: PerTA 1.5 + r16 + NPO K=10 (milder PerTA + Adam)
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
PROGRESS="$BASE/saves/unlearn/_Za_series_progress.log"

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
)

log "=== Za-series: LoRA-BiAL with Adam ==="

# ═══════════════════════════════════════════════════════════════════
# Za0: PerTA 3.5 + r16 + NPO K=5 (Adam baseline)
# Direct comparison with Z0 (SGD): Z0 got fk=0.292 rk=0.395 δ=+0.064
# Adam should produce much larger LoRA updates per step
# ═══════════════════════════════════════════════════════════════════
run_experiment "Za0_adam_r16_npo_K5" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.args.gradient_accumulation_steps=1

# ═══════════════════════════════════════════════════════════════════
# Za1: PerTA 3.5 + r16 + NPO K=10 (Adam + strong inner)
# Z3 (SGD K=10) was best at δ=+0.075. Adam should amplify this.
# ═══════════════════════════════════════════════════════════════════
run_experiment "Za1_adam_r16_npo_K10" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=10 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.args.gradient_accumulation_steps=1

# ═══════════════════════════════════════════════════════════════════
# Za2: PerTA 3.5 + r16 + GA K=5 (Adam + GA)
# GA through LoRA didn't saturate but SGD updates too small.
# Adam should fix the update magnitude issue.
# ═══════════════════════════════════════════════════════════════════
run_experiment "Za2_adam_r16_ga_K5" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=ga \
    trainer.method_args.ga_clip=1.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.args.gradient_accumulation_steps=1

# ═══════════════════════════════════════════════════════════════════
# Za3: PerTA 3.5 + r16 + GA K=10 (Adam + GA + strong inner)
# Best of both: GA doesn't saturate + K=10 inner correction + Adam
# ═══════════════════════════════════════════════════════════════════
run_experiment "Za3_adam_r16_ga_K10" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=ga \
    trainer.method_args.ga_clip=1.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=10 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.args.gradient_accumulation_steps=1

# ═══════════════════════════════════════════════════════════════════
# Za4: PerTA 3.5 + r16 + NPO K=10 + higher outer LR
# Push Adam harder on outer loop — 5e-4 is standard LoRA LR
# ═══════════════════════════════════════════════════════════════════
run_experiment "Za4_adam_r16_npo_K10_lr5e4" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=10 \
    trainer.method_args.eta_theta=5e-4 \
    trainer.method_args.eta_in=5e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.args.gradient_accumulation_steps=1

# ═══════════════════════════════════════════════════════════════════
# Za5: PerTA 3.5 + r16 + GA K=10 + T=50 (Adam GA long)
# Z6 (SGD GA long) got δ=+0.066. Adam should do much better.
# ═══════════════════════════════════════════════════════════════════
run_experiment "Za5_adam_r16_ga_K10_T50" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=ga \
    trainer.method_args.ga_clip=1.0 \
    trainer.method_args.T=50 \
    trainer.method_args.K=10 \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.args.gradient_accumulation_steps=1

# ═══════════════════════════════════════════════════════════════════
# Za6: PerTA 3.5 + r4 + NPO K=10 (smaller LoRA + Adam)
# r4 showed best rk recovery in prior LoRA exps. With Adam + K=10.
# ═══════════════════════════════════════════════════════════════════
run_experiment "Za6_adam_r4_npo_K10" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=4 \
    trainer.method_args.lora_alpha=8 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=10 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.70 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.args.gradient_accumulation_steps=1

# ═══════════════════════════════════════════════════════════════════
# Za7: PerTA 1.5 + r16 + NPO K=10 (milder PerTA + Adam)
# PerTA 1.5 has more fk headroom (fk=0.537). Adam bilevel
# should push fk down harder than SGD (Z5 got δ=+0.050).
# ═══════════════════════════════════════════════════════════════════
run_experiment "Za7_adam_perta15_npo_K10" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=1.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=10 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.85 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.args.gradient_accumulation_steps=1

log "=== Za-series complete ==="

log "=== Za-series results ==="
for name in Za0_adam_r16_npo_K5 Za1_adam_r16_npo_K10 Za2_adam_r16_ga_K5 \
            Za3_adam_r16_ga_K10 Za4_adam_r16_npo_K10_lr5e4 Za5_adam_r16_ga_K10_T50 \
            Za6_adam_r4_npo_K10 Za7_adam_perta15_npo_K10; do
    log "$name:"
    report_metrics "$name"
done
