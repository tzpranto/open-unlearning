#!/bin/bash
# Z-series: LoRA-BiAL — LoRA bilevel from PerTA init
#
# Key innovation: ref model = base model without LoRA (zero memory overhead)
# LoRA constrains update space → no accum=1 requirement, no OOM
#
# Z0: PerTA 3.5 + LoRA r16 + NPO bilevel (baseline)
# Z1: PerTA 3.5 + LoRA r4  + NPO bilevel (r4 had best rk in prior LoRA experiments)
# Z2: PerTA 3.5 + LoRA r16 + GA bilevel  (gradient ascent — no saturation)
# Z3: PerTA 3.5 + LoRA r16 + NPO + K=10 (stronger inner correction)
# Z4: PerTA 3.5 + LoRA r16 + NPO + accum=4 (LoRA handles larger batches)
# Z5: PerTA 1.5 + LoRA r16 + NPO (milder PerTA, more fk headroom)
# Z6: PerTA 3.5 + LoRA r16 + GA + K=10 + T=50 (GA with more training)
# Z7: No PerTA + LoRA r16 + NPO from target (pure LoRA bilevel, control)
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
PROGRESS="$BASE/saves/unlearn/_Z_series_progress.log"

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

log "=== Z-series: LoRA-BiAL ==="

# ═══════════════════════════════════════════════════════════════════
# Z0: PerTA 3.5 + LoRA r16 + NPO (baseline)
# PerTA 3.5 base: fk=0.282, rk=0.396, δ=+0.071
# LoRA r16 on PerTA 3.5 (prior CE-only): fk=0.329, rk=0.401, δ=+0.050
# Goal: bilevel NPO prevents fk leakage → rk > 0.401
# ═══════════════════════════════════════════════════════════════════
run_experiment "Z0_lora_bial_r16_npo" \
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
# Z1: PerTA 3.5 + LoRA r4 + NPO (r4 showed best rk recovery +0.035)
# ═══════════════════════════════════════════════════════════════════
run_experiment "Z1_lora_bial_r4_npo" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lora_r=4 \
    trainer.method_args.lora_alpha=8 \
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
# Z2: PerTA 3.5 + LoRA r16 + GA (gradient ascent — no saturation issue)
# GA through LoRA: constrained destructiveness, direct forget push
# ═══════════════════════════════════════════════════════════════════
run_experiment "Z2_lora_bial_r16_ga" \
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
# Z3: PerTA 3.5 + LoRA r16 + NPO + K=10 (stronger inner correction)
# X1f showed K=10 is critical for bilevel. Test with LoRA.
# ═══════════════════════════════════════════════════════════════════
run_experiment "Z3_lora_bial_r16_K10" \
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
# Z4: PerTA 3.5 + LoRA r16 + NPO + accum=4
# LoRA should handle larger effective batch (OOM-free)
# Tests if accum=1 requirement is relaxed with LoRA constraint
# ═══════════════════════════════════════════════════════════════════
run_experiment "Z4_lora_bial_r16_accum4" \
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
    trainer.args.gradient_accumulation_steps=4

# ═══════════════════════════════════════════════════════════════════
# Z5: PerTA 1.5 + LoRA r16 + NPO (milder PerTA, more fk headroom)
# PerTA 1.5: fk=0.537, rk=0.513, δ=+0.047
# LoRA bilevel may push fk down while recovering rk
# ═══════════════════════════════════════════════════════════════════
run_experiment "Z5_lora_bial_perta15" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=1.5 \
    trainer.method_args.lora_r=16 \
    trainer.method_args.lora_alpha=32 \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.T=25 \
    trainer.method_args.K=5 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.85 \
    trainer.method_args.rho=0.1 \
    trainer.method_args.lambda_init=1.0 \
    trainer.args.gradient_accumulation_steps=1

# ═══════════════════════════════════════════════════════════════════
# Z6: PerTA 3.5 + LoRA r16 + GA + K=10 + T=50 (GA with more training)
# GA doesn't saturate → more steps may help
# ═══════════════════════════════════════════════════════════════════
run_experiment "Z6_lora_bial_ga_long" \
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
# Z7: No PerTA + LoRA r16 + NPO from target (control)
# Pure LoRA bilevel without PerTA init — measures LoRA bilevel alone
# ═══════════════════════════════════════════════════════════════════
run_experiment "Z7_lora_bial_no_perta" \
    "${COMMON[@]}" \
    trainer.method_args.perta_lambda=0.0 \
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

log "=== Z-series complete ==="

log "=== Z-series results ==="
for name in Z0_lora_bial_r16_npo Z1_lora_bial_r4_npo Z2_lora_bial_r16_ga \
            Z3_lora_bial_r16_K10 Z4_lora_bial_r16_accum4 Z5_lora_bial_perta15 \
            Z6_lora_bial_ga_long Z7_lora_bial_no_perta; do
    log "$name:"
    report_metrics "$name"
done
