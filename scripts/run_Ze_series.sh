#!/bin/bash
# Ze-series: Fine-tune outer LR around Zd6 champion (olr=5e-5)
#
# Zd6 (β=4.0, K=3, olr=5e-5) = δ=+0.107 — new champion.
# Gentler outer correction preserves more retain.
# Now explore outer LR + inner LR + β fine-tuning.
#
# Ze0: olr=3e-5 (even gentler outer)
# Ze1: olr=7e-5 (between 5e-5 and 1e-4)
# Ze2: olr=5e-5, ilr=3e-4 (faster inner with gentle outer)
# Ze3: olr=5e-5, β=5.0 (combine both winning ingredients)
# Ze4: olr=5e-5, β=3.0 (slower saturation with gentle outer)
# Ze5: olr=5e-5, ε=0.50 (tighter constraint)
# Ze6: olr=5e-5, K=4 (one more inner step)
# Ze7: olr=2e-5 (very gentle outer — limiting case)
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
PROGRESS="$BASE/saves/unlearn/_Ze_series_progress.log"

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

# Base config (Zd6 champion)
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
    trainer.method_args.K=3
    trainer.method_args.npo_beta=4.0
    trainer.method_args.epsilon=0.70
    trainer.method_args.rho=0.1
    trainer.method_args.lambda_init=1.0
)

log "=== Ze-series: outer LR fine-tuning ==="

# Ze0: olr=3e-5 (even gentler)
run_experiment "Ze0_olr3e5" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4

# Ze1: olr=7e-5 (between 5e-5 and 1e-4)
run_experiment "Ze1_olr7e5" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.eta_theta=7e-5 \
    trainer.method_args.eta_in=2e-4

# Ze2: olr=5e-5, ilr=3e-4 (faster inner)
run_experiment "Ze2_ilr3e4" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=3e-4

# Ze3: olr=5e-5, β=5.0
run_experiment "Ze3_beta5_olr5e5" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.npo_beta=5.0

# Ze4: olr=5e-5, β=3.0
run_experiment "Ze4_beta3_olr5e5" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.npo_beta=3.0

# Ze5: olr=5e-5, ε=0.50
run_experiment "Ze5_eps050_olr5e5" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.50

# Ze6: olr=5e-5, K=4
run_experiment "Ze6_K4_olr5e5" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.eta_theta=5e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.K=4

# Ze7: olr=2e-5 (very gentle)
run_experiment "Ze7_olr2e5" \
    "${BASE_COMMON[@]}" \
    trainer.method_args.eta_theta=2e-5 \
    trainer.method_args.eta_in=2e-4

log "=== Ze-series complete ==="

log "=== Ze-series results ==="
for name in Ze0_olr3e5 Ze1_olr7e5 Ze2_ilr3e4 Ze3_beta5_olr5e5 \
            Ze4_beta3_olr5e5 Ze5_eps050_olr5e5 Ze6_K4_olr5e5 Ze7_olr2e5; do
    log "$name:"
    report_metrics "$name"
done
