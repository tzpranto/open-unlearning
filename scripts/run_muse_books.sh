#!/bin/bash
# MUSE Books benchmark: gold evals → baselines → PerTA → LoRA-BiAL
#
# Pipeline:
#   1. Gold standard evals (retrain, target, pretrained)
#   2. Gradient baselines: GradAscent, GradDiff, NPO, SimNPO, RMU, BLURNPO
#   3. Fisher cache + PerTA lambda sweep
#   4. LoRA-BiAL with Ze0 champion config
#
# Restart-resistant: every step checks if output already exists.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="/datadrive/forked/open-unlearning"
SAVES="$BASE/saves/unlearn"
EVAL_SAVES="$BASE/saves/eval"
PYTHON="${CONDA_PREFIX:-/datadrive/conda/envs/unlearning}/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="Books"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
FISHER_CACHE="saves/unlearn/_perta_fisher_cache_Books_n64.pt"
PROGRESS="$BASE/saves/unlearn/_muse_books_progress.log"

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
fk = d.get('forget_knowmem_ROUGE', {}).get('agg_value', d.get('forget_knowmem_ROUGE', {}).get('mean', -1))
vm = d.get('forget_verbmem_ROUGE', {}).get('agg_value', d.get('forget_verbmem_ROUGE', {}).get('mean', -1))
rk = d.get('retain_knowmem_ROUGE', {}).get('agg_value', d.get('retain_knowmem_ROUGE', {}).get('mean', -1))
ex = d.get('extraction_strength', {}).get('agg_value', d.get('extraction_strength', {}).get('mean', -1))
delta = rk - (0.55*fk + 0.17)
print(f'  fk={fk:.4f}  vm={vm:.4f}  rk={rk:.4f}  ex={ex:.4f}  delta={delta:+.4f}')
" 2>/dev/null || echo "  (metrics parse failed)"
    fi
}

report_gold() {
    local label=$1
    local json=$2
    if [ -f "$json" ]; then
        "$PYTHON" -c "
import json
with open('$json') as f:
    d = json.load(f)
fk = d.get('forget_knowmem_ROUGE', {}).get('agg_value', -1)
vm = d.get('forget_verbmem_ROUGE', {}).get('agg_value', -1)
rk = d.get('retain_knowmem_ROUGE', {}).get('agg_value', -1)
ex = d.get('extraction_strength', {}).get('agg_value', -1)
delta = rk - (0.55*fk + 0.17)
print(f'  fk={fk:.4f}  vm={vm:.4f}  rk={rk:.4f}  ex={ex:.4f}  delta={delta:+.4f}')
" 2>/dev/null || echo "  (metrics parse failed)"
    else
        echo "  (not found)"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Gold standard evals (retrain, target, pretrained)
# Retrain and target already in saves/eval/. Only pretrained may be missing.
# ═══════════════════════════════════════════════════════════════════
log "=== MUSE Books benchmark ==="
log "=== Step 1: Gold standard evals ==="

RETRAIN_JSON="$EVAL_SAVES/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
TARGET_JSON="$EVAL_SAVES/muse_${MODEL}_${DATA_SPLIT}_target/MUSE_EVAL.json"
PRETRAINED_JSON="$EVAL_SAVES/muse_${MODEL}_${DATA_SPLIT}_pretrained/MUSE_EVAL.json"

# Only pretrained needs computing
if [ -f "$PRETRAINED_JSON" ]; then
    log "[SKIP] Pretrained eval exists"
else
    log "[EVAL] Pretrained (base) model"
    "$PYTHON" src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=muse_${MODEL}_${DATA_SPLIT}_pretrained \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=meta-llama/Llama-2-7b-hf \
        model.model_args.attn_implementation=sdpa \
        paths.output_dir=$EVAL_SAVES/muse_${MODEL}_${DATA_SPLIT}_pretrained \
        retain_logs_path=${RETAIN_LOGS} \
        2>&1 | tee /tmp/eval_books_pretrained.log
fi

log "Retrain (gold):"; report_gold "retrain" "$RETRAIN_JSON"
log "Target (finetuned):"; report_gold "target" "$TARGET_JSON"
log "Pretrained (base):"; report_gold "pretrained" "$PRETRAINED_JSON"

log "=== Gold evals done ==="

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Gradient baselines
# ═══════════════════════════════════════════════════════════════════
log "=== Step 2: Gradient baselines ==="

run_baseline() {
    local name=$1
    local experiment=$2
    shift 2
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
            experiment=unlearn/muse/${experiment} \
            model=${MODEL} \
            data_split=${DATA_SPLIT} \
            task_name=${name} \
            retain_logs_path=${RETAIN_LOGS} \
            model.model_args.attn_implementation=sdpa \
            "${overrides[@]}" \
            2>&1 | tee /tmp/train_books_${name}.log
    fi

    run_eval "$name"
    report_metrics "$name"
    log "[DONE] $name"
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
        2>&1 | tee /tmp/eval_books_${name}.log
}

# GradAscent
run_baseline "books_gradasc" "default" \
    trainer.args.gradient_checkpointing=true

# GradDiff
run_baseline "books_graddiff" "default" \
    trainer=GradDiff \
    trainer.args.gradient_checkpointing=true

# NPO
run_baseline "books_npo" "npo" \
    trainer.args.gradient_checkpointing=true

# SimNPO
run_baseline "books_simnpo" "simnpo" \
    trainer.args.gradient_checkpointing=true

# RMU
run_baseline "books_rmu" "rmu"

# BLURNPO
run_baseline "books_blurnpo" "blurnpo" \
    trainer.args.gradient_checkpointing=true

log "=== Gradient baselines complete ==="

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Fisher cache + PerTA lambda sweep
# ═══════════════════════════════════════════════════════════════════
log "=== Step 3: PerTA sweep ==="

# Fisher cache
if [ -f "$BASE/$FISHER_CACHE" ]; then
    log "[SKIP] Fisher cache exists: $FISHER_CACHE"
else
    log "[FISHER] Computing Fisher cache for Books..."
    "$PYTHON" scripts/perta_unlearn.py \
        --target "muse-bench/MUSE-Books_target" \
        --pretrained "meta-llama/Llama-2-7b-hf" \
        --data_split Books \
        --lambdas 999 \
        --alpha 1.0 \
        --n_samples 64 \
        --fisher_cache "$BASE/$FISHER_CACHE" \
        2>&1 | tee /tmp/fisher_books.log
    rm -rf "$SAVES/perta_l999.0_a1.0" 2>/dev/null
    log "[FISHER] Done."
fi

PERTA_LAMBDAS="1.0 1.5 2.0 2.5 3.0 3.5 4.0"

# Check which lambdas still need models
NEED_PERTA=()
for lam in $PERTA_LAMBDAS; do
    name="books_perta_l${lam}_a1.0"
    if has_model "$name"; then
        log "[SKIP] $name model exists"
    else
        NEED_PERTA+=("$lam")
    fi
done

if [ ${#NEED_PERTA[@]} -gt 0 ]; then
    log "[PERTA] Generating models for lambdas: ${NEED_PERTA[*]}"
    LAMBDA_STR=$(IFS=,; echo "${NEED_PERTA[*]}")
    "$PYTHON" -c "
import sys, os, json, shutil, torch, logging
sys.path.insert(0, '.')
from scripts.perta_unlearn import apply_perta, save_model
from transformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SAVES = '$SAVES'
fisher_cache = '$BASE/$FISHER_CACHE'
lambdas = [float(x) for x in '$LAMBDA_STR'.split(',')]

cached = torch.load(fisher_cache, map_location='cpu', weights_only=True)
fisher_forget, fisher_retain = cached['forget'], cached['retain']

logger.info('Loading target: muse-bench/MUSE-Books_target')
target_model = AutoModelForCausalLM.from_pretrained('muse-bench/MUSE-Books_target', torch_dtype=torch.bfloat16, device_map='cpu')
target_sd = {k: v.clone() for k, v in target_model.state_dict().items()}
target_config_dir = target_model.config._name_or_path
if not os.path.isdir(target_config_dir):
    from huggingface_hub import snapshot_download
    target_config_dir = snapshot_download('muse-bench/MUSE-Books_target', local_files_only=True)
del target_model

logger.info('Loading pretrained: meta-llama/Llama-2-7b-hf')
pretrained_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', torch_dtype=torch.bfloat16, device_map='cpu')
pretrained_sd = {k: v.clone() for k, v in pretrained_model.state_dict().items()}
pretrained_config_dir = pretrained_model.config._name_or_path
if not os.path.isdir(pretrained_config_dir):
    from huggingface_hub import snapshot_download
    pretrained_config_dir = snapshot_download('meta-llama/Llama-2-7b-hf', local_files_only=True)
del pretrained_model

for lam in lambdas:
    out_name = f'books_perta_l{lam:.1f}_a1.0'
    out_dir = os.path.join(SAVES, out_name)
    logger.info(f'Applying PerTA lambda={lam} -> {out_name}')
    result_sd = apply_perta(target_sd, pretrained_sd, fisher_forget, fisher_retain, lam=lam, alpha=1.0)
    save_model(result_sd, out_dir, pretrained_config_dir)
    os.makedirs(os.path.join(out_dir, 'evals'), exist_ok=True)
    del result_sd
    logger.info(f'Done: {out_name}')
" 2>&1 | tee /tmp/perta_books_sweep.log
fi

# Eval all PerTA models
for lam in $PERTA_LAMBDAS; do
    name="books_perta_l${lam}_a1.0"
    if is_done "$name"; then
        log "[SKIP] $name eval exists"
        report_metrics "$name"
    else
        run_eval "$name"
        report_metrics "$name"
    fi
done

log "=== PerTA sweep complete ==="

# ═══════════════════════════════════════════════════════════════════
# STEP 4: LoRA-BiAL (Ze0 champion config, Books domain)
# ═══════════════════════════════════════════════════════════════════
log "=== Step 4: LoRA-BiAL ==="

run_lora_bial() {
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
            trainer.method_args.fisher_cache_path=${FISHER_CACHE} \
            "${overrides[@]}" \
            2>&1 | tee /tmp/train_books_${name}.log
    fi

    run_eval "$name"
    report_metrics "$name"
    log "[DONE] $name"
}

LORA_COMMON=(
    trainer.args.per_device_train_batch_size=2
    trainer.args.gradient_accumulation_steps=1
    trainer.args.num_train_epochs=25
    trainer.method_args.lora_r=16
    trainer.method_args.lora_alpha=32
    trainer.method_args.forget_loss_type=npo
    trainer.method_args.T=25
    trainer.method_args.K=3
    trainer.method_args.npo_beta=4.0
    trainer.method_args.epsilon=0.70
    trainer.method_args.rho=0.1
    trainer.method_args.lambda_init=1.0
    trainer.method_args.eta_theta=3e-5
    trainer.method_args.eta_in=2e-4
)

# V0: exact Ze0 config with lambda=3.5
run_lora_bial "books_lora_bial_v0" \
    "${LORA_COMMON[@]}" \
    trainer.method_args.perta_lambda=3.5

# V1: lambda=3.0 (in case Books needs less aggressive PerTA)
run_lora_bial "books_lora_bial_v1_l30" \
    "${LORA_COMMON[@]}" \
    trainer.method_args.perta_lambda=3.0

# V2: lambda=2.5 (even gentler)
run_lora_bial "books_lora_bial_v2_l25" \
    "${LORA_COMMON[@]}" \
    trainer.method_args.perta_lambda=2.5

log "=== LoRA-BiAL complete ==="

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
log "=== MUSE Books — Final Results ==="

log "--- Gold ---"
log "Retrain:"; report_gold "retrain" "$RETRAIN_JSON"
log "Target:"; report_gold "target" "$TARGET_JSON"
log "Pretrained:"; report_gold "pretrained" "$PRETRAINED_JSON"

log "--- Gradient Baselines ---"
for name in books_gradasc books_graddiff books_npo books_simnpo books_rmu books_blurnpo; do
    log "$name:"; report_metrics "$name"
done

log "--- PerTA ---"
for lam in $PERTA_LAMBDAS; do
    name="books_perta_l${lam}_a1.0"
    log "$name:"; report_metrics "$name"
done

log "--- LoRA-BiAL ---"
for name in books_lora_bial_v0 books_lora_bial_v1_l30 books_lora_bial_v2_l25; do
    log "$name:"; report_metrics "$name"
done

log "=== MUSE Books benchmark complete ==="
