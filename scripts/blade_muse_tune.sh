#!/bin/bash
# BLADE (LoRA-BiAL-Adaptive) MUSE tuning — higher LR + lambda_max experiments
# Queue after: BLUR Books + PDU Books epoch sweep
# Usage:
#   nohup DATA_SPLIT=News bash scripts/blade_muse_tune.sh > saves/unlearn/blade_muse_tune_news.log 2>&1 &
#   nohup DATA_SPLIT=Books bash scripts/blade_muse_tune.sh > saves/unlearn/blade_muse_tune_books.log 2>&1 &
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

# ── Config ──────────────────────────────────────────────────
DATA_SPLIT="${DATA_SPLIT:-News}"
MODEL="${MODEL:-Llama-2-7b-hf}"
SEED="${SEED:-42}"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"

# ── Split-specific defaults ─────────────────────────────────
if [[ "$DATA_SPLIT" == "Books" ]]; then
    EXP_CONFIG="unlearn/muse/lora_bial_adaptive_books.yaml"
    # Books baseline: eta_theta=3e-5, eps_mul=3.2, T=250
    # Tune: higher LR, lambda_max cap
    ETA_THETA="${ETA_THETA:-5e-5}"
    EPS_MUL="${EPS_MUL:-3.2}"
    T="${T:-250}"
    LAMBDA_MAX="${LAMBDA_MAX:-10.0}"
elif [[ "$DATA_SPLIT" == "News" ]]; then
    EXP_CONFIG="unlearn/muse/lora_bial_adaptive_news.yaml"
    # News baseline: eta_theta=2e-5, eps_mul=1.3, T=150
    # Tune: higher LR, lambda_max cap
    ETA_THETA="${ETA_THETA:-4e-5}"
    EPS_MUL="${EPS_MUL:-1.3}"
    T="${T:-150}"
    LAMBDA_MAX="${LAMBDA_MAX:-10.0}"
else
    echo "Invalid DATA_SPLIT: $DATA_SPLIT (must be Books or News)"
    exit 1
fi

TASK="muse_${MODEL}_${DATA_SPLIT}_BLADE_tune_lr${ETA_THETA}_lmax${LAMBDA_MAX}_s${SEED}"
OUTDIR="saves/unlearn/${TASK}"

# ── Skip if done ────────────────────────────────────────────
if [[ -f "$OUTDIR/evals/MUSE_SUMMARY.json" ]]; then
    echo "[SKIP] $TASK — already evaluated"
    cat "$OUTDIR/evals/MUSE_SUMMARY.json"
    exit 0
fi

# ── Train ───────────────────────────────────────────────────
echo "============================================================"
echo "[TRAIN] $(date) $TASK"
echo "  DATA_SPLIT=$DATA_SPLIT  eta_theta=$ETA_THETA  eps_mul=$EPS_MUL  T=$T  lambda_max=$LAMBDA_MAX"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
    experiment=${EXP_CONFIG} \
    data_split=${DATA_SPLIT} \
    task_name=${TASK} \
    retain_logs_path=${RETAIN_LOGS} \
    trainer.method_args.eta_theta=${ETA_THETA} \
    trainer.method_args.epsilon_multiplier=${EPS_MUL} \
    trainer.method_args.T=${T} \
    trainer.method_args.lambda_max=${LAMBDA_MAX} \
    trainer.args.seed=${SEED}

# ── Eval ────────────────────────────────────────────────────
echo "[EVAL] $(date) $TASK"
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    experiment=eval/muse/default.yaml \
    data_split=${DATA_SPLIT} \
    task_name=${TASK} \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${OUTDIR} \
    paths.output_dir=${OUTDIR}/evals \
    retain_logs_path=${RETAIN_LOGS}

echo "[DONE] $(date) $TASK"
cat "$OUTDIR/evals/MUSE_SUMMARY.json" 2>/dev/null || true

# ── Compute HM ──────────────────────────────────────────────
python3 -c "
import json
with open('$OUTDIR/evals/MUSE_SUMMARY.json') as f:
    d = json.load(f)
fk = d['forget_knowmem_ROUGE']
fv = d['forget_verbmem_ROUGE']
rk = d['retain_knowmem_ROUGE']
a, b, c = 1-fk, 1-fv, rk
hm = 3/(1/max(a,1e-9)+1/max(b,1e-9)+1/max(c,1e-9)) if min(a,b,c)>0 else 0
print(f'HM={hm:.4f}  fgt_know={fk:.4f}  fgt_verb={fv:.4f}  ret_know={rk:.4f}')
"

# ── Clean weights ───────────────────────────────────────────
rm -rf "$OUTDIR"/checkpoint-* 2>/dev/null
rm -f "$OUTDIR"/model*.safetensors "$OUTDIR"/model.safetensors.index.json 2>/dev/null
rm -f "$OUTDIR"/pytorch_model* "$OUTDIR"/config.json "$OUTDIR"/generation_config* 2>/dev/null
rm -f "$OUTDIR"/tokenizer* "$OUTDIR"/special_tokens* "$OUTDIR"/added_tokens* 2>/dev/null
rm -f "$OUTDIR"/optimizer* "$OUTDIR"/scheduler* "$OUTDIR"/training_args* 2>/dev/null
find "$OUTDIR" -maxdepth 1 -name "*.safetensors" -delete 2>/dev/null
find "$OUTDIR" -maxdepth 1 -name "*.bin" -delete 2>/dev/null
echo "[CLEANED] $TASK"
