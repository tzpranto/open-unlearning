#!/bin/bash
# LoRA-BiAL Ablation Runner — MUSE Books, Llama-2-7b-hf, single seed
# Usage:
#   nohup bash scripts/ablation_muse_books.sh > saves/unlearn/ablation_muse_books.log 2>&1 &
#   START_FROM=A5 bash scripts/ablation_muse_books.sh   # resume from A5
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

# ── Config ──────────────────────────────────────────────────
SEED=42
START_FROM="${START_FROM:-}"
RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_Books_retrain/MUSE_EVAL.json"

# ── Common Hydra overrides (matches A0) ─────────────────────
BASE_OVERRIDES=(
    experiment=unlearn/muse/lora_bial_adaptive_books.yaml
    retain_logs_path=${RETAIN_LOGS}
    trainer.method_args.T=250
    trainer.method_args.conv_patience=20
    trainer.args.seed=${SEED}
)

# ── Ablation definitions: ID|LABEL|OVERRIDES ────────────────
# A0 = full method (existing: muse_Llama-2-7b-hf_Books_adaptive_T250_s42)
ABLATIONS=(
    "A1|K0_no_inner|trainer.method_args.K=0 trainer.method_args.inner_warmup_steps=999999"
    "A4|GA_forget|trainer.method_args.forget_loss_type=ga"
    "A5|NPO_forget|trainer.method_args.forget_loss_type=npo trainer.method_args.npo_beta=4.0"
    "A9|tau_1.0_unclamped|trainer.method_args.clamped_entropy_tau=1.0"
    "A10|fixed_lambda_1.0|trainer.method_args.lambda_init=1.0 trainer.method_args.lambda_min=1.0 trainer.method_args.lambda_max=1.0"
    "A12|rho_0_no_penalty|trainer.method_args.rho=0.0"
    "A14|symmetric_dual|trainer.method_args.dual_decay_factor=1.0"
)

# ── Runner ──────────────────────────────────────────────────
TOTAL=0
DONE=0
SKIP=0
FAIL=0
STARTED=false

if [[ -z "$START_FROM" ]]; then
    STARTED=true
fi

for entry in "${ABLATIONS[@]}"; do
    IFS='|' read -r aid label overrides <<< "$entry"

    # Resume support
    if [[ "$STARTED" == "false" ]]; then
        if [[ "$aid" == "$START_FROM" ]]; then
            STARTED=true
        else
            continue
        fi
    fi

    TOTAL=$((TOTAL+1))
    TASK="ablation_muse_books_${aid}_${label}_s${SEED}"
    OUTDIR="saves/unlearn/${TASK}"
    EVAL_DIR="${OUTDIR}/evals"

    # Skip if already evaluated (check both inline and explicit eval dirs)
    if [[ -f "${EVAL_DIR}/MUSE_SUMMARY.json" ]] || [[ -f "${OUTDIR}/checkpoint-0/evals/MUSE_SUMMARY.json" ]]; then
        echo "[SKIP] ${TASK}"
        SKIP=$((SKIP+1))
        continue
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "[TRAIN] $(date) ${TASK}"
    echo "  ${aid} | ${overrides}"
    echo "════════════════════════════════════════════════════════════"

    # Build override array
    IFS=' ' read -ra EXTRA <<< "$overrides"

    # Train
    if ! CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
        "${BASE_OVERRIDES[@]}" \
        task_name=${TASK} \
        "${EXTRA[@]}"; then
        echo "[TRAIN FAILED] ${TASK}"
        FAIL=$((FAIL+1))
        continue
    fi

    # Inline eval already ran during training — copy results to standard location
    if [[ -f "${OUTDIR}/checkpoint-0/evals/MUSE_SUMMARY.json" ]]; then
        mkdir -p "${EVAL_DIR}"
        cp "${OUTDIR}/checkpoint-0/evals/"* "${EVAL_DIR}/" 2>/dev/null
    fi

    echo "[DONE] $(date) ${TASK}"
    cat "${EVAL_DIR}/MUSE_SUMMARY.json" 2>/dev/null || true

    # Compute HM
    python3 -c "
import json
from scipy.stats import hmean
with open('${EVAL_DIR}/MUSE_SUMMARY.json') as f:
    d = json.load(f)
fk = d['forget_knowmem_ROUGE']
vm = d['forget_verbmem_ROUGE']
rk = d['retain_knowmem_ROUGE']
vals = [max(1e-6, 1-fk), max(1e-6, 1-vm), max(1e-6, rk)]
hm = hmean(vals)
print(f'HM={hm:.4f}  fgt_know={fk:.4f}  fgt_verb={vm:.4f}  ret_know={rk:.4f}')
" 2>/dev/null || true

    # Clean weights to save space
    rm -f "$OUTDIR"/model*.safetensors "$OUTDIR"/model.safetensors.index.json 2>/dev/null
    rm -f "$OUTDIR"/pytorch_model* "$OUTDIR"/config.json "$OUTDIR"/generation_config* 2>/dev/null
    rm -f "$OUTDIR"/tokenizer* "$OUTDIR"/special_tokens* "$OUTDIR"/added_tokens* 2>/dev/null
    rm -f "$OUTDIR"/optimizer* "$OUTDIR"/scheduler* "$OUTDIR"/training_args* 2>/dev/null
    rm -rf "$OUTDIR"/checkpoint-* 2>/dev/null
    find "$OUTDIR" -maxdepth 1 -name "*.safetensors" -delete 2>/dev/null
    find "$OUTDIR" -maxdepth 1 -name "*.bin" -delete 2>/dev/null
    echo "[CLEANED] ${TASK}"

    DONE=$((DONE+1))
done

echo ""
echo "════════════════════════════════════════"
echo "ALL DONE: total=${TOTAL} done=${DONE} skip=${SKIP} fail=${FAIL}"
echo "════════════════════════════════════════"

# ── Summary table ───────────────────────────────────────────
echo ""
echo "ABLATION RESULTS (MUSE Books):"
echo "ID | Label | fgt_know↓ | fgt_verb↓ | ret_know↑ | HM↑"
echo "---|-------|-----------|-----------|-----------|----"

# A0 reference
python3 -c "
import json
from scipy.stats import hmean
with open('saves/unlearn/muse_Llama-2-7b-hf_Books_adaptive_T250_s42/evals/MUSE_SUMMARY.json') as f:
    d = json.load(f)
fk = d['forget_knowmem_ROUGE']
vm = d['forget_verbmem_ROUGE']
rk = d['retain_knowmem_ROUGE']
hm = hmean([max(1e-6,1-fk), max(1e-6,1-vm), max(1e-6,rk)])
print(f'A0 | full_method | {fk:.4f} | {vm:.4f} | {rk:.4f} | {hm:.4f}')
" 2>/dev/null || true

for entry in "${ABLATIONS[@]}"; do
    IFS='|' read -r aid label overrides <<< "$entry"
    TASK="ablation_muse_books_${aid}_${label}_s${SEED}"
    summary="saves/unlearn/${TASK}/evals/MUSE_SUMMARY.json"
    if [[ -f "$summary" ]]; then
        python3 -c "
import json
from scipy.stats import hmean
with open('$summary') as f:
    d = json.load(f)
fk = d['forget_knowmem_ROUGE']
vm = d['forget_verbmem_ROUGE']
rk = d['retain_knowmem_ROUGE']
hm = hmean([max(1e-6,1-fk), max(1e-6,1-vm), max(1e-6,rk)])
print(f'${aid} | ${label} | {fk:.4f} | {vm:.4f} | {rk:.4f} | {hm:.4f}')
" 2>/dev/null || true
    fi
done
