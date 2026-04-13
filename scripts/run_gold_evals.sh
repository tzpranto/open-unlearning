#!/bin/bash
# Evaluate all three reference models on MUSE News to verify gold targets
# 1. Retrain model (gold standard): muse-bench/MUSE-news_retrain
# 2. Target/finetuned model (starting point): muse-bench/MUSE-News_target
# 3. Pretrained model (base): meta-llama/Llama-2-7b-hf
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"

cd /datadrive/forked/open-unlearning

print_all_metrics() {
    local EVAL_FILE=$1
    "${PYTHON_BIN}" -c "
import json
with open('$EVAL_FILE') as f: d = json.load(f)
def agg(d, k):
    v = d.get(k, {})
    return v.get('agg_value', v) if isinstance(v, dict) else v
metrics = [
    'forget_knowmem_ROUGE', 'retain_knowmem_ROUGE',
    'forget_verbmem_ROUGE', 'exact_memorization',
    'extraction_strength', 'privleak',
    'mia_loss', 'mia_gradnorm', 'mia_zlib', 'mia_min_k', 'mia_min_k_plus_plus'
]
print(f'  {\"Metric\":<30} {\"Value\":>10}')
print(f'  {\"-\"*30} {\"-\"*10}')
for m in metrics:
    v = agg(d, m)
    if isinstance(v, (int, float)):
        print(f'  {m:<30} {v:>10.4f}')
    else:
        print(f'  {m:<30} {str(v):>10}')
# Frontier check
fk=float(agg(d,'forget_knowmem_ROUGE')); rk=float(agg(d,'retain_knowmem_ROUGE'))
pred_rk = 0.55*fk + 0.17
delta = rk - pred_rk
print(f'  ---')
print(f'  CE frontier pred rk: {pred_rk:.4f}, actual rk: {rk:.4f}, delta: {delta:+.4f}')
" 2>/dev/null || echo "  [metrics parse failed]"
}

# ── 1. Retrain model (gold standard) ────────────────────────────────────────
echo "=== RETRAIN MODEL (gold standard) ==="
RETRAIN_OUT="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain_FRESH"
rm -rf "$RETRAIN_OUT"
"${PYTHON_BIN}" src/eval.py \
    experiment=eval/muse/default.yaml data_split=${DATA_SPLIT} \
    task_name=retrain_verify model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-news_retrain \
    "${ATTN_ARGS[@]}" \
    paths.output_dir=${RETRAIN_OUT} \
    retain_logs_path=saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json \
    2>&1 | tee "${LOG_DIR}/eval_retrain_verify.log"
echo ""
echo "=== RETRAIN RESULTS ==="
print_all_metrics "${RETRAIN_OUT}/MUSE_EVAL.json"

# ── 2. Target/finetuned model (starting point for unlearning) ────────────────
echo ""
echo "=== TARGET MODEL (finetuned, starting point) ==="
TARGET_OUT="saves/eval/muse_${MODEL}_${DATA_SPLIT}_target"
rm -rf "$TARGET_OUT"
"${PYTHON_BIN}" src/eval.py \
    experiment=eval/muse/default.yaml data_split=${DATA_SPLIT} \
    task_name=target_verify model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-News_target \
    "${ATTN_ARGS[@]}" \
    paths.output_dir=${TARGET_OUT} \
    retain_logs_path=saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json \
    2>&1 | tee "${LOG_DIR}/eval_target_verify.log"
echo ""
echo "=== TARGET RESULTS ==="
print_all_metrics "${TARGET_OUT}/MUSE_EVAL.json"

# ── 3. Pretrained model (base LLM) ──────────────────────────────────────────
echo ""
echo "=== PRETRAINED MODEL (base LLM) ==="
PRETRAINED_OUT="saves/eval/muse_${MODEL}_${DATA_SPLIT}_pretrained"
rm -rf "$PRETRAINED_OUT"
"${PYTHON_BIN}" src/eval.py \
    experiment=eval/muse/default.yaml data_split=${DATA_SPLIT} \
    task_name=pretrained_verify model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=meta-llama/Llama-2-7b-hf \
    "${ATTN_ARGS[@]}" \
    paths.output_dir=${PRETRAINED_OUT} \
    retain_logs_path=saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json \
    2>&1 | tee "${LOG_DIR}/eval_pretrained_verify.log"
echo ""
echo "=== PRETRAINED RESULTS ==="
print_all_metrics "${PRETRAINED_OUT}/MUSE_EVAL.json"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "=== GOLD STANDARD COMPARISON ==="
echo "========================================="
for name_path in "RETRAIN:${RETRAIN_OUT}" "TARGET:${TARGET_OUT}" "PRETRAINED:${PRETRAINED_OUT}"; do
    name="${name_path%%:*}"
    path="${name_path#*:}"
    ef="${path}/MUSE_EVAL.json"
    if [ -f "$ef" ]; then
        "${PYTHON_BIN}" -c "
import json
with open('$ef') as f: d = json.load(f)
def agg(d, k):
    v = d.get(k, {})
    return v.get('agg_value', v) if isinstance(v, dict) else v
fk=float(agg(d,'forget_knowmem_ROUGE')); rk=float(agg(d,'retain_knowmem_ROUGE'))
fv=float(agg(d,'forget_verbmem_ROUGE')); ex=float(agg(d,'extraction_strength'))
print(f'  {\"$name\":<12} fk={fk:.4f}  rk={rk:.4f}  fv={fv:.4f}  ex={ex:.4f}')
" 2>/dev/null
    fi
done
echo ""
echo "=== DONE ==="
