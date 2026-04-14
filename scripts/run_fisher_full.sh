#!/bin/bash
# Full Fisher pipeline: aggregate (paper) + per-sample, eval both, then LoRA-BiAL
# Uses ALL forget (889) and ALL retain (1777) samples
set -e
cd /datadrive/forked/open-unlearning

SPLIT="News"
RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_${SPLIT}_retrain/MUSE_EVAL.json"

echo "============================================="
echo "PHASE 1: Aggregate Fisher (paper methodology)"
echo "============================================="
python scripts/perta_fisher_fast.py \
    --mode aggregate \
    --data_split "$SPLIT" \
    --lambdas 3.5 \
    --alpha 1.0

echo ""
echo "============================================="
echo "PHASE 2: Evaluate aggregate PerTA"
echo "============================================="
OUT_AGG="perta_l3.5_agg_full"
if [ ! -f "saves/unlearn/${OUT_AGG}/evals/MUSE_EVAL.json" ]; then
    python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split="$SPLIT" \
        task_name="$OUT_AGG" \
        model=Llama-2-7b-hf \
        model.model_args.pretrained_model_name_or_path="saves/unlearn/${OUT_AGG}" \
        model.model_args.attn_implementation=sdpa \
        paths.output_dir="saves/unlearn/${OUT_AGG}/evals" \
        retain_logs_path="$RETAIN_LOGS"
else
    echo "[SKIP] Eval already exists"
fi

# Extract results
echo ""
echo "=== Aggregate PerTA Results ==="
python -c "
import json
with open('saves/unlearn/${OUT_AGG}/evals/MUSE_EVAL.json') as f:
    d = json.load(f)
fk = d.get('forget_knowmem_ROUGE', {}).get('agg_value', 'N/A')
rk = d.get('retain_knowmem_ROUGE', {}).get('agg_value', 'N/A')
print(f'  fk={fk}, rk={rk}')
if isinstance(fk, (int, float)) and isinstance(rk, (int, float)):
    delta = rk - (0.55*fk + 0.17)
    print(f'  delta={delta:+.3f}')
"

echo ""
echo "============================================="
echo "PHASE 3: Per-sample Fisher (true diagonal)"
echo "============================================="
python scripts/perta_fisher_fast.py \
    --mode per_sample \
    --data_split "$SPLIT" \
    --lambdas 3.5 \
    --alpha 1.0

echo ""
echo "============================================="
echo "PHASE 4: Evaluate per-sample PerTA"
echo "============================================="
OUT_PS="perta_l3.5_ps_full"
if [ ! -f "saves/unlearn/${OUT_PS}/evals/MUSE_EVAL.json" ]; then
    python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split="$SPLIT" \
        task_name="$OUT_PS" \
        model=Llama-2-7b-hf \
        model.model_args.pretrained_model_name_or_path="saves/unlearn/${OUT_PS}" \
        model.model_args.attn_implementation=sdpa \
        paths.output_dir="saves/unlearn/${OUT_PS}/evals" \
        retain_logs_path="$RETAIN_LOGS"
else
    echo "[SKIP] Eval already exists"
fi

echo ""
echo "=== Per-sample PerTA Results ==="
python -c "
import json
with open('saves/unlearn/${OUT_PS}/evals/MUSE_EVAL.json') as f:
    d = json.load(f)
fk = d.get('forget_knowmem_ROUGE', {}).get('agg_value', 'N/A')
rk = d.get('retain_knowmem_ROUGE', {}).get('agg_value', 'N/A')
print(f'  fk={fk}, rk={rk}')
if isinstance(fk, (int, float)) and isinstance(rk, (int, float)):
    delta = rk - (0.55*fk + 0.17)
    print(f'  delta={delta:+.3f}')
"

echo ""
echo "============================================="
echo "PHASE 5: Comparison Summary"
echo "============================================="
python -c "
import json
results = {}
for name in ['${OUT_AGG}', '${OUT_PS}']:
    path = f'saves/unlearn/{name}/evals/MUSE_EVAL.json'
    try:
        with open(path) as f:
            d = json.load(f)
        fk = d['forget_knowmem_ROUGE']['agg_value']
        rk = d['retain_knowmem_ROUGE']['agg_value']
        delta = rk - (0.55*fk + 0.17)
        results[name] = (fk, rk, delta)
        print(f'{name}: fk={fk:.3f}, rk={rk:.3f}, δ={delta:+.3f}')
    except Exception as e:
        print(f'{name}: ERROR - {e}')

# Reference points
print()
print('References:')
print('  Ze0 champion:   fk=0.289, rk=0.449, δ=+0.120')
print('  PerTA n=64:     fk=0.282, rk=0.396, δ=+0.071')
print('  Gold:           fk=0.324, rk=0.552')
print('  Paper (News):   fk=0.385, rk=0.464')
"

echo ""
echo "Fisher + PerTA pipeline complete."
echo "Next: run LoRA-BiAL with the better Fisher variant."
