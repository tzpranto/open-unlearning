#!/bin/bash
set -e
cd /datadrive/forked/open-unlearning

SPLIT="News"
RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_${SPLIT}_retrain/MUSE_EVAL.json"

echo "============================================="
echo "20% Importance-Sampled Fisher + PerTA"
echo "============================================="
python scripts/fisher_importance_from_cache.py \
    --data_split "$SPLIT" \
    --n_select 178 \
    --n_select_retain 356 \
    --lambdas 3.5

echo ""
echo "============================================="
echo "Evaluate importance-sampled PerTA"
echo "============================================="
OUT="perta_l3.5_imp20pct"
if [ ! -f "saves/unlearn/${OUT}/evals/MUSE_EVAL.json" ]; then
    python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split="$SPLIT" \
        task_name="$OUT" \
        model=Llama-2-7b-hf \
        model.model_args.pretrained_model_name_or_path="saves/unlearn/${OUT}" \
        model.model_args.attn_implementation=sdpa \
        paths.output_dir="saves/unlearn/${OUT}/evals" \
        retain_logs_path="$RETAIN_LOGS"
else
    echo "[SKIP] Eval already exists"
fi

echo ""
echo "=== Results ==="
python -c "
import json
with open('saves/unlearn/${OUT}/evals/MUSE_EVAL.json') as f:
    d = json.load(f)
fk = d['forget_knowmem_ROUGE']['agg_value']
rk = d['retain_knowmem_ROUGE']['agg_value']
delta = rk - (0.55*fk + 0.17)
print(f'20% Imp-sampled PerTA: fk={fk:.4f}, rk={rk:.4f}, δ={delta:+.3f}')
for k in sorted(d.keys()):
    if isinstance(d[k], dict) and 'agg_value' in d[k]:
        print(f'  {k}: {d[k][\"agg_value\"]:.4f}')
print()
print('References:')
print('  Full per-sample: fk=0.376, rk=0.416, δ=+0.039')
print('  n=64 PerTA:      fk=0.282, rk=0.396, δ=+0.071')
print('  Ze0 champion:    fk=0.289, rk=0.449, δ=+0.120')
print('  Paper (News):    fk=0.385, rk=0.464')
print('  Gold:            fk=0.324, rk=0.552')
"

echo ""
echo "Done."
