#!/bin/bash
# Paper-faithful PerTA: Fisher at θ_0 (pretrained), per-sample mode, all samples
# This is for the standalone PerTA baseline comparison
set -e
cd /datadrive/forked/open-unlearning

SPLIT="News"
RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_${SPLIT}_retrain/MUSE_EVAL.json"

echo "============================================="
echo "Paper-faithful PerTA: Fisher at θ_0 (pretrained)"
echo "============================================="
python scripts/perta_fisher_fast.py \
    --mode per_sample \
    --fisher_at_pretrained \
    --data_split "$SPLIT" \
    --lambdas 3.5 \
    --alpha 1.0

echo ""
echo "============================================="
echo "Evaluate paper-faithful PerTA"
echo "============================================="
OUT="perta_l3.5_ps_theta0_full"
python src/eval.py \
    experiment=eval/muse/default.yaml \
    data_split="$SPLIT" \
    task_name="$OUT" \
    model=Llama-2-7b-hf \
    model.model_args.pretrained_model_name_or_path="saves/unlearn/${OUT}" \
    model.model_args.attn_implementation=sdpa \
    paths.output_dir="saves/unlearn/${OUT}/evals" \
    retain_logs_path="$RETAIN_LOGS"

echo ""
echo "=== Paper-faithful PerTA Results ==="
python -c "
import json
with open('saves/unlearn/${OUT}/evals/MUSE_EVAL.json') as f:
    d = json.load(f)
fk = d['forget_knowmem_ROUGE']['agg_value']
rk = d['retain_knowmem_ROUGE']['agg_value']
delta = rk - (0.55*fk + 0.17)
print(f'PerTA (θ_0 Fisher): fk={fk:.4f}, rk={rk:.4f}, δ={delta:+.3f}')
for k in sorted(d.keys()):
    if isinstance(d[k], dict) and 'agg_value' in d[k]:
        print(f'  {k}: {d[k][\"agg_value\"]:.4f}')
print()
print('Comparison:')
print('  θ_target Fisher: fk=0.376, rk=0.416, δ=+0.039')
print('  Paper reported:  fk=0.385, rk=0.464')
print('  Gold:            fk=0.324, rk=0.552')
"
