#!/bin/bash
# Fisher sample size ablation for MUSE News
# Tests whether n=64 Fisher samples are sufficient, or if more samples improve PerTA
# Computes Fisher caches for n=128, 256, 889(all), applies PerTA lambda=3.5, evaluates each

set -e
cd /datadrive/forked/open-unlearning

RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json"

echo "=========================================="
echo "Fisher Sample Size Ablation — MUSE News"
echo "=========================================="
echo "Start: $(date)"

# Step 1: Compute Fisher caches for different sample sizes
for n in 128 256 889; do
    cache="saves/unlearn/_perta_fisher_cache_News_n${n}.pt"
    if [ -f "$cache" ]; then
        echo "[SKIP] Fisher cache exists: $cache"
        continue
    fi
    echo ""
    echo "=== Computing Fisher cache: n=$n ==="
    echo "Start: $(date)"
    python scripts/perta_unlearn.py \
        --n_samples $n \
        --lambdas 999 \
        --data_split News \
        --fisher_cache "$cache"
    echo "Done: $(date)"
    # The lambda=999 won't produce useful models, but we only need the Fisher cache
    # Clean up any junk model dir
    rm -rf saves/unlearn/perta_l999.0_a1.0
done

# Step 2: Apply PerTA lambda=3.5 with each Fisher cache and evaluate
for n in 64 128 256 889; do
    cache="saves/unlearn/_perta_fisher_cache_News_n${n}.pt"
    model_dir="saves/unlearn/perta_l3.5_n${n}"

    if [ ! -f "$cache" ]; then
        echo "[ERROR] Fisher cache not found: $cache"
        continue
    fi

    # Generate PerTA model if needed
    if [ -d "$model_dir" ] && ls "$model_dir"/model-*.safetensors >/dev/null 2>&1; then
        echo "[SKIP] Model exists: $model_dir"
    else
        echo ""
        echo "=== Applying PerTA lambda=3.5 with Fisher n=$n ==="
        python -c "
import sys, os
sys.path.insert(0, 'scripts')
from perta_unlearn import apply_perta, save_model
import torch
from transformers import AutoModelForCausalLM

cache = torch.load('$cache', map_location='cpu', weights_only=True)
ff, fr = cache['forget'], cache['retain']

target = AutoModelForCausalLM.from_pretrained('muse-bench/MUSE-News_target', torch_dtype=torch.bfloat16, device_map='cpu')
target_sd = {k: v.clone() for k, v in target.state_dict().items()}
config_dir = target.config._name_or_path
if not os.path.isdir(config_dir):
    from huggingface_hub import snapshot_download
    config_dir = snapshot_download('muse-bench/MUSE-News_target', local_files_only=True)
del target

pretrained = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', torch_dtype=torch.bfloat16, device_map='cpu')
pretrained_sd = {k: v.clone() for k, v in pretrained.state_dict().items()}
pre_dir = pretrained.config._name_or_path
if not os.path.isdir(pre_dir):
    from huggingface_hub import snapshot_download
    pre_dir = snapshot_download('meta-llama/Llama-2-7b-hf', local_files_only=True)
del pretrained

result = apply_perta(target_sd, pretrained_sd, ff, fr, lam=3.5, alpha=1.0)
save_model(result, '$model_dir', pre_dir)
"
    fi

    # Evaluate if needed
    eval_file="$model_dir/evals/MUSE_EVAL.json"
    if [ -f "$eval_file" ]; then
        echo "[SKIP] Eval exists: $eval_file"
    else
        echo "=== Evaluating: perta_l3.5_n${n} ==="
        mkdir -p "$model_dir/evals"
        python src/eval.py experiment=eval/muse/default.yaml \
            data_split=News task_name=perta_l3.5_n${n} model=Llama-2-7b-hf \
            "model.model_args.pretrained_model_name_or_path=$model_dir" \
            model.model_args.attn_implementation=sdpa \
            "paths.output_dir=$model_dir/evals" \
            "retain_logs_path=$RETAIN_LOGS"
    fi
done

echo ""
echo "=========================================="
echo "RESULTS — Fisher Sample Size Ablation"
echo "=========================================="
for n in 64 128 256 889; do
    eval_file="saves/unlearn/perta_l3.5_n${n}/evals/MUSE_EVAL.json"
    if [ -f "$eval_file" ]; then
        python -c "
import json
with open('$eval_file') as f:
    d = json.load(f)
fk = d.get('forget_knowmem_ROUGE',{}).get('agg_value',0)
rk = d.get('retain_knowmem_ROUGE',{}).get('agg_value',0)
vm = d.get('forget_verbmem_ROUGE',{}).get('agg_value',0)
delta = rk - (0.55*fk + 0.17)
print(f'n=${n:>3}: fk={fk:.4f} rk={rk:.4f} vm={vm:.4f} delta={delta:+.4f}')
"
    else
        echo "n=${n}: [no eval yet]"
    fi
done
echo ""
echo "Done: $(date)"
