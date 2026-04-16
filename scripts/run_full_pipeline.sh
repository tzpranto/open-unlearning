#!/bin/bash
# Full pipeline: PerTA eval (whole-sample Fisher) → LoRA-BiAL with intermediate evals
# Run this after Fisher n=889 cache is ready
set -e
cd /datadrive/forked/open-unlearning

RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json"
FISHER_N889="saves/unlearn/_perta_fisher_cache_News_n889.pt"
FISHER_N64="saves/unlearn/_perta_fisher_cache_News_n64.pt"

eval_model() {
    local name=$1
    local model_dir=$2
    local eval_file="$model_dir/evals/MUSE_EVAL.json"
    if [ -f "$eval_file" ]; then
        echo "[SKIP] Eval exists: $name"
    else
        echo "=== Evaluating: $name ==="
        mkdir -p "$model_dir/evals"
        python src/eval.py experiment=eval/muse/default.yaml \
            data_split=News task_name=$name model=Llama-2-7b-hf \
            "model.model_args.pretrained_model_name_or_path=$model_dir" \
            model.model_args.attn_implementation=sdpa \
            "paths.output_dir=$model_dir/evals" \
            "retain_logs_path=$RETAIN_LOGS"
    fi
}

print_result() {
    local name=$1
    local eval_file=$2
    if [ -f "$eval_file" ]; then
        python -c "
import json
with open('$eval_file') as f:
    d = json.load(f)
fk = d.get('forget_knowmem_ROUGE',{}).get('agg_value',0)
rk = d.get('retain_knowmem_ROUGE',{}).get('agg_value',0)
vm = d.get('forget_verbmem_ROUGE',{}).get('agg_value',0)
delta = rk - (0.55*fk + 0.17)
print(f'$name: fk={fk:.4f} rk={rk:.4f} vm={vm:.4f} delta={delta:+.4f}')
"
    fi
}

echo "============================================"
echo "Phase 1: PerTA with whole-sample Fisher"
echo "============================================"

# Step 1a: Create PerTA model with n=889 Fisher
MODEL_N889="saves/unlearn/perta_l3.5_n889"
if [ -d "$MODEL_N889" ] && ls "$MODEL_N889"/model-*.safetensors >/dev/null 2>&1; then
    echo "[SKIP] Model exists: $MODEL_N889"
else
    echo "Applying PerTA λ=3.5 with whole-sample Fisher (n=889)..."
    python -c "
import sys, os
sys.path.insert(0, 'scripts')
from perta_unlearn import apply_perta, save_model
import torch
from transformers import AutoModelForCausalLM

cache = torch.load('$FISHER_N889', map_location='cpu', weights_only=True)
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
save_model(result, '$MODEL_N889', pre_dir)
"
fi

# Step 1b: Evaluate n=889 PerTA
eval_model "perta_l3.5_n889" "$MODEL_N889"

echo ""
echo "=== PerTA Fisher comparison ==="
print_result "n=64 (current)" "saves/unlearn/perta_l3.5_a1.0/evals/MUSE_EVAL.json"
print_result "n=889 (whole)" "$MODEL_N889/evals/MUSE_EVAL.json"
echo "Ze0 reference: fk=0.2893 rk=0.4494 delta=+0.120"
echo ""

echo "============================================"
echo "Phase 2: LoRA-BiAL 1-epoch with intermediate evals"
echo "============================================"

# Decide which Fisher to use based on Phase 1 results
# For now, use n=889 Fisher if it exists, otherwise n=64
if [ -f "$FISHER_N889" ]; then
    FISHER="$FISHER_N889"
    echo "Using whole-sample Fisher (n=889)"
else
    FISHER="$FISHER_N64"
    echo "Using n=64 Fisher (fallback)"
fi

# E0: Sanity check — T=25 with new code (should match Ze0)
echo ""
echo "--- E0: T=25 sanity check ---"
EXP_E0="epoch_E0_T25"
if [ -f "saves/unlearn/$EXP_E0/evals/MUSE_EVAL.json" ]; then
    echo "[SKIP] E0 already done"
else
    python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/lora_bial \
        data_split=News \
        task_name=$EXP_E0 \
        trainer.method_args.fisher_cache_path=$FISHER \
        trainer.method_args.perta_lambda=3.5 \
        trainer.method_args.npo_beta=4.0 \
        trainer.method_args.K=3 \
        trainer.method_args.T=25 \
        trainer.method_args.eta_theta=3e-5 \
        trainer.method_args.eta_in=2e-4 \
        trainer.method_args.epsilon=0.70 \
        trainer.method_args.rho=0.1 \
        trainer.method_args.lambda_init=1.0 \
        trainer.args.num_train_epochs=1
    eval_model "$EXP_E0" "saves/unlearn/$EXP_E0"
fi
echo "--- E0 result ---"
print_result "E0 T=25" "saves/unlearn/$EXP_E0/evals/MUSE_EVAL.json"
echo "--- E0 training history ---"
python scripts/analyze_history.py "saves/unlearn/$EXP_E0/lora_bial_history.json" 2>/dev/null || true
echo ""

# E1: 1 epoch, cosine LR, with intermediate evals at key steps
echo "--- E1: 1 epoch, cosine LR, intermediate evals ---"
EXP_E1="epoch_E1_1ep_cosine"
if [ -f "saves/unlearn/$EXP_E1/evals/MUSE_EVAL.json" ]; then
    echo "[SKIP] E1 already done"
else
    python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/lora_bial \
        data_split=News \
        task_name=$EXP_E1 \
        trainer.method_args.fisher_cache_path=$FISHER \
        trainer.method_args.perta_lambda=3.5 \
        trainer.method_args.npo_beta=4.0 \
        trainer.method_args.K=3 \
        "trainer.method_args.T=-1" \
        trainer.method_args.eta_theta=3e-5 \
        trainer.method_args.eta_in=2e-4 \
        trainer.method_args.epsilon=0.70 \
        trainer.method_args.rho=0.1 \
        trainer.method_args.lambda_init=1.0 \
        trainer.method_args.lr_schedule=cosine \
        "trainer.method_args.eval_at_steps=[25,50,100,200,300]" \
        trainer.args.num_train_epochs=1

    # Evaluate final model
    eval_model "$EXP_E1" "saves/unlearn/$EXP_E1"

    # Evaluate intermediate checkpoints
    for step in 25 50 100 200 300; do
        ckpt="saves/unlearn/$EXP_E1/step-$step"
        if [ -d "$ckpt" ]; then
            eval_model "${EXP_E1}_step${step}" "$ckpt"
        fi
    done
fi

echo ""
echo "--- E1 training history ---"
python scripts/analyze_history.py "saves/unlearn/$EXP_E1/lora_bial_history.json" 2>/dev/null || true

echo ""
echo "============================================"
echo "FULL RESULTS"
echo "============================================"
echo "Gold:     fk=0.3243 rk=0.5523"
echo "PerTA paper: fk=0.385 rk=0.464 (their PerTA-fisher)"
echo ""
print_result "PerTA n=64" "saves/unlearn/perta_l3.5_a1.0/evals/MUSE_EVAL.json"
print_result "PerTA n=889" "$MODEL_N889/evals/MUSE_EVAL.json"
print_result "Ze0 (T=25,n=64)" "saves/unlearn/Ze0_olr3e5/evals/MUSE_EVAL.json"
print_result "E0 (T=25,new)" "saves/unlearn/$EXP_E0/evals/MUSE_EVAL.json"
for step in 25 50 100 200 300; do
    ckpt="saves/unlearn/$EXP_E1/step-$step"
    print_result "E1 step$step" "$ckpt/evals/MUSE_EVAL.json"
done
print_result "E1 final" "saves/unlearn/$EXP_E1/evals/MUSE_EVAL.json"

echo ""
echo "Done: $(date)"
