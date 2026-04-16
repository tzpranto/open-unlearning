#!/bin/bash
# LoRA-BiAL Epoch Ablation — MUSE News
# Tests data coverage: from T=25 (5.6% of 1 epoch) to full 1-3 epochs
# All use Ze0 champion hyperparams + existing Fisher n=64 cache
#
# Experiments:
#   E0: T=25 sanity check (should reproduce Ze0: delta=+0.120)
#   E1: 1 epoch, constant LR
#   E2: 1 epoch, cosine LR
#   E3: 1 epoch, cosine LR + retain-only after NPO saturation
#   E4: 3 epochs, cosine LR
#   E5: 3 epochs, cosine LR + retain-only after saturation

set -e
cd /datadrive/forked/open-unlearning

RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json"
FISHER_CACHE="saves/unlearn/_perta_fisher_cache_News_n64.pt"

is_done() { [ -f "saves/unlearn/$1/evals/MUSE_EVAL.json" ]; }
has_model() { ls "saves/unlearn/$1"/model*.safetensors >/dev/null 2>&1 || [ -f "saves/unlearn/$1/config.json" ]; }

run_exp() {
    local name=$1
    shift
    local extra_args="$@"

    echo ""
    echo "============================================"
    echo "Experiment: $name"
    echo "============================================"

    if is_done "$name"; then
        echo "[SKIP] Already has eval: $name"
        return
    fi

    if ! has_model "$name"; then
        echo "Training: $name ($(date))"
        python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/lora_bial \
            data_split=News \
            task_name=$name \
            trainer.method_args.fisher_cache_path=$FISHER_CACHE \
            trainer.method_args.perta_lambda=3.5 \
            trainer.method_args.npo_beta=4.0 \
            trainer.method_args.K=3 \
            trainer.method_args.eta_theta=3e-5 \
            trainer.method_args.eta_in=2e-4 \
            trainer.method_args.epsilon=0.70 \
            trainer.method_args.rho=0.1 \
            trainer.method_args.lambda_init=1.0 \
            $extra_args
        echo "Training done: $(date)"
    else
        echo "[SKIP] Model exists: $name"
    fi

    echo "Evaluating: $name"
    python src/eval.py experiment=eval/muse/default.yaml \
        data_split=News task_name=$name model=Llama-2-7b-hf \
        "model.model_args.pretrained_model_name_or_path=saves/unlearn/$name" \
        model.model_args.attn_implementation=sdpa \
        "paths.output_dir=saves/unlearn/$name/evals" \
        "retain_logs_path=$RETAIN_LOGS"
    echo "Eval done: $(date)"
}

echo "=========================================="
echo "LoRA-BiAL Epoch Ablation — MUSE News"
echo "=========================================="
echo "Start: $(date)"
echo ""

# E0: Sanity check — reproduce Ze0 with new code
run_exp "epoch_E0_T25" \
    "trainer.method_args.T=25" \
    "trainer.args.num_train_epochs=1"

# E1: 1 epoch, constant LR (445 steps with batch=2)
run_exp "epoch_E1_1ep_const" \
    "trainer.method_args.T=-1" \
    "trainer.args.num_train_epochs=1" \
    "trainer.method_args.lr_schedule=constant"

# E2: 1 epoch, cosine LR decay
run_exp "epoch_E2_1ep_cosine" \
    "trainer.method_args.T=-1" \
    "trainer.args.num_train_epochs=1" \
    "trainer.method_args.lr_schedule=cosine"

# E3: 1 epoch, cosine LR + retain-only after NPO saturation
run_exp "epoch_E3_1ep_cos_sat" \
    "trainer.method_args.T=-1" \
    "trainer.args.num_train_epochs=1" \
    "trainer.method_args.lr_schedule=cosine" \
    "trainer.method_args.retain_only_after_saturation=true" \
    "trainer.method_args.saturation_patience=5"

# E4: 3 epochs, cosine LR
run_exp "epoch_E4_3ep_cosine" \
    "trainer.method_args.T=-1" \
    "trainer.args.num_train_epochs=3" \
    "trainer.method_args.lr_schedule=cosine"

# E5: 3 epochs, cosine LR + retain-only after saturation
run_exp "epoch_E5_3ep_cos_sat" \
    "trainer.method_args.T=-1" \
    "trainer.args.num_train_epochs=3" \
    "trainer.method_args.lr_schedule=cosine" \
    "trainer.method_args.retain_only_after_saturation=true" \
    "trainer.method_args.saturation_patience=5"

echo ""
echo "=========================================="
echo "RESULTS — Epoch Ablation"
echo "=========================================="
echo "Reference: Ze0 champion: fk=0.289, rk=0.449, delta=+0.120"
echo ""

for name in epoch_E0_T25 epoch_E1_1ep_const epoch_E2_1ep_cosine epoch_E3_1ep_cos_sat epoch_E4_3ep_cosine epoch_E5_3ep_cos_sat; do
    eval_file="saves/unlearn/$name/evals/MUSE_EVAL.json"
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
    else
        echo "$name: [no eval]"
    fi
done

echo ""
echo "Done: $(date)"
