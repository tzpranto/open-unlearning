#!/bin/bash
# Seed Stability Test: Does Ze0's n=64 Fisher sampling matter?
# Compute 3 Fisher caches with different random subsets of 64 samples,
# apply PerTA + run Ze0 bilevel for each, compare deltas.
#
# Original Ze0 used first 64 samples (no shuffle) → _perta_fisher_cache_News_n64.pt
# This test shuffles with seeds {42, 123, 456} to check variance.
set -e
cd /datadrive/forked/open-unlearning

SPLIT="News"
RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_${SPLIT}_retrain/MUSE_EVAL.json"

eval_and_print() {
    local TASK=$1
    local LABEL=$2
    local MODEL_PATH=$3

    if [ ! -f "${MODEL_PATH}/evals/MUSE_EVAL.json" ]; then
        python src/eval.py \
            experiment=eval/muse/default.yaml \
            data_split="$SPLIT" \
            task_name="${TASK}_eval" \
            model=Llama-2-7b-hf \
            model.model_args.pretrained_model_name_or_path="$MODEL_PATH" \
            model.model_args.attn_implementation=sdpa \
            paths.output_dir="${MODEL_PATH}/evals" \
            retain_logs_path="$RETAIN_LOGS"
    fi
    python -c "
import json
with open('${MODEL_PATH}/evals/MUSE_EVAL.json') as f:
    d = json.load(f)
fk = d['forget_knowmem_ROUGE']['agg_value']
rk = d['retain_knowmem_ROUGE']['agg_value']
delta = rk - (0.55*fk + 0.17)
print(f'  ${LABEL}: fk={fk:.4f}, rk={rk:.4f}, delta={delta:+.3f}')
"
}

# Step 1: Compute 3 Fisher caches with different seeds
echo "============================================="
echo "Step 1: Computing Fisher caches (3 seeds)"
echo "============================================="

for SEED in 42 123 456; do
    CACHE="saves/unlearn/_perta_fisher_cache_News_n64_seed${SEED}.pt"
    if [ -f "$CACHE" ]; then
        echo "  Seed $SEED: cache exists, skipping"
    else
        echo "  Seed $SEED: computing Fisher (n=64, shuffled)..."
        python scripts/perta_unlearn.py \
            --data_split News \
            --n_samples 64 \
            --seed "$SEED" \
            --fisher_cache "$CACHE" \
            --lambdas 3.5 \
            --skip_after_fisher
    fi
done

# Step 2: Apply PerTA + run Ze0 bilevel for each seed
echo ""
echo "============================================="
echo "Step 2: PerTA + Ze0 bilevel for each seed"
echo "============================================="

for SEED in 42 123 456; do
    CACHE="saves/unlearn/_perta_fisher_cache_News_n64_seed${SEED}.pt"
    TASK="Ze0_seed${SEED}"

    echo ""
    echo "--- Seed $SEED ---"

    # Skip if already done
    if [ -f "saves/unlearn/${TASK}/config.json" ]; then
        echo "  Already done, skipping training."
    else
        python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/lora_bial \
            task_name="$TASK" \
            trainer.method_args.fisher_cache_path="$CACHE" \
            trainer.method_args.perta_lambda=3.5 \
            trainer.method_args.T=25 \
            trainer.method_args.K=3 \
            trainer.method_args.eta_theta=3e-5 \
            trainer.method_args.eta_in=2e-4 \
            trainer.method_args.npo_beta=4.0 \
            trainer.method_args.lora_r=16 \
            trainer.method_args.lora_alpha=32 \
            trainer.method_args.epsilon=0.70 \
            trainer.method_args.rho=0.1 \
            trainer.method_args.lambda_init=1.0 \
            trainer.method_args.lambda_max=0.0
    fi

    eval_and_print "$TASK" "seed-${SEED}" "saves/unlearn/${TASK}"
done

# Also eval original Ze0 for comparison
echo ""
echo "--- Original Ze0 (first 64, no shuffle) ---"
eval_and_print "Ze0_olr3e5" "original" "saves/unlearn/Ze0_olr3e5"

echo ""
echo "============================================="
echo "Summary"
echo "============================================="
echo "If deltas are within ±0.010: sampling is stable, n=64 is robust"
echo "If deltas vary >0.020: sampling matters, need diversity strategy"
echo ""
echo "References:"
echo "  Ze0 original:  fk=0.289, rk=0.449, delta=+0.120"
echo "  PerTA (full):  fk=0.376, rk=0.416, delta=+0.039"
echo "  Gold:          fk=0.324, rk=0.552"
echo "Done."
