#!/bin/bash
# LoRA-BiAL 1-epoch with full per-sample Fisher PerTA init
# Uses Ze0-winning hyperparams (K=3, β=4, olr=3e-5) with epoch-based training
set -e
cd /datadrive/forked/open-unlearning

TASK="lora_bial_1ep_v0"
SPLIT="News"
RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_${SPLIT}_retrain/MUSE_EVAL.json"

echo "============================================="
echo "LoRA-BiAL 1-epoch: ${TASK}"
echo "============================================="

# Train
python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/muse/lora_bial \
    task_name="$TASK" \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.T=-1 \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.npo_beta=4.0 \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.lr_schedule=cosine \
    trainer.method_args.warmup_fraction=0.05 \
    trainer.method_args.eval_at_steps=[25,50,100,200,300] \
    trainer.method_args.fisher_cache_path=saves/unlearn/_perta_fisher_cache_News_ps_full.pt

echo ""
echo "============================================="
echo "Evaluate final model"
echo "============================================="
python src/eval.py \
    experiment=eval/muse/default.yaml \
    data_split="$SPLIT" \
    task_name="$TASK" \
    model=Llama-2-7b-hf \
    model.model_args.pretrained_model_name_or_path="saves/unlearn/${TASK}" \
    model.model_args.attn_implementation=sdpa \
    paths.output_dir="saves/unlearn/${TASK}/evals" \
    retain_logs_path="$RETAIN_LOGS"

echo ""
echo "=== Final Results ==="
python -c "
import json
with open('saves/unlearn/${TASK}/evals/MUSE_EVAL.json') as f:
    d = json.load(f)
fk = d['forget_knowmem_ROUGE']['agg_value']
rk = d['retain_knowmem_ROUGE']['agg_value']
delta = rk - (0.55*fk + 0.17)
print(f'LoRA-BiAL 1ep: fk={fk:.4f}, rk={rk:.4f}, δ={delta:+.3f}')
for k in sorted(d.keys()):
    if isinstance(d[k], dict) and 'agg_value' in d[k]:
        print(f'  {k}: {d[k][\"agg_value\"]:.4f}')
print()
print('References:')
print('  PerTA init:    fk=0.376, rk=0.416, δ=+0.039')
print('  Ze0 (T=25):    fk=0.289, rk=0.449, δ=+0.120')
print('  Gold:          fk=0.324, rk=0.552')
"

# Evaluate intermediate checkpoints if they exist
echo ""
echo "=== Intermediate Checkpoint Results ==="
for step in 25 50 100 200 300; do
    CKPT="saves/unlearn/${TASK}/step-${step}"
    if [ -d "$CKPT" ]; then
        echo "--- Step ${step} ---"
        EVAL_DIR="${CKPT}/evals"
        if [ ! -f "${EVAL_DIR}/MUSE_EVAL.json" ]; then
            python src/eval.py \
                experiment=eval/muse/default.yaml \
                data_split="$SPLIT" \
                task_name="${TASK}_step${step}" \
                model=Llama-2-7b-hf \
                model.model_args.pretrained_model_name_or_path="$CKPT" \
                model.model_args.attn_implementation=sdpa \
                paths.output_dir="$EVAL_DIR" \
                retain_logs_path="$RETAIN_LOGS"
        fi
        python -c "
import json
with open('${EVAL_DIR}/MUSE_EVAL.json') as f:
    d = json.load(f)
fk = d['forget_knowmem_ROUGE']['agg_value']
rk = d['retain_knowmem_ROUGE']['agg_value']
delta = rk - (0.55*fk + 0.17)
print(f'  Step ${step}: fk={fk:.4f}, rk={rk:.4f}, δ={delta:+.3f}')
"
    fi
done

echo ""
echo "Done."
