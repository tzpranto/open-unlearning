#!/bin/bash
# PDU Books — train with save_strategy=epoch, eval each checkpoint, find best HM
# Usage: nohup bash scripts/pdu_books_epoch_sweep.sh > saves/unlearn/pdu_books_sweep.log 2>&1 &
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

MODEL="Llama-2-7b-hf"
DATA_SPLIT="Books"
SEED=42
TASK="muse_${MODEL}_${DATA_SPLIT}_PDU_epoch_sweep_s${SEED}"
OUTDIR="saves/unlearn/${TASK}"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"

# ── Train with save_strategy=epoch ──
echo "[TRAIN] $(date) $TASK"
CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/muse/pdu.yaml \
    data_split=${DATA_SPLIT} \
    task_name=${TASK} \
    retain_logs_path=${RETAIN_LOGS} \
    trainer.method_args.retain_loss_eps=0.1 \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=16 \
    trainer.args.gradient_checkpointing=true \
    trainer.args.eval_strategy=no \
    trainer.args.do_eval=false \
    trainer.args.eval_on_start=false \
    trainer.args.save_strategy=epoch \
    trainer.args.seed=${SEED}

echo "[TRAIN DONE] $(date)"

# ── Eval each checkpoint ──
echo ""
echo "============================================================"
echo "Evaluating checkpoints..."
echo "============================================================"

for ckpt_dir in "$OUTDIR"/checkpoint-*; do
    [ -d "$ckpt_dir" ] || continue
    ckpt_name=$(basename "$ckpt_dir")
    eval_dir="${ckpt_dir}/evals"

    if [[ -f "${eval_dir}/MUSE_SUMMARY.json" ]]; then
        echo "[SKIP EVAL] ${ckpt_name} — already evaluated"
        continue
    fi

    echo "[EVAL] $(date) ${ckpt_name}"
    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${TASK} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${ckpt_dir} \
        paths.output_dir=${eval_dir} \
        retain_logs_path=${RETAIN_LOGS} || { echo "[EVAL FAILED] ${ckpt_name}"; continue; }

    echo "[EVAL DONE] ${ckpt_name}"
    cat "${eval_dir}/MUSE_SUMMARY.json" 2>/dev/null || true
    echo ""
done

# ── Print all results for comparison ──
echo ""
echo "============================================================"
echo "ALL CHECKPOINT RESULTS:"
echo "============================================================"
for ckpt_dir in "$OUTDIR"/checkpoint-*; do
    [ -d "$ckpt_dir" ] || continue
    ckpt_name=$(basename "$ckpt_dir")
    summary="${ckpt_dir}/evals/MUSE_SUMMARY.json"
    if [[ -f "$summary" ]]; then
        echo "--- ${ckpt_name} ---"
        python3 -c "
import json, sys
with open('$summary') as f:
    d = json.load(f)
fk = d.get('forget_knowmem_ROUGE', 0)
fv = d.get('forget_verbmem_ROUGE', 0)
rk = d.get('retain_knowmem_ROUGE', 0)
a = 1 - fk; b = 1 - fv; c = rk
hm = 3 / (1/max(a,1e-9) + 1/max(b,1e-9) + 1/max(c,1e-9)) if min(a,b,c) > 0 else 0
print(f'  fgt_know={fk:.4f}  fgt_verb={fv:.4f}  ret_know={rk:.4f}  HM={hm:.4f}')
"
    fi
done

echo ""
echo "[ALL DONE] $(date)"
