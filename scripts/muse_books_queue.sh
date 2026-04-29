#!/bin/bash
# MUSE Books queue: BLUR-NPO → PDU epoch sweep → BLADE v4
# Usage: nohup bash scripts/muse_books_queue.sh > saves/unlearn/muse_books_queue.log 2>&1 &
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

MODEL="Llama-2-7b-hf"
DATA_SPLIT="Books"
SEED=42
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"

TOTAL=3
DONE=0
SKIP=0
MISSING=0

hm_calc() {
    python3 -c "
import json, sys
with open('$1') as f:
    d = json.load(f)
fk = d.get('forget_knowmem_ROUGE', 0)
fv = d.get('forget_verbmem_ROUGE', 0)
rk = d.get('retain_knowmem_ROUGE', 0)
a = 1 - fk; b = 1 - fv; c = rk
hm = 3 / (1/max(a,1e-9) + 1/max(b,1e-9) + 1/max(c,1e-9)) if min(a,b,c) > 0 else 0
print(f'  fgt_know={fk:.4f}  fgt_verb={fv:.4f}  ret_know={rk:.4f}  HM={hm:.4f}')
"
}

# ════════════════════════════════════════════════════════════
# 1. BLUR-NPO Books
# ════════════════════════════════════════════════════════════
TASK1="muse_${MODEL}_${DATA_SPLIT}_BLURNPO_s${SEED}"
OUTDIR1="saves/unlearn/${TASK1}"

if [[ -f "$OUTDIR1/evals/MUSE_SUMMARY.json" ]]; then
    echo "[SKIP] $TASK1 — already evaluated"
    cat "$OUTDIR1/evals/MUSE_SUMMARY.json"
    hm_calc "$OUTDIR1/evals/MUSE_SUMMARY.json"
    SKIP=$((SKIP+1))
else
    echo "============================================================"
    echo "[TRAIN] $(date) $TASK1"
    echo "  beta=0.4  lr=1e-5  (Books defaults from Table 6)"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/blurnpo_muse.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${TASK1} \
        retain_logs_path=${RETAIN_LOGS} \
        trainer.method_args.beta=0.4 \
        trainer.args.learning_rate=1e-5 \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.seed=${SEED}

    echo "[EVAL] $(date) $TASK1"
    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${TASK1} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${OUTDIR1} \
        paths.output_dir=${OUTDIR1}/evals \
        retain_logs_path=${RETAIN_LOGS}

    echo "[DONE] $(date) $TASK1"
    cat "$OUTDIR1/evals/MUSE_SUMMARY.json" 2>/dev/null || true
    hm_calc "$OUTDIR1/evals/MUSE_SUMMARY.json" 2>/dev/null || true
    DONE=$((DONE+1))
fi

# ════════════════════════════════════════════════════════════
# 2. PDU Books (epoch sweep)
# ════════════════════════════════════════════════════════════
TASK2="muse_${MODEL}_${DATA_SPLIT}_PDU_epoch_sweep_s${SEED}"
OUTDIR2="saves/unlearn/${TASK2}"

if [[ -f "$OUTDIR2/evals/MUSE_SUMMARY.json" ]] || ls "$OUTDIR2"/checkpoint-*/evals/MUSE_SUMMARY.json &>/dev/null; then
    echo "[SKIP] $TASK2 — already evaluated"
    SKIP=$((SKIP+1))
else
    echo ""
    echo "============================================================"
    echo "[TRAIN] $(date) $TASK2"
    echo "  eps=0.1  save_strategy=epoch (Books)"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/pdu.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${TASK2} \
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

    echo "[TRAIN DONE] $(date) $TASK2"

    # Eval each checkpoint
    echo ""
    echo "Evaluating checkpoints..."
    for ckpt_dir in "$OUTDIR2"/checkpoint-*; do
        [ -d "$ckpt_dir" ] || continue
        ckpt_name=$(basename "$ckpt_dir")
        eval_dir="${ckpt_dir}/evals"

        if [[ -f "${eval_dir}/MUSE_SUMMARY.json" ]]; then
            echo "[SKIP EVAL] ${ckpt_name}"
            continue
        fi

        echo "[EVAL] $(date) ${ckpt_name}"
        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/muse/default.yaml \
            data_split=${DATA_SPLIT} \
            task_name=${TASK2} \
            model=${MODEL} \
            model.model_args.pretrained_model_name_or_path=${ckpt_dir} \
            paths.output_dir=${eval_dir} \
            retain_logs_path=${RETAIN_LOGS} || { echo "[EVAL FAILED] ${ckpt_name}"; continue; }

        echo "[EVAL DONE] ${ckpt_name}"
        cat "${eval_dir}/MUSE_SUMMARY.json" 2>/dev/null || true
        hm_calc "${eval_dir}/MUSE_SUMMARY.json" 2>/dev/null || true
        echo ""
    done

    # Summary
    echo ""
    echo "ALL CHECKPOINT RESULTS:"
    for ckpt_dir in "$OUTDIR2"/checkpoint-*; do
        [ -d "$ckpt_dir" ] || continue
        ckpt_name=$(basename "$ckpt_dir")
        summary="${ckpt_dir}/evals/MUSE_SUMMARY.json"
        if [[ -f "$summary" ]]; then
            echo "--- ${ckpt_name} ---"
            hm_calc "$summary"
        fi
    done
    DONE=$((DONE+1))
fi

# ════════════════════════════════════════════════════════════
# 3. BLADE v4 (adaptive Books — higher LR + lambda_max)
# ════════════════════════════════════════════════════════════
TASK3="muse_${MODEL}_${DATA_SPLIT}_BLADE_v4_s${SEED}"
OUTDIR3="saves/unlearn/${TASK3}"

if [[ -f "$OUTDIR3/evals/MUSE_SUMMARY.json" ]]; then
    echo "[SKIP] $TASK3 — already evaluated"
    cat "$OUTDIR3/evals/MUSE_SUMMARY.json"
    hm_calc "$OUTDIR3/evals/MUSE_SUMMARY.json"
    SKIP=$((SKIP+1))
else
    echo ""
    echo "============================================================"
    echo "[TRAIN] $(date) $TASK3"
    echo "  BLADE v4: eta_theta=5e-5  eps_mul=3.2  T=250  lambda_max=10.0"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/lora_bial_adaptive_books.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${TASK3} \
        retain_logs_path=${RETAIN_LOGS} \
        trainer.method_args.eta_theta=5e-5 \
        trainer.method_args.epsilon_multiplier=3.2 \
        trainer.method_args.T=250 \
        trainer.method_args.lambda_max=10.0 \
        trainer.args.seed=${SEED}

    echo "[EVAL] $(date) $TASK3"
    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${TASK3} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${OUTDIR3} \
        paths.output_dir=${OUTDIR3}/evals \
        retain_logs_path=${RETAIN_LOGS}

    echo "[DONE] $(date) $TASK3"
    cat "$OUTDIR3/evals/MUSE_SUMMARY.json" 2>/dev/null || true
    hm_calc "$OUTDIR3/evals/MUSE_SUMMARY.json" 2>/dev/null || true
    DONE=$((DONE+1))
fi

echo ""
echo "════════════════════════════════════════"
echo "ALL DONE: total=$TOTAL skip=$SKIP done=$DONE"
echo "════════════════════════════════════════"
