#!/bin/bash
# MUSE Books: GradDiff baseline
# Restart-safe: skips if final model exists, resumes from last checkpoint
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="/datadrive/forked/open-unlearning"
PYTHON="${CONDA_PREFIX:-/datadrive/conda/envs/unlearning}/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="Books"
TASK="muse_${MODEL}_${DATA_SPLIT}_GradDiff"
TASK_DIR="saves/unlearn/${TASK}"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
PROGRESS="/tmp/books_graddiff_progress.log"

cd "$BASE"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

# ─── Train ────────────────────────────────────────────────────────────────────
if compgen -G "${TASK_DIR}/model-*.safetensors" > /dev/null 2>&1; then
    log "[SKIP] ${TASK}: model already exists"
else
    RESUME_ARG=()
    LAST_CKPT=$(ls -d "${TASK_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$LAST_CKPT" ]; then
        RESUME_ARG=("+trainer.args.resume_from_checkpoint=${LAST_CKPT}")
        log "[RESUME] ${TASK} from ${LAST_CKPT}"
    else
        log "[TRAIN] ${TASK}: fresh (bs=16 accum=2 eff_bs=32)"
    fi

    "${PYTHON}" src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default \
        trainer=GradDiff \
        model=${MODEL} \
        data_split=${DATA_SPLIT} \
        task_name=${TASK} \
        retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=16 \
        trainer.args.gradient_accumulation_steps=2 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.save_strategy=steps \
        +trainer.args.save_steps=100 \
        +trainer.args.save_total_limit=2 \
        model.model_args.attn_implementation=sdpa \
        "${RESUME_ARG[@]}" \
        2>&1 | tee /tmp/train_${TASK}.log
fi

# ─── Eval ─────────────────────────────────────────────────────────────────────
log "[EVAL] ${TASK}"
"${PYTHON}" src/eval.py \
    experiment=eval/muse/default.yaml \
    data_split=${DATA_SPLIT} \
    task_name=${TASK} \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK} \
    model.model_args.attn_implementation=sdpa \
    paths.output_dir=saves/unlearn/${TASK}/evals \
    retain_logs_path=${RETAIN_LOGS} \
    2>&1 | tee /tmp/eval_${TASK}.log

log "[DONE] ${TASK}"
