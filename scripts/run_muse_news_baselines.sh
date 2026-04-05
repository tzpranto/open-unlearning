#!/bin/bash
# Run MUSE News Phase 2 (GradAscent, GradDiff, NPO, SimNPO) + DS-BiAL canonical
# All methods use paper-correct configs and checkpoint save/resume
set -euo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG_DIR="${LOG_DIR:-/tmp}"

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
# save_steps/save_total_limit use + (not in finetune.yaml struct)
TRAIN_ARGS=(
    trainer.args.per_device_train_batch_size=1
    trainer.args.gradient_accumulation_steps=16
    trainer.args.ddp_find_unused_parameters=true
    trainer.args.gradient_checkpointing=true
    trainer.args.eval_strategy=no
    trainer.args.do_eval=false
    trainer.args.eval_on_start=false
    trainer.args.save_strategy=steps
    +trainer.args.save_steps=100
    +trainer.args.save_total_limit=2
)
PROGRESS="${LOG_DIR}/muse_news_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting MUSE News Phase2 + DS-BiAL" | tee "$PROGRESS"

run_eval() {
    local TASK=$1
    "${PYTHON_BIN}" src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${TASK} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK} \
        "${ATTN_ARGS[@]}" \
        paths.output_dir=saves/unlearn/${TASK}/evals \
        retain_logs_path=${RETAIN_LOGS} \
        2>&1 | tee /tmp/eval_${TASK}.log
}

# smart_train TASK EXPERIMENT [extra hydra args...]
# EXPERIMENT is passed as-is to experiment= (e.g., unlearn/muse/default.yaml or unlearn/muse/npo.yaml)
smart_train() {
    local TASK=$1
    local EXPERIMENT=$2
    shift 2
    local TASK_DIR="saves/unlearn/${TASK}"

    if compgen -G "${TASK_DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] ${TASK}: already done" | tee -a "$PROGRESS"
        return 0
    fi

    local RESUME_ARG=()
    local LAST_CKPT
    LAST_CKPT=$(ls -d "${TASK_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$LAST_CKPT" ]; then
        echo "[RESUME] ${TASK} from ${LAST_CKPT}" | tee -a "$PROGRESS"
        RESUME_ARG=("+trainer.args.resume_from_checkpoint=${LAST_CKPT}")
    else
        echo "[TRAIN] ${TASK}: fresh" | tee -a "$PROGRESS"
    fi

    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=${EXPERIMENT} model=${MODEL} data_split=${DATA_SPLIT} \
        task_name=${TASK} retain_logs_path=${RETAIN_LOGS} \
        "${TRAIN_ARGS[@]}" "${ATTN_ARGS[@]}" \
        "${RESUME_ARG[@]}" \
        "$@" \
        2>&1 | tee /tmp/train_${TASK}.log
}

# DS-BiAL: skip-if-done only (SIBL has its own internal loop, no HF checkpoint)
smart_train_dsbial() {
    local TASK=$1
    local TASK_DIR="saves/unlearn/${TASK}"

    if compgen -G "${TASK_DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] ${TASK}: already done" | tee -a "$PROGRESS"
        return 0
    fi

    echo "[TRAIN] ${TASK}: fresh DS-BiAL" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/ds_bial model=${MODEL} data_split=${DATA_SPLIT} \
        trainer=SIBL task_name=${TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_accumulation_steps=16 \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        "${ATTN_ARGS[@]}" \
        2>&1 | tee /tmp/train_${TASK}.log
}

# ─── 1. GradAscent ────────────────────────────────────────────────────────────
TASK="muse_${MODEL}_${DATA_SPLIT}_GradAscent"
echo "=== [1/5] GradAscent ===" | tee -a "$PROGRESS"
rm -rf saves/unlearn/${TASK}
if smart_train ${TASK} unlearn/muse/default.yaml trainer=GradAscent; then
    run_eval ${TASK}
    echo "[GradAscent done]" | tee -a "$PROGRESS"
else echo "[WARN] GradAscent failed" | tee -a "$PROGRESS"; fi

# ─── 2. GradDiff ──────────────────────────────────────────────────────────────
TASK="muse_${MODEL}_${DATA_SPLIT}_GradDiff"
echo "=== [2/5] GradDiff ===" | tee -a "$PROGRESS"
rm -rf saves/unlearn/${TASK}
if smart_train ${TASK} unlearn/muse/default.yaml trainer=GradDiff; then
    run_eval ${TASK}
    echo "[GradDiff done]" | tee -a "$PROGRESS"
else echo "[WARN] GradDiff failed" | tee -a "$PROGRESS"; fi

# ─── 3. NPO (paper-correct: beta=0.1, LR=3e-5) ───────────────────────────────
TASK="muse_${MODEL}_${DATA_SPLIT}_NPO"
echo "=== [3/5] NPO ===" | tee -a "$PROGRESS"
rm -rf saves/unlearn/${TASK}
if smart_train ${TASK} unlearn/muse/npo.yaml; then
    run_eval ${TASK}
    echo "[NPO done]" | tee -a "$PROGRESS"
else echo "[WARN] NPO failed" | tee -a "$PROGRESS"; fi

# ─── 4. SimNPO (paper-correct: beta=0.7, LR=1e-5) ───────────────────────────
TASK="muse_${MODEL}_${DATA_SPLIT}_SimNPO"
echo "=== [4/5] SimNPO ===" | tee -a "$PROGRESS"
if smart_train ${TASK} unlearn/muse/simnpo.yaml; then
    run_eval ${TASK}
    echo "[SimNPO done]" | tee -a "$PROGRESS"
else echo "[WARN] SimNPO failed" | tee -a "$PROGRESS"; fi

# ─── 5. DS-BiAL canonical (steering_only=false, NPO+steering) ────────────────
TASK="muse_${MODEL}_${DATA_SPLIT}_DSBiAL"
echo "=== [5/6] DS-BiAL ===" | tee -a "$PROGRESS"
rm -rf saves/unlearn/${TASK}
if smart_train_dsbial ${TASK}; then
    run_eval ${TASK}
    echo "[DS-BiAL done]" | tee -a "$PROGRESS"
else echo "[WARN] DS-BiAL failed" | tee -a "$PROGRESS"; fi

# ─── 6. BLURNPO re-run (paper-correct: beta=0.05, LR=2.5e-5) ─────────────────
TASK="muse_${MODEL}_${DATA_SPLIT}_BLURNPO"
echo "=== [6/7] BLURNPO (paper params) ===" | tee -a "$PROGRESS"
rm -rf saves/unlearn/${TASK}
if smart_train ${TASK} unlearn/muse/blurnpo.yaml; then
    run_eval ${TASK}
    echo "[BLURNPO done]" | tee -a "$PROGRESS"
else echo "[WARN] BLURNPO failed" | tee -a "$PROGRESS"; fi

# ─── 7. RMU re-run (paper-correct: max_steps=80, LR=5e-5, layers 5-7 only) ───
TASK="muse_${MODEL}_${DATA_SPLIT}_RMU"
echo "=== [7/7] RMU (paper params) ===" | tee -a "$PROGRESS"
rm -rf saves/unlearn/${TASK}
if smart_train ${TASK} unlearn/muse/rmu.yaml +trainer.args.max_steps=80; then
    run_eval ${TASK}
    echo "[RMU done]" | tee -a "$PROGRESS"
else echo "[WARN] RMU failed" | tee -a "$PROGRESS"; fi

echo "=== MUSE NEWS FULLY DONE ===" | tee -a "$PROGRESS"
