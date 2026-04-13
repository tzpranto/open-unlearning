#!/bin/bash
# MUSE News baselines: GradAscent, GradDiff, NPO, SimNPO, DS-BiAL, BLURNPO, RMU
# All methods use paper-correct configs + per-method max bs (eff_bs=32)
# Resume logic: skip if final weights exist, resume from last checkpoint otherwise
set -euo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
COMMON_ARGS=(
    trainer.args.gradient_checkpointing=true
    trainer.args.eval_strategy=no
    trainer.args.do_eval=false
    trainer.args.eval_on_start=false
    trainer.args.save_strategy=steps
    +trainer.args.save_steps=100
    +trainer.args.save_total_limit=2
)
PROGRESS="/tmp/news_phase2_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting MUSE News baselines" | tee "$PROGRESS"

run_eval() {
    local TASK=$1
    "${PYTHON_BIN}" src/eval.py \
        experiment=eval/muse/default.yaml data_split=${DATA_SPLIT} \
        task_name=${TASK} model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK} \
        "${ATTN_ARGS[@]}" \
        paths.output_dir=saves/unlearn/${TASK}/evals \
        retain_logs_path=${RETAIN_LOGS} \
        2>&1 | tee /tmp/eval_${TASK}.log
}

# smart_train TASK EXPERIMENT BS ACCUM [extra args...]
smart_train() {
    local TASK=$1 EXPERIMENT=$2 BS=$3 ACCUM=$4
    shift 4
    local TASK_DIR="saves/unlearn/${TASK}"

    if compgen -G "${TASK_DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] ${TASK}: already done" | tee -a "$PROGRESS"; return 0
    fi

    local RESUME_ARG=()
    local LAST_CKPT
    LAST_CKPT=$(ls -d "${TASK_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    [ -n "$LAST_CKPT" ] && RESUME_ARG=("+trainer.args.resume_from_checkpoint=${LAST_CKPT}") \
                        && echo "[RESUME] ${TASK} from ${LAST_CKPT}" | tee -a "$PROGRESS" \
                        || echo "[TRAIN] ${TASK}: fresh (bs=${BS} accum=${ACCUM} eff_bs=$((BS*ACCUM)))" | tee -a "$PROGRESS"

    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=${EXPERIMENT} model=${MODEL} data_split=${DATA_SPLIT} \
        task_name=${TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=${BS} \
        trainer.args.gradient_accumulation_steps=${ACCUM} \
        "${COMMON_ARGS[@]}" "${ATTN_ARGS[@]}" "${RESUME_ARG[@]}" \
        "$@" \
        2>&1 | tee /tmp/train_${TASK}.log
}

# DS-BiAL: skip-if-done only (SIBL has own internal loop, no HF checkpoints)
smart_train_dsbial() {
    local TASK=$1
    local TASK_DIR="saves/unlearn/${TASK}"

    if compgen -G "${TASK_DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] ${TASK}: already done" | tee -a "$PROGRESS"; return 0
    fi

    echo "[TRAIN] ${TASK}: fresh DS-BiAL (bs=1 accum=32 eff_bs=32)" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/ds_bial model=${MODEL} data_split=${DATA_SPLIT} \
        trainer=SIBL task_name=${TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_accumulation_steps=32 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        "${ATTN_ARGS[@]}" \
        2>&1 | tee /tmp/train_${TASK}.log
}

# ─── 1. GradAscent (bs=16, accum=2, eff=32) ──────────────────────────────────
TASK="muse_${MODEL}_${DATA_SPLIT}_GradAscent"
echo "=== [1/7] GradAscent ===" | tee -a "$PROGRESS"
if smart_train ${TASK} unlearn/muse/default.yaml 16 2 trainer=GradAscent; then
    run_eval ${TASK}; echo "[GradAscent done]" | tee -a "$PROGRESS"
else echo "[WARN] GradAscent failed" | tee -a "$PROGRESS"; fi

# ─── 2. GradDiff (bs=16, accum=2, eff=32) ────────────────────────────────────
TASK="muse_${MODEL}_${DATA_SPLIT}_GradDiff"
echo "=== [2/7] GradDiff ===" | tee -a "$PROGRESS"
if smart_train ${TASK} unlearn/muse/default.yaml 16 2 trainer=GradDiff; then
    run_eval ${TASK}; echo "[GradDiff done]" | tee -a "$PROGRESS"
else echo "[WARN] GradDiff failed" | tee -a "$PROGRESS"; fi

# ─── 3. NPO (bs=4, accum=8, eff=32) ─────────────────────────────────────────
TASK="muse_${MODEL}_${DATA_SPLIT}_NPO"
echo "=== [3/7] NPO ===" | tee -a "$PROGRESS"
if smart_train ${TASK} unlearn/muse/npo.yaml 4 8; then
    run_eval ${TASK}; echo "[NPO done]" | tee -a "$PROGRESS"
else echo "[WARN] NPO failed" | tee -a "$PROGRESS"; fi

# ─── 4. SimNPO (bs=8, accum=4, eff=32) ───────────────────────────────────────
TASK="muse_${MODEL}_${DATA_SPLIT}_SimNPO"
echo "=== [4/7] SimNPO ===" | tee -a "$PROGRESS"
if smart_train ${TASK} unlearn/muse/simnpo.yaml 8 4; then
    run_eval ${TASK}; echo "[SimNPO done]" | tee -a "$PROGRESS"
else echo "[WARN] SimNPO failed" | tee -a "$PROGRESS"; fi

# ─── 5. DS-BiAL (bs=1, accum=32, eff=32) ────────────────────────────────────
TASK="muse_${MODEL}_${DATA_SPLIT}_DSBiAL"
echo "=== [5/7] DS-BiAL ===" | tee -a "$PROGRESS"
if smart_train_dsbial ${TASK}; then
    run_eval ${TASK}; echo "[DS-BiAL done]" | tee -a "$PROGRESS"
else echo "[WARN] DS-BiAL failed" | tee -a "$PROGRESS"; fi

# ─── 6. BLURNPO (bs=1, accum=32, eff=32) ────────────────────────────────────
TASK="muse_${MODEL}_${DATA_SPLIT}_BLURNPO"
echo "=== [6/7] BLURNPO ===" | tee -a "$PROGRESS"
if smart_train ${TASK} unlearn/muse/blurnpo.yaml 1 32; then
    run_eval ${TASK}; echo "[BLURNPO done]" | tee -a "$PROGRESS"
else echo "[WARN] BLURNPO failed" | tee -a "$PROGRESS"; fi

# ─── 7. RMU (bs=4, accum=8, eff=32, max_steps=80) ───────────────────────────
TASK="muse_${MODEL}_${DATA_SPLIT}_RMU"
echo "=== [7/7] RMU ===" | tee -a "$PROGRESS"
if smart_train ${TASK} unlearn/muse/rmu.yaml 4 8 +trainer.args.max_steps=80; then
    run_eval ${TASK}; echo "[RMU done]" | tee -a "$PROGRESS"
else echo "[WARN] RMU failed" | tee -a "$PROGRESS"; fi

echo "=== MUSE NEWS FULLY DONE ===" | tee -a "$PROGRESS"
