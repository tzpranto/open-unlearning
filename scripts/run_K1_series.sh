#!/bin/bash
# K1 series: gradient projection experiments
# K1a: G1 + projection (projection_scope=layer, strength=1.0)
# K1b: G1 + projection + implicit (stack both)
#
# Hypothesis: rk damage is in the gradient direction.
# Projecting NPO gradient orthogonal to retain gradient should preserve rk.
# G1 anchor: fk=0.274 rk=0.327 | Gold: fk<=0.328, rk>=0.560
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/K1_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting K1 series" | tee "$PROGRESS"
echo "G1 anchor: fk=0.274 rk=0.327 | Gold: fk<=0.328 rk>=0.560" | tee -a "$PROGRESS"

# ── Metric helpers ─────────────────────────────────────────────────────────────
print_metrics() {
    local EVAL_FILE=$1 LABEL=$2
    "${PYTHON_BIN}" -c "
import json
with open('$EVAL_FILE') as f: d = json.load(f)
def agg(d, k):
    v = d.get(k, {})
    return v.get('agg_value', v) if isinstance(v, dict) else v
fk=agg(d,'forget_knowmem_ROUGE'); rk=agg(d,'retain_knowmem_ROUGE')
fv=agg(d,'forget_verbmem_ROUGE'); ex=agg(d,'extraction_strength')
fk_tag = 'GOLD' if fk<=0.328 else ('NEAR' if fk<=0.36 else 'HURT')
fk_d = fk - 0.274; rk_d = rk - 0.327
print(f'  $LABEL: fk={fk:.4f}[{fk_tag}]({fk_d:+.3f} vs G1) rk={rk:.4f}({rk_d:+.3f} vs G1) fv={fv:.4f} ex={ex:.4f}')
" 2>/dev/null || echo "  $LABEL: [parse failed]"
}

run_train_eval() {
    local TASK=$1 CONFIG=$2 ACCUM=${3:-32}
    local DIR="saves/unlearn/${TASK}"
    local EVAL_FILE="${DIR}/evals/MUSE_EVAL.json"

    if compgen -G "${DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] $TASK weights exist" | tee -a "$PROGRESS"
    else
        echo "" | tee -a "$PROGRESS"
        echo "[$(date '+%H:%M:%S')] [TRAIN] $TASK (accum=$ACCUM)" | tee -a "$PROGRESS"
        "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/${CONFIG} model=${MODEL} data_split=${DATA_SPLIT} \
            trainer=SIBL task_name=${TASK} retain_logs_path=${RETAIN_LOGS} \
            trainer.args.per_device_train_batch_size=1 \
            trainer.args.gradient_accumulation_steps=${ACCUM} \
            trainer.args.gradient_checkpointing=true \
            trainer.args.num_train_epochs=1 \
            trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
            "${ATTN_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/train_${TASK}.log"
        local STATUS=$?
        if [ $STATUS -ne 0 ]; then
            echo "[ERROR] $TASK training failed (exit $STATUS)" | tee -a "$PROGRESS"
            return $STATUS
        fi
    fi

    if [ -f "$EVAL_FILE" ]; then
        echo "[CACHED] $TASK eval exists" | tee -a "$PROGRESS"
        print_metrics "$EVAL_FILE" "$TASK" | tee -a "$PROGRESS"
    else
        echo "[$(date '+%H:%M:%S')] [EVAL] $TASK" | tee -a "$PROGRESS"
        "${PYTHON_BIN}" src/eval.py \
            experiment=eval/muse/default.yaml data_split=${DATA_SPLIT} \
            task_name=${TASK} model=${MODEL} \
            model.model_args.pretrained_model_name_or_path=${DIR} \
            "${ATTN_ARGS[@]}" \
            paths.output_dir=${DIR}/evals \
            retain_logs_path=${RETAIN_LOGS} \
            2>&1 | tee "${LOG_DIR}/eval_${TASK}.log"
        [ -f "$EVAL_FILE" ] && print_metrics "$EVAL_FILE" "$TASK" | tee -a "$PROGRESS"
    fi
}

# ── K1a: projection only ───────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [K1a] G1 + gradient projection ===" | tee -a "$PROGRESS"
run_train_eval "ablation_K1a_projection" "ablation_K1a_projection" 32

# ── K1b: projection + implicit ────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [K1b] G1 + projection + implicit ===" | tee -a "$PROGRESS"
# Use accum=16 (same as G5) since implicit + steering needs extra memory
# K1b: projection + implicit — needs neumann_steps=1 to fit in 93GB
# (steering + implicit HVP + projection backward = ~90GB; neumann_steps=2 OOMs)
echo "[$(date '+%H:%M:%S')] [TRAIN] K1b (accum=16, neumann_steps=1)" | tee -a "$PROGRESS"
K1B_TASK="ablation_K1b_proj_implicit"
K1B_DIR="saves/unlearn/${K1B_TASK}"
if compgen -G "${K1B_DIR}/model-*.safetensors" > /dev/null 2>&1; then
    echo "[SKIP] K1b weights exist" | tee -a "$PROGRESS"
    [ -f "${K1B_DIR}/evals/MUSE_EVAL.json" ] && print_metrics "${K1B_DIR}/evals/MUSE_EVAL.json" "K1b" | tee -a "$PROGRESS"
else
    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/ablation_K1b_proj_implicit model=${MODEL} data_split=${DATA_SPLIT} \
        trainer=SIBL task_name=${K1B_TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_accumulation_steps=16 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        trainer.method_args.neumann_steps=1 \
        "${ATTN_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/train_${K1B_TASK}.log"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "[ERROR] K1b training failed (exit $STATUS) — skipping" | tee -a "$PROGRESS"
    else
        echo "[$(date '+%H:%M:%S')] [EVAL] K1b" | tee -a "$PROGRESS"
        "${PYTHON_BIN}" src/eval.py \
            experiment=eval/muse/default.yaml data_split=${DATA_SPLIT} \
            task_name=${K1B_TASK} model=${MODEL} \
            model.model_args.pretrained_model_name_or_path=${K1B_DIR} \
            "${ATTN_ARGS[@]}" \
            paths.output_dir=${K1B_DIR}/evals \
            retain_logs_path=${RETAIN_LOGS} \
            2>&1 | tee "${LOG_DIR}/eval_${K1B_TASK}.log"
        [ -f "${K1B_DIR}/evals/MUSE_EVAL.json" ] && print_metrics "${K1B_DIR}/evals/MUSE_EVAL.json" "K1b" | tee -a "$PROGRESS"
    fi
fi
# (run_train_eval placeholder removed — K1b uses custom block above)

# ── Summary ───────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] K1 series DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "  Gold targets: fk<=0.328, rk>=0.560" | tee -a "$PROGRESS"
echo "  G1 anchor:    fk=0.274, rk=0.327" | tee -a "$PROGRESS"
for task_eval in \
    "G1:saves/unlearn/ablation_G1_npo_weak_steering/evals/MUSE_EVAL.json" \
    "K1a:saves/unlearn/ablation_K1a_projection/evals/MUSE_EVAL.json" \
    "K1b:saves/unlearn/ablation_K1b_proj_implicit/evals/MUSE_EVAL.json"; do
    label="${task_eval%%:*}"; ef="${task_eval##*:}"
    [ -f "$ef" ] && print_metrics "$ef" "$label" | tee -a "$PROGRESS" || echo "  $label: no eval" | tee -a "$PROGRESS"
done
