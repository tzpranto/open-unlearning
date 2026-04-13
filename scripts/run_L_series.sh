#!/bin/bash
# L series: stacking experiments
# L3: G1 x2 epochs (lambda accumulation — does more training help rk?)
# L4: G1 + epsilon=0.50 (tighter ALM constraint, forces lambda to drive rk harder)
# L1: G1 + projection + epsilon=0.50 (gradient-space + constraint-space stacked)
#
# G1 anchor: fk=0.274 rk=0.327 | Gold: fk<=0.328, rk>=0.560
# Note: good forgetting + reasonable retain is also valuable
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/L_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting L series" | tee "$PROGRESS"
echo "G1 anchor: fk=0.274 rk=0.327 | Gold: fk<=0.328 rk>=0.560" | tee -a "$PROGRESS"

# ── Helpers ─────────────────────────────────────────────────────────────────
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
    local TASK=$1 ACCUM=${2:-32}
    shift 2
    local OVERRIDES=("$@")
    local DIR="saves/unlearn/${TASK}"
    local EVAL_FILE="${DIR}/evals/MUSE_EVAL.json"

    if compgen -G "${DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] $TASK weights exist" | tee -a "$PROGRESS"
        [ -f "$EVAL_FILE" ] && print_metrics "$EVAL_FILE" "$TASK" | tee -a "$PROGRESS" && return 0
    fi

    echo "" | tee -a "$PROGRESS"
    echo "[$(date '+%H:%M:%S')] [TRAIN] $TASK" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/ablation_K2_trajectory \
        model=${MODEL} data_split=${DATA_SPLIT} \
        trainer=SIBL task_name=${TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_accumulation_steps=${ACCUM} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        trainer.method_args.checkpoint_every_steps=0 \
        "${ATTN_ARGS[@]}" \
        "${OVERRIDES[@]}" \
        2>&1 | tee "${LOG_DIR}/train_${TASK}.log"
    local STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "[ERROR] $TASK failed (exit $STATUS)" | tee -a "$PROGRESS"; return $STATUS
    fi

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
}

# ── L3: G1 x2 epochs ─────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [L3] G1 x2 epochs ===" | tee -a "$PROGRESS"
run_train_eval "ablation_L3_2epochs" 32 \
    trainer.args.num_train_epochs=2

# ── L4: G1 + epsilon=0.50 ────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [L4] G1 + epsilon=0.50 ===" | tee -a "$PROGRESS"
run_train_eval "ablation_L4_tight_eps" 32 \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.epsilon=0.50

# ── L1: G1 + projection + epsilon=0.50 ───────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [L1] G1 + projection + epsilon=0.50 ===" | tee -a "$PROGRESS"
run_train_eval "ablation_L1_proj_tight" 32 \
    trainer.args.num_train_epochs=1 \
    trainer.method_args.epsilon=0.50 \
    trainer.method_args.gradient_projection=true \
    trainer.method_args.gradient_projection_scope=layer \
    trainer.method_args.projection_strength=1.0

# ── Summary ───────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] L series DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "  Gold targets: fk<=0.328, rk>=0.560" | tee -a "$PROGRESS"
echo "  G1 anchor:    fk=0.274, rk=0.327" | tee -a "$PROGRESS"
for task_eval in \
    "G1:saves/unlearn/ablation_G1_npo_weak_steering/evals/MUSE_EVAL.json" \
    "L3_2epochs:saves/unlearn/ablation_L3_2epochs/evals/MUSE_EVAL.json" \
    "L4_tight_eps:saves/unlearn/ablation_L4_tight_eps/evals/MUSE_EVAL.json" \
    "L1_proj_tight:saves/unlearn/ablation_L1_proj_tight/evals/MUSE_EVAL.json"; do
    label="${task_eval%%:*}"; ef="${task_eval##*:}"
    [ -f "$ef" ] && print_metrics "$ef" "$label" | tee -a "$PROGRESS" || echo "  $label: no eval" | tee -a "$PROGRESS"
done
