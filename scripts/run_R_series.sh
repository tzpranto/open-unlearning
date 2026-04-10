#!/bin/bash
# R series: KL(pretrained||model) as outer ALM retain loss — frontier shift attempt
# KEY INSIGHT: The CE-based Pareto frontier has slope 0.55 rk/fk — gold target unreachable.
# CE retain loss re-learns forget through weight sharing. KL toward pretrained doesn't.
# Pretrained has rk=0.555 (near gold) — KL anchors retain to that level.
#
# Experiments:
#   R0: G1 + KL outer retain (control: does KL shift the frontier from G1?)
#   R1: P0 + KL outer retain (strong ALM + KL: best of P0 + frontier shift?)
#   R2: Q0 + KL outer retain (two-phase + KL: combine timing + loss improvement?)
#
# Frontier anchors: G1 (0.274, 0.327) | P0 (0.524, 0.465) | Gold (0.328, 0.560)
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/R_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting R series (KL outer retain — frontier shift)" | tee "$PROGRESS"
echo "Frontier: G1 (0.274, 0.327) | P0 (0.524, 0.465) | Gold (0.328, 0.560)" | tee -a "$PROGRESS"

print_metrics() {
    local EVAL_FILE=$1
    "${PYTHON_BIN}" -c "
import json
with open('$EVAL_FILE') as f: d = json.load(f)
def agg(d, k):
    v = d.get(k, {})
    return v.get('agg_value', v) if isinstance(v, dict) else v
fk=agg(d,'forget_knowmem_ROUGE'); rk=agg(d,'retain_knowmem_ROUGE')
fv=agg(d,'forget_verbmem_ROUGE'); ex=agg(d,'extraction_strength')
fk_gold = 'GOLD' if fk<=0.328 else ('OK' if fk<=0.340 else 'HURT')
rk_gold = 'GOLD' if rk>=0.560 else f'{rk:.3f}'
fk_delta = fk - 0.274
rk_delta = rk - 0.327
print(f'  fk={fk:.4f}[{fk_gold}]({fk_delta:+.3f} vs G1) rk={rk:.4f}[{rk_gold}]({rk_delta:+.3f} vs G1) fv={fv:.4f} ex={ex:.4f}')
" 2>/dev/null || echo "  [metrics parse failed]"
}

run_eval() {
    local TASK=$1
    echo "[$(date '+%H:%M:%S')] Evaluating ${TASK}" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/eval.py \
        experiment=eval/muse/default.yaml data_split=${DATA_SPLIT} \
        task_name=${TASK} model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK} \
        "${ATTN_ARGS[@]}" \
        paths.output_dir=saves/unlearn/${TASK}/evals \
        retain_logs_path=${RETAIN_LOGS} \
        2>&1 | tee "${LOG_DIR}/eval_${TASK}.log"
    local EVAL_FILE="saves/unlearn/${TASK}/evals/MUSE_EVAL.json"
    [ -f "$EVAL_FILE" ] && print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"
}

run_sibl() {
    local TASK=$1 EXPERIMENT=$2
    local TASK_DIR="saves/unlearn/${TASK}"
    if compgen -G "${TASK_DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] ${TASK}: weights exist" | tee -a "$PROGRESS"
        local EVAL_FILE="${TASK_DIR}/evals/MUSE_EVAL.json"
        if [ ! -f "$EVAL_FILE" ]; then run_eval ${TASK}
        else echo "[CACHED]" | tee -a "$PROGRESS"; print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"; fi
        return 0
    fi
    echo "[$(date '+%H:%M:%S')] [TRAIN] ${TASK}" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=${EXPERIMENT} model=${MODEL} data_split=${DATA_SPLIT} \
        trainer=SIBL task_name=${TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_accumulation_steps=32 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        "${ATTN_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/train_${TASK}.log"
    local STATUS=$?
    [ $STATUS -ne 0 ] && echo "[WARN] ${TASK} failed (exit $STATUS)" | tee -a "$PROGRESS" && return $STATUS
    run_eval ${TASK}
}

# ── R0: G1 + KL outer retain (CONTROL — cleanest test) ───────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [R0] G1 config + KL outer retain (vs G1's CE outer retain) ===" | tee -a "$PROGRESS"
echo "  Tests: does KL outer shift the frontier from G1's (0.274, 0.327)?" | tee -a "$PROGRESS"
run_sibl ablation_R0_kl_outer unlearn/muse/ablation_R0_kl_outer \
    || echo "[WARN] R0 failed" | tee -a "$PROGRESS"

# ── R1: P0 + KL outer retain ─────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [R1] P0 config + KL outer retain (strong ALM + KL) ===" | tee -a "$PROGRESS"
echo "  Tests: does KL outer improve P0's (0.524, 0.465)?" | tee -a "$PROGRESS"
run_sibl ablation_R1_kl_outer_strong_alm unlearn/muse/ablation_R1_kl_outer_strong_alm \
    || echo "[WARN] R1 failed" | tee -a "$PROGRESS"

# ── R2: Q0 + KL outer retain ─────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [R2] Q0 config + KL outer retain (two-phase + KL) ===" | tee -a "$PROGRESS"
echo "  Tests: does KL outer improve Q0's (0.466, 0.423)?" | tee -a "$PROGRESS"
run_sibl ablation_R2_kl_outer_twophase unlearn/muse/ablation_R2_kl_outer_twophase \
    || echo "[WARN] R2 failed" | tee -a "$PROGRESS"

# ── Summary ──────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] R series DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "  Gold targets: fk<=0.328, rk>=0.560" | tee -a "$PROGRESS"
echo "  CE frontier: G1 (0.274, 0.327) | P0 (0.524, 0.465)" | tee -a "$PROGRESS"
for task in ablation_R0_kl_outer ablation_R1_kl_outer_strong_alm ablation_R2_kl_outer_twophase; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done
