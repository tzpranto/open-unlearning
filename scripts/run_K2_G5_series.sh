#!/bin/bash
# K2 + G5 series
# K2: G1 per-step Pareto trajectory (checkpoint every 5 steps, eval each)
# G5: G1 + implicit correction (stackability test)
#
# K2 goal: find step t* where fk first crosses 0.328, and report rk at that step.
# G5 goal: does implicit correction give free rk gain on top of G1's steering?
# Gold targets: fk<=0.328, rk>=0.560 | G1 anchor: fk=0.274, rk=0.327
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/K2G5_progress.log"
TSV_K2="${LOG_DIR}/K2_trajectory.tsv"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting K2 + G5 series" | tee "$PROGRESS"
echo "G1 anchor: fk=0.274 rk=0.327 | Gold: fk<=0.328 rk>=0.560" | tee -a "$PROGRESS"

# ── Metric helpers ─────────────────────────────────────────────────────────────
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
fk_ok = 'GOLD' if fk<=0.328 else ('NEAR' if fk<=0.360 else 'HURT')
rk_ok = 'GOLD' if rk>=0.560 else ('NEAR' if rk>=0.450 else f'{rk:.3f}')
fk_delta = fk - 0.274; rk_delta = rk - 0.327
print(f'  fk={fk:.4f}[{fk_ok}]({fk_delta:+.3f} vs G1) rk={rk:.4f}[{rk_ok}]({rk_delta:+.3f} vs G1) fv={fv:.4f} ex={ex:.4f}')
" 2>/dev/null || echo "  [metrics parse failed]"
}

run_eval() {
    local TASK=$1 MODEL_PATH=$2 EVAL_DIR=$3
    echo "[$(date '+%H:%M:%S')] Evaluating ${TASK} from ${MODEL_PATH}" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/eval.py \
        experiment=eval/muse/default.yaml data_split=${DATA_SPLIT} \
        task_name=${TASK} model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
        "${ATTN_ARGS[@]}" \
        paths.output_dir=${EVAL_DIR} \
        retain_logs_path=${RETAIN_LOGS} \
        2>&1 | tee "${LOG_DIR}/eval_${TASK}.log"
    local EVAL_FILE="${EVAL_DIR}/MUSE_EVAL.json"
    [ -f "$EVAL_FILE" ] && print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"
}

# ── K2: Train G1 with per-step checkpoints ─────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [K2] G1 trajectory (checkpoint every 5 steps) ===" | tee -a "$PROGRESS"
K2_TASK="ablation_K2_trajectory"
K2_DIR="saves/unlearn/${K2_TASK}"

if compgen -G "${K2_DIR}/model-*.safetensors" > /dev/null 2>&1; then
    echo "[SKIP] K2 final weights exist — skipping training" | tee -a "$PROGRESS"
else
    echo "[$(date '+%H:%M:%S')] [TRAIN] K2" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/ablation_K2_trajectory model=${MODEL} data_split=${DATA_SPLIT} \
        trainer=SIBL task_name=${K2_TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_accumulation_steps=32 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        "${ATTN_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/train_${K2_TASK}.log"
    STATUS=$?
    [ $STATUS -ne 0 ] && echo "[ERROR] K2 training failed (exit $STATUS)" | tee -a "$PROGRESS" && exit $STATUS
fi

# ── K2: Eval each checkpoint (steps 5, 10, 15, 20, 25) ───────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [K2] Evaluating step checkpoints ===" | tee -a "$PROGRESS"

# Write TSV header
echo -e "step\tfk\trk\tfv\tex\tfk_vs_gold\trk_vs_gold" > "$TSV_K2"

for STEP in 5 10 15 20 25; do
    CKPT_PATH="${K2_DIR}/checkpoint-step-${STEP}"
    EVAL_DIR="${K2_DIR}/evals/step_${STEP}"
    if [ ! -d "$CKPT_PATH" ]; then
        echo "[WARN] K2 checkpoint-step-${STEP} not found, skipping" | tee -a "$PROGRESS"
        continue
    fi
    if [ -f "${EVAL_DIR}/MUSE_EVAL.json" ]; then
        echo "[CACHED] K2 step ${STEP} eval exists" | tee -a "$PROGRESS"
    else
        run_eval "K2_step${STEP}" "$CKPT_PATH" "$EVAL_DIR"
    fi
    # Append to TSV
    [ -f "${EVAL_DIR}/MUSE_EVAL.json" ] && "${PYTHON_BIN}" -c "
import json
with open('${EVAL_DIR}/MUSE_EVAL.json') as f: d = json.load(f)
def agg(d, k):
    v = d.get(k, {})
    return v.get('agg_value', v) if isinstance(v, dict) else v
fk=agg(d,'forget_knowmem_ROUGE'); rk=agg(d,'retain_knowmem_ROUGE')
fv=agg(d,'forget_verbmem_ROUGE'); ex=agg(d,'extraction_strength')
fk_g = 'GOLD' if fk<=0.328 else 'MISS'
rk_g = 'GOLD' if rk>=0.560 else 'MISS'
print(f'${STEP}\t{fk:.4f}\t{rk:.4f}\t{fv:.4f}\t{ex:.4f}\t{fk_g}\t{rk_g}')
" 2>/dev/null >> "$TSV_K2"
done

# K2 final checkpoint eval (saves/unlearn/K2/evals/)
FINAL_EVAL_DIR="${K2_DIR}/evals"
if [ ! -f "${FINAL_EVAL_DIR}/MUSE_EVAL.json" ]; then
    run_eval "${K2_TASK}" "${K2_DIR}" "${FINAL_EVAL_DIR}"
fi

# Print K2 trajectory table
echo "" | tee -a "$PROGRESS"
echo "=== K2 TRAJECTORY ===" | tee -a "$PROGRESS"
echo "step | fk        | rk        | fk_gold | rk_gold" | tee -a "$PROGRESS"
echo "-----|-----------|-----------|---------|--------" | tee -a "$PROGRESS"
tail -n +2 "$TSV_K2" | while IFS=$'\t' read step fk rk fv ex fk_g rk_g; do
    printf "  %2s | %9s | %9s | %-7s | %s\n" "$step" "$fk" "$rk" "$fk_g" "$rk_g" | tee -a "$PROGRESS"
done

# Find first step where fk <= 0.328
echo "" | tee -a "$PROGRESS"
echo "--- First step fk<=0.328 ---" | tee -a "$PROGRESS"
"${PYTHON_BIN}" -c "
import csv
rows = list(csv.DictReader(open('$TSV_K2'), delimiter='\t'))
hits = [(r['step'], float(r['fk']), float(r['rk'])) for r in rows if float(r['fk']) <= 0.328]
if hits:
    s, fk, rk = hits[0]
    print(f'  Step {s}: fk={fk:.4f} rk={rk:.4f} (rk gap to gold: {0.560-rk:+.3f})')
else:
    print('  No step reached fk<=0.328 — fk never hits gold threshold in 25 steps')
" 2>/dev/null | tee -a "$PROGRESS"

# ── G5: G1 + implicit correction ──────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [G5] G1 + implicit correction ===" | tee -a "$PROGRESS"
G5_TASK="ablation_G5_implicit"
G5_DIR="saves/unlearn/${G5_TASK}"

if compgen -G "${G5_DIR}/model-*.safetensors" > /dev/null 2>&1; then
    echo "[SKIP] G5 weights exist" | tee -a "$PROGRESS"
    EVAL_FILE="${G5_DIR}/evals/MUSE_EVAL.json"
    if [ ! -f "$EVAL_FILE" ]; then
        run_eval "$G5_TASK" "$G5_DIR" "${G5_DIR}/evals"
    else
        echo "[CACHED]" | tee -a "$PROGRESS"
        print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"
    fi
else
    echo "[$(date '+%H:%M:%S')] [TRAIN] G5" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/ablation_G5_implicit model=${MODEL} data_split=${DATA_SPLIT} \
        trainer=SIBL task_name=${G5_TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_accumulation_steps=16 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        "${ATTN_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/train_${G5_TASK}.log"
    STATUS=$?
    [ $STATUS -ne 0 ] && echo "[ERROR] G5 training failed (exit $STATUS)" | tee -a "$PROGRESS" && exit $STATUS
    run_eval "$G5_TASK" "$G5_DIR" "${G5_DIR}/evals"
fi

# ── Final summary ──────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] K2+G5 DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "  Gold targets: fk<=0.328, rk>=0.560" | tee -a "$PROGRESS"
echo "  G1 anchor:    fk=0.274, rk=0.327" | tee -a "$PROGRESS"
for task_eval in \
    "ablation_G1_npo_weak_steering:saves/unlearn/ablation_G1_npo_weak_steering/evals/MUSE_EVAL.json" \
    "ablation_G5_implicit:saves/unlearn/ablation_G5_implicit/evals/MUSE_EVAL.json"; do
    task="${task_eval%%:*}"; ef="${task_eval##*:}"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done
echo "  K2 trajectory: $TSV_K2" | tee -a "$PROGRESS"
