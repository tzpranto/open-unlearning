#!/bin/bash
# Master pipeline for paradigm-shift experiments
# Chains: PerTA eval → LoRA training → LoRA eval → Results summary
set -euo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="${CONDA_PREFIX:-/datadrive/miniconda3}/bin/python"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
PROGRESS="/tmp/paradigm_shift_progress.log"
SAVES="saves/unlearn"

cd /datadrive/forked/open-unlearning

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$PROGRESS"; }

run_eval() {
    local TASK=$1 MODEL_DIR=$2
    local EVAL_DIR="${MODEL_DIR}/evals"
    if [ -f "${EVAL_DIR}/MUSE_SUMMARY.json" ]; then
        log "[SKIP] ${TASK}: already evaluated"
        return 0
    fi
    rm -rf "${EVAL_DIR}/.hydra" "${EVAL_DIR}/eval.log" 2>/dev/null
    log "[EVAL] ${TASK}"
    "${PYTHON_BIN}" src/eval.py \
        experiment=eval/muse/default.yaml data_split=${DATA_SPLIT} \
        task_name=${TASK} model=Llama-2-7b-hf \
        model.model_args.pretrained_model_name_or_path=${MODEL_DIR} \
        model.model_args.attn_implementation=sdpa \
        paths.output_dir=${EVAL_DIR} \
        retain_logs_path=${RETAIN_LOGS} 2>&1 | tail -5

    if [ -f "${EVAL_DIR}/MUSE_SUMMARY.json" ]; then
        "${PYTHON_BIN}" -c "
import json
d = json.load(open('${EVAL_DIR}/MUSE_SUMMARY.json'))
fk, rk = d['forget_knowmem_ROUGE'], d['retain_knowmem_ROUGE']
vm = d['forget_verbmem_ROUGE']
pred = 0.55*fk + 0.17
delta = rk - pred
m = '***' if delta > 0.03 else '**' if delta > 0 else ''
print(f'  => fk={fk:.3f} rk={rk:.3f} vm={vm:.3f} delta={delta:+.3f} {m}')
" | tee -a "$PROGRESS"
    fi
}

log "=== PARADIGM SHIFT PIPELINE ==="

# -------------------------------------------------------
# PHASE 1: Evaluate PerTA variants
# -------------------------------------------------------
log "--- Phase 1: PerTA evaluation ---"
for d in ${SAVES}/perta_l*_a*; do
    [ -d "$d" ] || continue
    TASK=$(basename "$d")
    # Check if model weights exist
    if compgen -G "${d}/model-*.safetensors" > /dev/null 2>&1 || [ -f "${d}/model.safetensors" ]; then
        run_eval "$TASK" "$d"
    else
        log "[SKIP] ${TASK}: no model weights"
    fi
done

# -------------------------------------------------------
# PHASE 2: LoRA retain recovery (ranks 4, 8, 16)
# -------------------------------------------------------
log "--- Phase 2: LoRA retain recovery ---"
"${PYTHON_BIN}" scripts/lora_retain_recovery.py \
    --base_model ${SAVES}/ablation_G1_npo_weak_steering \
    --data_split ${DATA_SPLIT} \
    --lora_ranks 4 8 16 \
    --epochs 3 --lr 2e-4 \
    --batch_size 4 --grad_accum 4 \
    2>&1 | tee -a "$PROGRESS"

# -------------------------------------------------------
# PHASE 3: Evaluate LoRA variants
# -------------------------------------------------------
log "--- Phase 3: LoRA evaluation ---"
for d in ${SAVES}/lora_r*_merged; do
    [ -d "$d" ] || continue
    TASK=$(basename "$d")
    if compgen -G "${d}/model-*.safetensors" > /dev/null 2>&1 || [ -f "${d}/model.safetensors" ]; then
        run_eval "$TASK" "$d"
    else
        log "[SKIP] ${TASK}: no model weights"
    fi
done

# -------------------------------------------------------
# PHASE 4: Results summary
# -------------------------------------------------------
log ""
log "=== RESULTS SUMMARY ==="
log ""
log "Gold targets: fk<=0.328, rk>=0.552"
log "CE frontier: rk = 0.55*fk + 0.17"
log ""
"${PYTHON_BIN}" << 'PYEOF'
import json, glob, os

results = []
for pattern in ["perta_l*_a*", "lora_r*_merged"]:
    for f in sorted(glob.glob(f"saves/unlearn/{pattern}/evals/MUSE_SUMMARY.json")):
        exp = f.split("/")[-3]
        d = json.load(open(f))
        fk = d["forget_knowmem_ROUGE"]
        rk = d["retain_knowmem_ROUGE"]
        vm = d["forget_verbmem_ROUGE"]
        pred = 0.55*fk + 0.17
        delta = rk - pred
        results.append((exp, fk, rk, vm, delta))

# Add reference points
refs = [
    ("Gold (retrain)", 0.328, 0.560, 0.202, 0.560 - 0.55*0.328 - 0.17),
    ("Target", 0.644, 0.555, 0.579, 0.555 - 0.55*0.644 - 0.17),
    ("G1 (best forget)", 0.274, 0.327, 0.224, 0.327 - 0.55*0.274 - 0.17),
    ("T8f (best frontier)", 0.428, 0.461, 0.344, 0.461 - 0.55*0.428 - 0.17),
]

print(f"{'Experiment':<40} {'fk':>6} {'rk':>6} {'vm':>6} {'delta':>7}")
print("-" * 70)
for name, fk, rk, vm, delta in refs:
    print(f"{name:<40} {fk:6.3f} {rk:6.3f} {vm:6.3f} {delta:+7.3f}  [ref]")
print("-" * 70)
for name, fk, rk, vm, delta in sorted(results, key=lambda x: -x[4]):
    m = "***" if delta > 0.03 else "**" if delta > 0 else ""
    print(f"{name:<40} {fk:6.3f} {rk:6.3f} {vm:6.3f} {delta:+7.3f}  {m}")

if results:
    best = max(results, key=lambda x: x[4])
    print(f"\nBest new result: {best[0]} (delta={best[4]:+.3f})")
    if best[4] > 0.056:
        print(">>> NEW FRONTIER RECORD! Beats T8f delta=+0.056 <<<")
PYEOF

log ""
log "=== PIPELINE COMPLETE ==="
