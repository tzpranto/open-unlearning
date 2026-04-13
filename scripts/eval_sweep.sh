#!/bin/bash
# Evaluate multiple PerTA/LoRA models against MUSE News benchmark
# Usage: bash scripts/eval_sweep.sh saves/unlearn/perta_l*_a* saves/unlearn/lora_*_merged
set -euo pipefail

DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
PYTHON_BIN="${CONDA_PREFIX:-/datadrive/miniconda3}/bin/python"

cd /datadrive/forked/open-unlearning

for MODEL_DIR in "$@"; do
    [ -d "$MODEL_DIR" ] || continue
    TASK=$(basename "$MODEL_DIR")
    EVAL_DIR="${MODEL_DIR}/evals"

    # Skip if already evaluated
    if [ -f "${EVAL_DIR}/MUSE_SUMMARY.json" ]; then
        echo "[SKIP] ${TASK}: already evaluated"
        continue
    fi

    # Check model weights exist
    if ! compgen -G "${MODEL_DIR}/model-*.safetensors" > /dev/null 2>&1; then
        if ! compgen -G "${MODEL_DIR}/model.safetensors" > /dev/null 2>&1; then
            echo "[SKIP] ${TASK}: no model weights found"
            continue
        fi
    fi

    echo "[EVAL] ${TASK}"
    # Clean stale eval artifacts
    rm -rf "${EVAL_DIR}/.hydra" "${EVAL_DIR}/eval.log" 2>/dev/null

    "${PYTHON_BIN}" src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${TASK} \
        model=Llama-2-7b-hf \
        model.model_args.pretrained_model_name_or_path=${MODEL_DIR} \
        model.model_args.attn_implementation=sdpa \
        paths.output_dir=${EVAL_DIR} \
        retain_logs_path=${RETAIN_LOGS} \
        2>&1 | tee /tmp/eval_${TASK}.log

    # Print quick results
    if [ -f "${EVAL_DIR}/MUSE_SUMMARY.json" ]; then
        python3 -c "
import json, sys
task='${TASK}'
d = json.load(open('${EVAL_DIR}/MUSE_SUMMARY.json'))
fk = d['forget_knowmem_ROUGE']
rk = d['retain_knowmem_ROUGE']
vm = d['forget_verbmem_ROUGE']
pred = 0.55*fk + 0.17
delta = rk - pred
marker = '***' if delta > 0.03 else '**' if delta > 0 else ''
print(f'  => {task}: fk={fk:.3f} rk={rk:.3f} vm={vm:.3f} delta={delta:+.3f} {marker}')
"
    fi
    echo ""
done

echo "[DONE] All evaluations complete"
