#!/bin/bash
# MemFlex Phase 1: Gradient-based parameter localization for KnowUnDo.
# Produces a JSON file listing which parameters to unfreeze during unlearning.
# Run this BEFORE the unlearning step.
#
# Usage:
#   bash scripts/knowundo_memflex_localize.sh            # copyright (default)
#   bash scripts/knowundo_memflex_localize.sh privacy    # privacy domain

DOMAIN=${1:-copyright}
OUTPUT_DIR="outputs/knowundo_memflex_localize_${DOMAIN}"
MODEL_PATH="outputs/knowundo_Llama-2-7b-chat_${DOMAIN}_ft"

mkdir -p ${OUTPUT_DIR}

nohup python scripts/knowundo_memflex_localize_run.py \
  --model_path ${MODEL_PATH} \
  --domain ${DOMAIN} \
  --output_path ${OUTPUT_DIR}/located_params.json \
  --mu 0.92 \
  --sigma 6e-4 \
  --num_copies 5 \
  > logs/knowundo_memflex_localize_${DOMAIN}.log 2>&1 &

echo "PID: $!"
echo "Log: logs/knowundo_memflex_localize_${DOMAIN}.log"
echo "Output: ${OUTPUT_DIR}/located_params.json"
