#!/bin/bash
# Fine-tune Llama-2-7b-chat on KnowUnDo to create the target model.
# This is a one-time step; all unlearning methods start from this checkpoint.
# ~400 examples, 10 epochs, LoRA r=8 — should take ~15 min on single GPU.
#
# Usage:
#   bash scripts/knowundo_finetune.sh          # copyright domain (default)
#   bash scripts/knowundo_finetune.sh privacy  # privacy domain

DOMAIN=${1:-copyright}

nohup python src/train.py \
  --config-name=train.yaml \
  experiment=finetune/knowundo/default \
  domain=${DOMAIN} \
  task_name=knowundo_Llama-2-7b-chat_${DOMAIN}_ft \
  > logs/knowundo_finetune_${DOMAIN}.log 2>&1 &

echo "PID: $!"
echo "Log: logs/knowundo_finetune_${DOMAIN}.log"
