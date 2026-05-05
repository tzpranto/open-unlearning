#!/bin/bash
# Fine-tune Llama-2-7b-chat on KnowUnDo with LoRA (matching original paper).
# LoRA r=8, alpha=16, dropout=0.1, all-linear. lr=1e-4, 10 epochs, eff. BS=32.
# Merges adapters and saves full model for use by all unlearn methods.
#
# Usage:
#   bash scripts/knowundo_finetune.sh          # copyright domain (default)
#   bash scripts/knowundo_finetune.sh privacy  # privacy domain

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

DOMAIN=${1:-copyright}
mkdir -p logs

nohup CUDA_VISIBLE_DEVICES=0 python scripts/knowundo_finetune_lora.py \
    --domain ${DOMAIN} \
    --output_dir saves/unlearn/knowundo_Llama-2-7b-chat_${DOMAIN}_ft \
    --seed 100 \
    > logs/knowundo_finetune_${DOMAIN}.log 2>&1 &

echo "PID: $!"
echo "Log: logs/knowundo_finetune_${DOMAIN}.log"
