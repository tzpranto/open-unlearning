#!/bin/bash
# Eval LoRA-BiAL exp_03 (our champion) on TOFU 1B forget01
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
cd /datadrive/forked/open-unlearning

OUTDIR="saves/unlearn/tofu_1b_01_exp_03/evals"
mkdir -p "$OUTDIR"

echo "[$(date)] Evaluating LoRA-BiAL exp_03 (1B forget01)"
python src/eval.py \
    experiment=eval/tofu/default.yaml \
    forget_split=forget01 \
    holdout_split=holdout01 \
    model=Llama-3.2-1B-Instruct \
    task_name=tofu_1b_01_exp_03 \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/tofu_1b_01_exp_03 \
    paths.output_dir="$OUTDIR" \
    retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json

echo "[$(date)] Done. Results:"
cat "$OUTDIR/TOFU_SUMMARY.json"
