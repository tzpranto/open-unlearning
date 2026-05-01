#!/bin/bash
# Reproduce TOFU evaluation from saved checkpoints
# Usage: bash results/evidence/tofu/reproduce_eval.sh <model> <split> <method> <seed>
# Example: bash results/evidence/tofu/reproduce_eval.sh Llama-3.2-1B-Instruct forget01 GradAscent 42
#
# Prerequisites: model checkpoint at saves/unlearn/bs32_${MODEL}_${SPLIT}_${METHOD}_s${SEED}/

set -e
cd "$(dirname "$0")/../../.."

MODEL=${1:?Usage: $0 <model> <split> <method> <seed>}
SPLIT=${2:?}
METHOD=${3:?}
SEED=${4:?}

TASK="bs32_${MODEL}_${SPLIT}_${METHOD}_s${SEED}"
OUTDIR="saves/unlearn/${TASK}"
RETAIN_LOGS="saves/eval/tofu_${MODEL}_retain${SPLIT#forget}/TOFU_EVAL.json"

if [[ ! -d "$OUTDIR" ]]; then
    echo "ERROR: Model checkpoint not found at $OUTDIR"
    exit 1
fi

# Map forget split to retain split
case "$SPLIT" in
    forget01) RETAIN_SPLIT="retain99" ;;
    forget05) RETAIN_SPLIT="retain95" ;;
    forget10) RETAIN_SPLIT="retain90" ;;
esac

echo "Running TOFU eval: ${TASK}"
python src/eval.py \
    experiment=eval/tofu/default.yaml \
    split=${SPLIT} \
    task_name=${TASK} \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${OUTDIR} \
    paths.output_dir=${OUTDIR}/evals \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

echo "Results saved to ${OUTDIR}/evals/TOFU_EVAL.json"
