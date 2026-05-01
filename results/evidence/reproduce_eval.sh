#!/bin/bash
# Reproduce MUSE evaluation from saved checkpoints
# Usage: bash results/evidence/reproduce_eval.sh <split> <method> <seed>
# Example: bash results/evidence/reproduce_eval.sh Books RMU 42
#
# Prerequisites:
#   - Trained model checkpoint at saves/unlearn/muse_Llama-2-7b-hf_${SPLIT}_${METHOD}_s${SEED}/
#   - Retrain eval at saves/eval/muse_Llama-2-7b-hf_${SPLIT}_retrain/MUSE_EVAL.json

set -e
cd "$(dirname "$0")/../.."

SPLIT=${1:?Usage: $0 <split> <method> <seed>}
METHOD=${2:?}
SEED=${3:?}
MODEL=Llama-2-7b-hf

TASK="muse_${MODEL}_${SPLIT}_${METHOD}_s${SEED}"
OUTDIR="saves/unlearn/${TASK}"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${SPLIT}_retrain/MUSE_EVAL.json"

if [[ ! -d "$OUTDIR" ]]; then
    echo "ERROR: Model checkpoint not found at $OUTDIR"
    exit 1
fi

echo "Running MUSE eval: ${TASK}"
python src/eval.py \
    experiment=eval/muse/default.yaml \
    data_split=${SPLIT} \
    task_name=${TASK} \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${OUTDIR} \
    paths.output_dir=${OUTDIR}/evals \
    retain_logs_path=${RETAIN_LOGS}

echo "Results saved to ${OUTDIR}/evals/MUSE_SUMMARY.json"
cat ${OUTDIR}/evals/MUSE_SUMMARY.json
