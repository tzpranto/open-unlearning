#!/bin/bash
# Run mechanistic interpretability traces for any dataset/model.
#
# Usage:
#   bash trace_analysis/scripts/run_trace.sh --preset muse-news
#   bash trace_analysis/scripts/run_trace.sh --preset muse-books
#   bash trace_analysis/scripts/run_trace.sh --preset wmdp-bio
#   bash trace_analysis/scripts/run_trace.sh --preset wmdp-cyber
#   bash trace_analysis/scripts/run_trace.sh --preset muse-news --skip_causal
#   bash trace_analysis/scripts/run_trace.sh --preset muse-news --n_samples 100
#
#   # Fully custom:
#   bash trace_analysis/scripts/run_trace.sh \
#       --model_name muse-bench/MUSE-News_target \
#       --tokenizer meta-llama/Llama-2-7b-hf \
#       --dataset muse-bench/MUSE-News \
#       --dataset_config train \
#       --forget_split forget --retain_split retain1 \
#       --text_field text --n_samples 3554 \
#       --output_dir trace_analysis/figures/traces/my_run
#
# Available presets: muse-news, muse-books, wmdp-bio, wmdp-cyber

set -euo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True

PYTHON_BIN=python
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
echo "[INFO] Python: ${PYTHON_BIN} ($(${PYTHON_BIN} -V 2>&1))"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/trace_activations.py" "$@"
