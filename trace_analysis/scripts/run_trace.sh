#!/bin/bash
# Collect mechanistic interpretability traces for MUSE News + Llama-2-7b.
#
# Usage:
#   bash trace_analysis/scripts/run_trace.sh                        # full run
#   bash trace_analysis/scripts/run_trace.sh --skip_causal           # quick (no causal tracing)
#   bash trace_analysis/scripts/run_trace.sh --n_samples 20          # tiny test run

set -euo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True

PYTHON_BIN=python
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
echo "[INFO] Using python: ${PYTHON_BIN} ($(${PYTHON_BIN} -V 2>&1))"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SCRIPT_DIR}/trace_activations.py" "$@"
