#!/usr/bin/env bash
# Full trace analysis pipeline for MUSE News
# Run in background: nohup bash trace_analysis/scripts/run_full_trace_pipeline.sh > trace_analysis/logs/pipeline.log 2>&1 &
#
# Steps:
#   1. trace_activations.py  — causal + gradient + activation traces  (~90 min)
#   2. analyze_traces.py     — neuron-level analysis, neuron_traces.pt (~5 min)
#   3. compute_psi_layers.py — ψ layer scores for DS-BiAL steering     (<1 min)

set -e
cd "$(dirname "$0")/../.."  # repo root

LOGDIR="trace_analysis/figures/traces/muse_news"
ANALYSIS_DIR="trace_analysis/figures/traces/analysis"
TRACE_PT="$LOGDIR/trace_results.pt"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1: Full trace collection ──────────────────────────────────────────────
log "=== Step 1: trace_activations.py (causal + gradient + activation) ==="
python trace_analysis/scripts/trace_activations.py \
    --preset muse-news \
    --output_dir "$LOGDIR"
log "Step 1 done. trace_results.pt written."

# ── Step 2: Neuron-level analysis → neuron_traces.pt ──────────────────────────
log "=== Step 2: analyze_traces.py (neuron traces + heatmaps) ==="
mkdir -p "$ANALYSIS_DIR"
python trace_analysis/scripts/analyze_traces.py \
    --trace_file "$TRACE_PT" \
    --dataset_config raw \
    --output_dir "$ANALYSIS_DIR"
log "Step 2 done. neuron_traces.pt written."

# ── Step 3: ψ layer scores ─────────────────────────────────────────────────────
log "=== Step 3: compute_psi_layers.py ==="
python scripts/compute_psi_layers.py \
    --trace_path "$TRACE_PT" \
    --json
log "Step 3 done."

log "=== Pipeline complete. ==="
log "  trace_results.pt : $TRACE_PT"
log "  neuron_traces.pt : $ANALYSIS_DIR/neuron_traces.pt"
log "  neuron_analysis  : $ANALYSIS_DIR/neuron_analysis.json"
