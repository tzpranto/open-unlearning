#!/usr/bin/env bash
# Resume pipeline from after trace collection — generates plots, summary.json,
# neuron_traces.pt, and psi scores from existing trace_results.pt.
# Run: nohup bash trace_analysis/scripts/run_postprocess_pipeline.sh > trace_analysis/logs/pipeline.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."  # repo root

LOGDIR="trace_analysis/figures/traces/muse_news"
ANALYSIS_DIR="trace_analysis/figures/traces/analysis"
TRACE_PT="$LOGDIR/trace_results.pt"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1b: Regenerate plots + summary.json from existing trace_results.pt ───
log "=== Step 1b: Regenerate plots + summary.json ==="
python3 - <<'PYEOF'
import json, os, sys, re, torch, time
import matplotlib
matplotlib.use("Agg")
import numpy as np

# Add repo to path for trace_activations imports
sys.path.insert(0, "trace_analysis/scripts")
from trace_activations import (
    plot_causal_traces, plot_activation_traces, plot_gradient_differential,
    plot_summary, _aggregate_grads_by_layer
)

output_dir = "trace_analysis/figures/traces/muse_news"
trace_path = f"{output_dir}/trace_results.pt"

print(f"Loading {trace_path} ...")
results = torch.load(trace_path, map_location="cpu", weights_only=False)
metadata = results["metadata"]
n_layers = metadata["n_layers"]

def _score_val(v):
    return float(v.mean()) if isinstance(v, torch.Tensor) else float(v)

# summary.json
json_out = {"metadata": metadata}
if "causal_traces" in results:
    json_out["causal_traces"] = {
        "forget": {str(k): v for k, v in results["causal_traces"]["forget"].items()},
        "retain": {str(k): v for k, v in results["causal_traces"]["retain"].items()},
    }
if "differential_scores" in results:
    json_out["top_50_differential_params"] = {
        k: _score_val(v)
        for k, v in sorted(results["differential_scores"].items(),
                           key=lambda x: _score_val(x[1]), reverse=True)[:50]
    }
if "layer_differential" in results:
    json_out["layer_differential"] = results["layer_differential"]
if "activation_traces" in results:
    json_out["activation_traces"] = results["activation_traces"]

json_path = f"{output_dir}/summary.json"
with open(json_path, "w") as f:
    json.dump(json_out, f, indent=2)
print(f"Saved {json_path}")

# Plots
if "causal_traces" in results:
    plot_causal_traces(results["causal_traces"]["forget"],
                       results["causal_traces"]["retain"], output_dir)
if "activation_traces" in results:
    plot_activation_traces(results["activation_traces"]["forget"],
                           results["activation_traces"]["retain"], output_dir)
if "gradient_traces" in results:
    plot_gradient_differential(results["gradient_traces"]["forget"],
                               results["gradient_traces"]["retain"],
                               results.get("differential_scores", {}),
                               n_layers, output_dir)
if all(k in results for k in ["causal_traces", "activation_traces", "gradient_traces"]):
    plot_summary(results["causal_traces"]["forget"], results["causal_traces"]["retain"],
                 results["activation_traces"]["forget"], results["activation_traces"]["retain"],
                 results["gradient_traces"]["forget"], results["gradient_traces"]["retain"],
                 n_layers, output_dir)

print("All plots saved.")
PYEOF
log "Step 1b done."

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
