#!/bin/bash
# Auto-launcher: waits for I0 (DGA scoring) to complete, then launches full I series.
# Exits after launch or after 3h timeout.
DGA_OUT="trace_analysis/figures/traces/analysis/dga_selectivity_G1.pt"
LOG="logs/ablation/I0_dga_scoring.log"
PROGRESS="logs/ablation/I_progress.log"
MAX_WAIT=180  # minutes
WAITED=0
cd /datadrive/forked/open-unlearning

while [ ! -f "$DGA_OUT" ]; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[$(date '+%H:%M:%S')] TIMEOUT waiting for I0 DGA scores" | tee -a "$PROGRESS"
        exit 1
    fi
    # Check for crash
    if ! pgrep -f "score_dga.py" > /dev/null 2>&1; then
        if [ ! -f "$DGA_OUT" ]; then
            echo "[$(date '+%H:%M:%S')] DGA scorer died without output. Last log:" | tee -a "$PROGRESS"
            tail -5 "$LOG" | tee -a "$PROGRESS"
            exit 1
        fi
    fi
    sleep 60
    WAITED=$((WAITED+1))
done

echo "[$(date '+%H:%M:%S')] I0 complete. Launching I series..." | tee -a "$PROGRESS"
nohup bash scripts/run_I_series.sh >> "$PROGRESS" 2>&1 &
echo "I series PID: $!"
