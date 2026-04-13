#!/bin/bash
# Wait for Q series to complete, then launch R series
LOG="/datadrive/forked/open-unlearning/logs/ablation/R_launch.log"
echo "[$(date '+%H:%M:%S')] Waiting for Q series PID 106680..." | tee "$LOG"

while kill -0 106680 2>/dev/null; do
    sleep 30
done

echo "[$(date '+%H:%M:%S')] Q series completed. Launching R series..." | tee -a "$LOG"
cd /datadrive/forked/open-unlearning
bash scripts/run_R_series.sh 2>&1 | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] R series completed." | tee -a "$LOG"
