#!/bin/bash
# Wait for R1 to finish (PID 125032), kill R runner, launch S series
LOG="/datadrive/forked/open-unlearning/logs/ablation/S_launch.log"
echo "[$(date '+%H:%M:%S')] Waiting for R1 training (PID 125032) to finish..." | tee "$LOG"

while kill -0 125032 2>/dev/null; do
    sleep 15
done

echo "[$(date '+%H:%M:%S')] R1 training done. Killing R runner..." | tee -a "$LOG"
# Kill the R series runner and its parent
kill 118984 2>/dev/null  # run_R_series.sh
kill 115960 2>/dev/null  # launch_R_after_Q.sh
sleep 5

# Kill any remaining R2 train process
pkill -f "ablation_R2_kl_outer_twophase" 2>/dev/null

echo "[$(date '+%H:%M:%S')] Launching S series..." | tee -a "$LOG"
cd /datadrive/forked/open-unlearning
bash scripts/run_S_series.sh 2>&1 | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] S series completed." | tee -a "$LOG"
