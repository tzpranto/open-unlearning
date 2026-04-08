#!/bin/bash
# Watchdog: monitors G series (and future series) — self-heals on failure.
# Runs forever in background. Safe to kill and restart.
# Logs to logs/ablation/watchdog.log
# Usage: nohup bash scripts/watchdog.sh &

BASE="/datadrive/forked/open-unlearning"
LOG_DIR="${BASE}/logs/ablation"
WATCHDOG_LOG="${LOG_DIR}/watchdog.log"
G_LOG="${LOG_DIR}/G_progress.log"
PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
CHECK_INTERVAL=60  # seconds between checks

mkdir -p "$LOG_DIR"
cd "$BASE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHDOG_LOG"; }

g_series_pid() {
    pgrep -f "run_G_series" 2>/dev/null | head -1
}

g_training_pid() {
    pgrep -f "src/train.py.*ablation_G" 2>/dev/null | head -1
}

g_series_done() {
    # G series is done if progress log contains the DONE marker
    grep -q "G0/G1/G2 DONE" "$G_LOG" 2>/dev/null
}

gpu_free() {
    # GPU is free if no compute processes are running
    local used
    used=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    [ "$used" -eq 0 ]
}

relaunch_g() {
    log "[RELAUNCH] Starting G series..."
    nohup bash "${BASE}/scripts/run_G_series.sh" >> "$G_LOG" 2>&1 &
    log "[RELAUNCH] G series PID $!"
}

log "=== Watchdog started ==="
log "Monitoring G series. Check interval: ${CHECK_INTERVAL}s"

consecutive_failures=0

while true; do
    sleep "$CHECK_INTERVAL"

    # If G series already finished successfully, nothing to do
    if g_series_done; then
        log "[OK] G series complete — watchdog done."
        exit 0
    fi

    # Check if G series process is alive
    pid=$(g_series_pid)
    if [ -n "$pid" ]; then
        log "[OK] G series running (PID $pid)"
        consecutive_failures=0
        continue
    fi

    # No process running and not done — dead
    log "[WARN] G series not running and not done. Investigating..."

    # Check last few lines of G log for clues
    last_lines=$(tail -5 "$G_LOG" 2>/dev/null)
    log "[LOG_TAIL] $last_lines"

    # Check for OOM in recent log
    if tail -30 "$G_LOG" 2>/dev/null | grep -q "OutOfMemoryError\|CUDA out of memory"; then
        log "[WARN] OOM detected. Waiting for GPU to clear..."
        for i in $(seq 1 12); do
            sleep 10
            if gpu_free; then
                log "[GPU] Cleared after ${i}×10s wait."
                break
            fi
        done
        if ! gpu_free; then
            log "[ERROR] GPU still occupied after 2min — killing stale processes"
            nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
            sleep 5
        fi
        # Delete incomplete weights so run_sibl doesn't skip
        for task in ablation_G0_npo_postinner ablation_G1_npo_weak_steering ablation_G2_npo_more_inner; do
            task_dir="${BASE}/saves/unlearn/${task}"
            eval_file="${task_dir}/evals/MUSE_EVAL.json"
            weight_file="${task_dir}/model-00001-of-00003.safetensors"
            # Only delete if weights exist but eval does NOT (incomplete run)
            if [ -f "$weight_file" ] && [ ! -f "$eval_file" ]; then
                log "[CLEANUP] Removing incomplete weights for $task"
                rm -rf "$task_dir"
            fi
        done
    fi

    ((consecutive_failures++))
    if [ "$consecutive_failures" -ge 5 ]; then
        log "[ERROR] 5 consecutive failures — stopping watchdog to avoid infinite loop."
        exit 1
    fi

    relaunch_g
done
