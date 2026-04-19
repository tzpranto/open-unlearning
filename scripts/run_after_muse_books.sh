#!/bin/bash
# Wait for MUSE Books pipeline to finish, then launch TOFU baselines
set -euo pipefail

BASE="/datadrive/forked/open-unlearning"
cd "$BASE"

MUSE_PID=$(cat /tmp/all_remaining_pid.txt 2>/dev/null || echo "")
PROGRESS="$BASE/saves/unlearn/_post_muse_progress.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

# Wait for MUSE pipeline if still running
if [[ -n "$MUSE_PID" ]] && kill -0 "$MUSE_PID" 2>/dev/null; then
    log "Waiting for MUSE Books pipeline (PID $MUSE_PID) to finish..."
    while kill -0 "$MUSE_PID" 2>/dev/null; do sleep 60; done
    log "MUSE Books pipeline finished."
else
    log "MUSE Books pipeline not running (or already done)."
fi

log "=========================================="
log "=== Starting TOFU All Baselines        ==="
log "=========================================="
if bash scripts/run_tofu_all_baselines.sh; then
    log "=== TOFU All Baselines DONE ==="
else
    log "[WARN] TOFU baselines had failures (exit code $?)"
fi

log "=========================================="
log "=== ALL WORK COMPLETE                  ==="
log "=========================================="
