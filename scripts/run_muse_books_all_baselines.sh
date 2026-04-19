#!/bin/bash
# MUSE Books: run ALL baselines sequentially (single GPU)
# Each sub-script is restart-safe and skips completed runs.
# If the machine restarts, just re-run this script.
set -euo pipefail

BASE="/datadrive/forked/open-unlearning"
PROGRESS="/tmp/books_all_baselines_progress.log"

cd "$BASE"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

SCRIPTS=(
    scripts/run_muse_books_gradasc.sh
    scripts/run_muse_books_graddiff.sh
    scripts/run_muse_books_npo.sh
    scripts/run_muse_books_simnpo.sh
    scripts/run_muse_books_blurnpo.sh
    scripts/run_muse_books_rmu.sh
    scripts/run_muse_books_pdu.sh
    scripts/run_muse_books_dsbial.sh
)

log "=== MUSE Books: all baselines ==="
for script in "${SCRIPTS[@]}"; do
    name=$(basename "$script" .sh | sed 's/run_muse_books_//')
    log "--- Starting: ${name} ---"
    if bash "$script"; then
        log "--- Finished: ${name} ---"
    else
        log "[WARN] ${name} failed with exit code $?"
    fi
done
log "=== MUSE Books: all baselines DONE ==="
