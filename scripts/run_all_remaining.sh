#!/bin/bash
# Master pipeline: PDU News → all Books baselines (sequential, single GPU)
# Restart-safe: re-run this script after machine restart
set -euo pipefail

BASE="/datadrive/forked/open-unlearning"
PROGRESS="/datadrive/forked/open-unlearning/saves/unlearn/_all_remaining_progress.log"

cd "$BASE"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

log "=========================================="
log "=== PHASE 1: PDU on MUSE News         ==="
log "=========================================="
if bash scripts/run_muse_news_pdu.sh; then
    log "=== PDU News DONE ==="
else
    log "[WARN] PDU News failed with exit code $?"
fi

log "=========================================="
log "=== PHASE 2: All baselines on MUSE Books ==="
log "=========================================="
if bash scripts/run_muse_books_all_baselines.sh; then
    log "=== Books baselines DONE ==="
else
    log "[WARN] Books baselines had failures"
fi

log "=========================================="
log "=== ALL REMAINING WORK COMPLETE        ==="
log "=========================================="
