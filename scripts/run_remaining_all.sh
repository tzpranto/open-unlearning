#!/bin/bash
# Master pipeline: BLURNPO Books → all TOFU baselines
# Restart-safe: re-run after machine restart
set -euo pipefail

BASE="/datadrive/forked/open-unlearning"
PROGRESS="$BASE/saves/unlearn/_remaining_all_progress.log"

cd "$BASE"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

log "=========================================="
log "=== PHASE 1: BLURNPO Books (retry)    ==="
log "=========================================="
if bash scripts/run_muse_books_blurnpo.sh; then
    log "=== BLURNPO Books DONE ==="
else
    log "[WARN] BLURNPO Books failed"
fi

log "=========================================="
log "=== PHASE 2: TOFU All Baselines       ==="
log "=== 7 methods × 3 splits (1/5/10%)    ==="
log "=========================================="
if bash scripts/run_tofu_all_baselines.sh; then
    log "=== TOFU baselines DONE ==="
else
    log "[WARN] TOFU baselines had failures"
fi

log "=========================================="
log "=== ALL WORK COMPLETE                 ==="
log "=========================================="
