#!/bin/bash
# LLM Judge for KnowUnDo: FT targets + 6 baselines (5-fold) + BLADE unified (5-fold)
# Uses Opus 4.6 (Opus 4.7 refuses on copyrighted content)
cd /data/open-unlearning
export LLM_JUDGE_MODEL="us.anthropic.claude-opus-4-6-v1"

# Initialize results CSV if not exists
if [ ! -f results/knowundo_llm_judge.csv ]; then
    echo "model,split,method,seed,forget_leakage,retain_accuracy,response_quality,forget_rq,retain_rq,n_forget,n_retain" > results/knowundo_llm_judge.csv
fi

# All eval dirs to judge
EVAL_DIRS=()

# FT target (copyright + privacy)
EVAL_DIRS+=("saves/unlearn/knowundo_Llama-2-7b-chat_copyright_ft/evals")
EVAL_DIRS+=("saves/unlearn/knowundo_Llama-2-7b-chat_privacy_ft/evals")

# 6 baselines × 5 seeds × 2 domains = 60
# Path: saves/eval/knowundo_<domain>_<method>_s<seed>/
for METHOD in default grad_diff npo simnpo rmu pdu; do
    for SEED in 42 123 456 789 1024; do
        EVAL_DIRS+=("saves/eval/knowundo_copyright_${METHOD}_s${SEED}")
        EVAL_DIRS+=("saves/eval/knowundo_privacy_${METHOD}_s${SEED}")
    done
done

# BLADE unified × 5 seeds × 2 domains = 10
# Path: saves/unlearn/knowundo_BLADE_<domain>_unified_s<seed>/evals/
for SEED in 42 123 456 789 1024; do
    EVAL_DIRS+=("saves/unlearn/knowundo_BLADE_copyright_unified_s${SEED}/evals")
    EVAL_DIRS+=("saves/unlearn/knowundo_BLADE_privacy_unified_s${SEED}/evals")
done

echo "[$(date)] Starting KnowUnDo LLM Judge (Opus 4.6)"
echo "Total eval dirs: ${#EVAL_DIRS[@]}"

DONE=0
FAIL=0
for DIR in "${EVAL_DIRS[@]}"; do
    if [ ! -f "${DIR}/KnowUnDo_EVAL.json" ]; then
        echo "[SKIP] ${DIR} - no KnowUnDo_EVAL.json"
        ((FAIL++))
        continue
    fi
    echo "[$(date)] Judging: ${DIR}..."
    python scripts/llm_judge.py --eval-dir "${DIR}" --benchmark knowundo --max-workers 32 || {
        echo "[WARN] Failed: ${DIR}"
        ((FAIL++))
        continue
    }
    ((DONE++))
done

echo "[$(date)] DONE. Success: ${DONE}, Failed/Skipped: ${FAIL}"
