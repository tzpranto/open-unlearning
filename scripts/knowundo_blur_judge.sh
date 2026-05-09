#!/bin/bash
# LLM Judge for KnowUnDo BLURNPO (completed models)
# Uses Opus 4.6 (Opus 4.7 refuses on copyrighted content)
cd /datadrive/forked/open-unlearning
export LLM_JUDGE_MODEL="us.anthropic.claude-opus-4-6-v1"
export AWS_REGION="us-east-1"

EVAL_DIRS=(
    "saves/unlearn/knowundo_BLURNPO_copyright_s42/evals"
    "saves/unlearn/knowundo_BLURNPO_copyright_s123/evals"
    "saves/unlearn/knowundo_BLURNPO_copyright_s456/evals"
    "saves/unlearn/knowundo_BLURNPO_copyright_s789/evals"
    "saves/unlearn/knowundo_BLURNPO_copyright_s1024/evals"
    "saves/unlearn/knowundo_BLURNPO_privacy_s42/evals"
    "saves/unlearn/knowundo_BLURNPO_privacy_s123/evals"
    "saves/unlearn/knowundo_BLURNPO_privacy_s456/evals"
)

echo "[$(date)] Starting KnowUnDo BLURNPO LLM Judge (Opus 4.6)"
echo "Models to judge: ${#EVAL_DIRS[@]}"

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
