#!/bin/bash
# LLM Judge for MUSE BLUR (BLURNPO) multi-seed runs
# Uses Opus 4.7 via Bedrock
cd /datadrive/forked/open-unlearning
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export LLM_JUDGE_MODEL="eu.anthropic.claude-opus-4-7"

echo "[$(date)] Starting MUSE BLUR LLM Judge (Opus 4.7)"

# News seeds 123, 456, 789, 1024 (s42 already done)
for SEED in 123 456 789 1024; do
    DIR="saves/unlearn/muse_Llama-2-7b-hf_News_BLURNPO_s${SEED}/evals"
    if [ ! -f "${DIR}/MUSE_EVAL.json" ]; then
        echo "[SKIP] News s${SEED} - no MUSE_EVAL.json"
        continue
    fi
    echo "[$(date)] Judging BLUR News s${SEED}..."
    python scripts/llm_judge.py --eval-dir "${DIR}" --benchmark muse --max-workers 16 || {
        echo "[WARN] Failed: News s${SEED}"
        continue
    }
done

# Books seeds 123, 456, 789 (s42 already done, s1024 not trained yet)
for SEED in 123 456 789; do
    DIR="saves/unlearn/muse_Llama-2-7b-hf_Books_BLURNPO_s${SEED}/evals"
    if [ ! -f "${DIR}/MUSE_EVAL.json" ]; then
        echo "[SKIP] Books s${SEED} - no MUSE_EVAL.json"
        continue
    fi
    echo "[$(date)] Judging BLUR Books s${SEED}..."
    python scripts/llm_judge.py --eval-dir "${DIR}" --benchmark muse --max-workers 16 || {
        echo "[WARN] Failed: Books s${SEED}"
        continue
    }
done

echo "[$(date)] ALL MUSE BLUR JUDGE RUNS COMPLETE"
