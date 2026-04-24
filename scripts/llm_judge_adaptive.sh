#!/bin/bash
# LLM Judge for LoRA-BiAL-Adaptive multi-seed runs
# Requires: OPENAI_BASE_URL + OPENAI_API_KEY (or AWS credentials for Bedrock)
# Usage: nohup bash scripts/llm_judge_adaptive.sh > saves/unlearn/llm_judge_adaptive.log 2>&1 &
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
cd /datadrive/forked/open-unlearning

MODEL="Llama-3.2-1B-Instruct"
SEEDS=(42 123 456 789 1337)
SPLITS=("forget01" "forget05" "forget10")
OUTPUT_CSV="docs/results/tofu_adaptive_llm_judge.csv"

# Create output CSV if needed
if [[ ! -f "$OUTPUT_CSV" ]]; then
    echo "model,split,method,seed,forget_leakage,retain_accuracy,response_quality,forget_rq,retain_rq,n_forget,n_retain" > "$OUTPUT_CSV"
fi

is_done() {
    local split=$1 seed=$2
    grep -q "^${MODEL},${split},LoRA-BiAL-Adaptive,${seed}," "$OUTPUT_CSV" 2>/dev/null
}

total=0; skip=0; done_count=0

for split in "${SPLITS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        total=$((total + 1))

        if is_done "$split" "$seed"; then
            echo "[SKIP] $split s$seed already in CSV"
            skip=$((skip + 1))
            continue
        fi

        eval_dir="saves/unlearn/adaptive_${MODEL}_${split}_s${seed}/evals"
        eval_json="${eval_dir}/TOFU_EVAL.json"

        if [[ ! -f "$eval_json" ]]; then
            echo "[MISSING] $eval_json"
            continue
        fi

        echo "════════════════════════════════════════"
        echo "[JUDGE] $MODEL / $split / s$seed"
        echo "════════════════════════════════════════"

        # Run judge and capture output
        result=$(python3 << PYEOF
import json, os, sys
sys.path.insert(0, "scripts")
from llm_judge import get_client, evaluate_single_eval

client = get_client()
scores = evaluate_single_eval(client, "$eval_json", max_workers=4)

row = f"${MODEL},${split},LoRA-BiAL-Adaptive,${seed},{scores['forget_leakage']},{scores['retain_accuracy']},{scores['response_quality']},{scores['forget_rq']},{scores['retain_rq']},{scores['n_forget']},{scores['n_retain']}"
print(row)
print(f"FL={scores['forget_leakage']:.3f} RA={scores['retain_accuracy']:.3f} ret_RQ={scores['retain_rq']:.3f} fgt_RQ={scores['forget_rq']:.3f}", file=sys.stderr)
PYEOF
)

        if [[ -n "$result" ]]; then
            echo "$result" >> "$OUTPUT_CSV"
            echo "[RESULT] $result"
            done_count=$((done_count + 1))
        else
            echo "[ERROR] Judge failed for $split s$seed"
        fi

        echo ""
    done
done

echo ""
echo "════════════════════════════════════════"
echo "ALL DONE: total=$total skip=$skip done=$done_count"
echo "Results: $OUTPUT_CSV"
echo "════════════════════════════════════════"
cat "$OUTPUT_CSV"
