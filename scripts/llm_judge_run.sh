#!/bin/bash
# Unified LLM Judge runner for TOFU baselines and our method
# Usage:
#   bash scripts/llm_judge_run.sh --model 1B --splits "forget01 forget05 forget10"
#   bash scripts/llm_judge_run.sh --model 3B --splits "forget05"
#   bash scripts/llm_judge_run.sh --model 1B --splits "forget01" --methods "PDU SimNPO" --seeds "42 123"
#   bash scripts/llm_judge_run.sh --model 1B --splits "forget01" --ours   # judge our adaptive runs
#
# nohup bash scripts/llm_judge_run.sh --model 3B --splits "forget05 forget10" > saves/unlearn/llm_judge_run.log 2>&1 &
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
cd /datadrive/forked/open-unlearning

# ── Defaults ──────────────────────────────────────────────
MODEL_SHORT=""
SPLITS=""
METHODS="GradAscent GradDiff NPO SimNPO RMU BLURNPO PDU"
SEEDS="42 123 456 789 1337"
OURS=false
OUTPUT_CSV="docs/results/tofu_llm_judge.csv"
MAX_WORKERS=4

# ── Parse args ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)   MODEL_SHORT="$2"; shift 2 ;;
        --splits)  SPLITS="$2"; shift 2 ;;
        --methods) METHODS="$2"; shift 2 ;;
        --seeds)   SEEDS="$2"; shift 2 ;;
        --ours)    OURS=true; shift ;;
        --output)  OUTPUT_CSV="$2"; shift 2 ;;
        --workers) MAX_WORKERS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_SHORT" || -z "$SPLITS" ]]; then
    echo "Usage: $0 --model {1B|3B} --splits \"forget01 forget05 ...\" [--methods \"...\"] [--seeds \"...\"] [--ours]"
    exit 1
fi

# ── Resolve model name ────────────────────────────────────
case "$MODEL_SHORT" in
    1B) MODEL="Llama-3.2-1B-Instruct" ;;
    3B) MODEL="Llama-3.2-3B-Instruct" ;;
    *)  MODEL="$MODEL_SHORT" ;;
esac

# ── CSV setup ─────────────────────────────────────────────
if [[ ! -f "$OUTPUT_CSV" ]]; then
    echo "model,split,method,seed,forget_leakage,retain_accuracy,response_quality,forget_rq,retain_rq,n_forget,n_retain" > "$OUTPUT_CSV"
fi

is_done() {
    local model=$1 split=$2 method=$3 seed=$4
    grep -q "^${model},${split},${method},${seed}," "$OUTPUT_CSV" 2>/dev/null
}

judge_one() {
    local eval_json=$1 model=$2 split=$3 method=$4 seed=$5

    result=$(python3 << PYEOF
import json, sys
sys.path.insert(0, "scripts")
from llm_judge import get_client, evaluate_single_eval

client = get_client()
scores = evaluate_single_eval(client, "$eval_json", max_workers=$MAX_WORKERS)

row = f"${model},${split},${method},${seed},{scores['forget_leakage']},{scores['retain_accuracy']},{scores['response_quality']},{scores['forget_rq']},{scores['retain_rq']},{scores['n_forget']},{scores['n_retain']}"
print(row)
print(f"FL={scores['forget_leakage']:.3f} RA={scores['retain_accuracy']:.3f} ret_RQ={scores['retain_rq']:.3f} fgt_RQ={scores['forget_rq']:.3f}", file=sys.stderr)
PYEOF
)

    if [[ -n "$result" ]]; then
        echo "$result" >> "$OUTPUT_CSV"
        echo "[RESULT] $result"
        return 0
    else
        echo "[ERROR] Judge failed for $split $method s$seed"
        return 1
    fi
}

# ── Main loop ─────────────────────────────────────────────
total=0; skip=0; done_count=0; missing=0

for split in $SPLITS; do
    if $OURS; then
        # Judge our adaptive runs
        for seed in $SEEDS; do
            total=$((total + 1))
            method="LoRA-BiAL-Adaptive"

            if is_done "$MODEL" "$split" "$method" "$seed"; then
                echo "[SKIP] $MODEL $split $method s$seed"
                skip=$((skip + 1))
                continue
            fi

            eval_json="saves/unlearn/adaptive_${MODEL}_${split}_s${seed}/evals/TOFU_EVAL.json"
            if [[ ! -f "$eval_json" ]]; then
                echo "[MISSING] $eval_json"
                missing=$((missing + 1))
                continue
            fi

            echo "════════════════════════════════════════"
            echo "[JUDGE] $MODEL / $split / $method / s$seed"
            echo "════════════════════════════════════════"

            judge_one "$eval_json" "$MODEL" "$split" "$method" "$seed" && done_count=$((done_count + 1))
            echo ""
        done
    else
        # Judge baselines
        for method in $METHODS; do
            for seed in $SEEDS; do
                total=$((total + 1))

                if is_done "$MODEL" "$split" "$method" "$seed"; then
                    echo "[SKIP] $MODEL $split $method s$seed"
                    skip=$((skip + 1))
                    continue
                fi

                eval_json="saves/unlearn/bs32_${MODEL}_${split}_${method}_s${seed}/evals/TOFU_EVAL.json"
                if [[ ! -f "$eval_json" ]]; then
                    echo "[MISSING] $eval_json"
                    missing=$((missing + 1))
                    continue
                fi

                echo "════════════════════════════════════════"
                echo "[JUDGE] $MODEL / $split / $method / s$seed"
                echo "════════════════════════════════════════"

                judge_one "$eval_json" "$MODEL" "$split" "$method" "$seed" && done_count=$((done_count + 1))
                echo ""
            done
        done
    fi
done

echo ""
echo "════════════════════════════════════════"
echo "ALL DONE: total=$total skip=$skip done=$done_count missing=$missing"
echo "Results: $OUTPUT_CSV"
echo "════════════════════════════════════════"
