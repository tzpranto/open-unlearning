#!/bin/bash
# Unified LLM Judge runner for TOFU and MUSE baselines and our method
# Usage:
#   # TOFU (default)
#   bash scripts/llm_judge_run.sh --model 1B --splits "forget01 forget05 forget10"
#   bash scripts/llm_judge_run.sh --model 3B --splits "forget05"
#   bash scripts/llm_judge_run.sh --model 1B --splits "forget01" --methods "PDU SimNPO" --seeds "42 123"
#   bash scripts/llm_judge_run.sh --model 1B --splits "forget01" --ours   # judge our adaptive runs
#   # MUSE
#   bash scripts/llm_judge_run.sh --benchmark muse --eval-dirs "saves/eval/muse_books_exp_17_epoch1 saves/eval/muse_books_exp_16_epoch1_eval"
#   bash scripts/llm_judge_run.sh --benchmark muse --model Llama-2-7b-hf --splits "Books" --methods "GradAscent GradDiff" --seeds "42"
#
# nohup bash scripts/llm_judge_run.sh --model 3B --splits "forget05 forget10" > saves/unlearn/llm_judge_run.log 2>&1 &
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
cd /datadrive/forked/open-unlearning

# ── Defaults ──────────────────────────────────────────────
BENCHMARK="tofu"
MODEL_SHORT=""
SPLITS=""
METHODS="GradAscent GradDiff NPO SimNPO RMU BLURNPO PDU"
SEEDS="42 123 456 789 1337"
OURS=false
OUTPUT_CSV=""
MAX_WORKERS=4
EVAL_DIRS=""  # For MUSE: explicit list of eval directories

# ── Parse args ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --model)     MODEL_SHORT="$2"; shift 2 ;;
        --splits)    SPLITS="$2"; shift 2 ;;
        --methods)   METHODS="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --ours)      OURS=true; shift ;;
        --output)    OUTPUT_CSV="$2"; shift 2 ;;
        --workers)   MAX_WORKERS="$2"; shift 2 ;;
        --eval-dirs) EVAL_DIRS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Default output CSV ────────────────────────────────────
if [[ -z "$OUTPUT_CSV" ]]; then
    if [[ "$BENCHMARK" == "muse" ]]; then
        OUTPUT_CSV="results/muse_llm_judge.csv"
    else
        OUTPUT_CSV="results/tofu_llm_judge.csv"
    fi
fi

# ── Validate args ─────────────────────────────────────────
if [[ "$BENCHMARK" == "muse" && -n "$EVAL_DIRS" ]]; then
    # MUSE explicit eval-dirs mode — model/splits not required
    :
elif [[ -z "$MODEL_SHORT" || -z "$SPLITS" ]]; then
    echo "Usage:"
    echo "  TOFU: $0 --model {1B|3B} --splits \"forget01 ...\" [--methods \"...\"] [--seeds \"...\"] [--ours]"
    echo "  MUSE: $0 --benchmark muse --eval-dirs \"dir1 dir2 ...\""
    echo "  MUSE: $0 --benchmark muse --model Llama-2-7b-hf --splits \"Books News\" [--methods \"...\"]"
    exit 1
fi

# ── Resolve model name ────────────────────────────────────
MODEL=""
if [[ -n "$MODEL_SHORT" ]]; then
    case "$MODEL_SHORT" in
        1B) MODEL="Llama-3.2-1B-Instruct" ;;
        3B) MODEL="Llama-3.2-3B-Instruct" ;;
        *)  MODEL="$MODEL_SHORT" ;;
    esac
fi

# ── CSV setup ─────────────────────────────────────────────
if [[ ! -f "$OUTPUT_CSV" ]]; then
    if [[ "$BENCHMARK" == "muse" ]]; then
        echo "model,split,method,seed,forget_leakage,forget_leakage_knowmem,forget_leakage_verbmem,retain_accuracy,response_quality,forget_rq,retain_rq,n_forget_knowmem,n_forget_verbmem,n_retain" > "$OUTPUT_CSV"
    else
        echo "model,split,method,seed,forget_leakage,retain_accuracy,response_quality,forget_rq,retain_rq,n_forget,n_retain" > "$OUTPUT_CSV"
    fi
fi

is_done() {
    local model=$1 split=$2 method=$3 seed=$4
    grep -q "^${model},${split},${method},${seed}," "$OUTPUT_CSV" 2>/dev/null
}

judge_one_tofu() {
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

judge_one_muse() {
    local eval_json=$1 model=$2 split=$3 method=$4 seed=$5

    result=$(python3 << PYEOF
import json, sys
sys.path.insert(0, "scripts")
from llm_judge import get_client, evaluate_muse_eval

client = get_client()
scores = evaluate_muse_eval(client, "$eval_json", max_workers=$MAX_WORKERS)

row = f"${model},${split},${method},${seed},{scores['forget_leakage']},{scores['forget_leakage_knowmem']},{scores['forget_leakage_verbmem']},{scores['retain_accuracy']},{scores['response_quality']},{scores['forget_rq']},{scores['retain_rq']},{scores['n_forget_knowmem']},{scores['n_forget_verbmem']},{scores['n_retain']}"
print(row)
print(f"FL={scores['forget_leakage']:.3f} FL_km={scores['forget_leakage_knowmem']:.3f} FL_vm={scores['forget_leakage_verbmem']:.3f} RA={scores['retain_accuracy']:.3f} RQ={scores['response_quality']:.3f}", file=sys.stderr)
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

if [[ "$BENCHMARK" == "muse" && -n "$EVAL_DIRS" ]]; then
    # ── MUSE: explicit eval directories mode ──────────────
    for eval_dir in $EVAL_DIRS; do
        total=$((total + 1))
        eval_json="${eval_dir}/MUSE_EVAL.json"
        # Use directory basename as method name
        dir_name=$(basename "$eval_dir")
        method="$dir_name"
        model="${MODEL:-unknown}"
        split="unknown"
        seed="0"

        if [[ ! -f "$eval_json" ]]; then
            echo "[MISSING] $eval_json"
            missing=$((missing + 1))
            continue
        fi

        if is_done "$model" "$split" "$method" "$seed"; then
            echo "[SKIP] $model $split $method s$seed"
            skip=$((skip + 1))
            continue
        fi

        echo "════════════════════════════════════════"
        echo "[JUDGE-MUSE] $eval_dir"
        echo "════════════════════════════════════════"

        judge_one_muse "$eval_json" "$model" "$split" "$method" "$seed" && done_count=$((done_count + 1))
        echo ""
    done
elif [[ "$BENCHMARK" == "muse" ]]; then
    # ── MUSE: structured eval directories mode ────────────
    for split in $SPLITS; do
        if $OURS; then
            for seed in $SEEDS; do
                total=$((total + 1))
                method="BLADE"

                if is_done "$MODEL" "$split" "$method" "$seed"; then
                    echo "[SKIP] $MODEL $split $method s$seed"
                    skip=$((skip + 1))
                    continue
                fi

                eval_json="saves/unlearn/muse_${MODEL}_${split}_adaptive_s${seed}/evals/MUSE_EVAL.json"
                if [[ ! -f "$eval_json" ]]; then
                    echo "[MISSING] $eval_json"
                    missing=$((missing + 1))
                    continue
                fi

                echo "════════════════════════════════════════"
                echo "[JUDGE-MUSE] $MODEL / $split / $method / s$seed"
                echo "════════════════════════════════════════"

                judge_one_muse "$eval_json" "$MODEL" "$split" "$method" "$seed" && done_count=$((done_count + 1))
                echo ""
            done
        else
            for method in $METHODS; do
                for seed in $SEEDS; do
                    total=$((total + 1))

                    if is_done "$MODEL" "$split" "$method" "$seed"; then
                        echo "[SKIP] $MODEL $split $method s$seed"
                        skip=$((skip + 1))
                        continue
                    fi

                    eval_json="saves/unlearn/muse_${MODEL}_${split}_${method}_s${seed}/evals/MUSE_EVAL.json"
                    if [[ ! -f "$eval_json" ]]; then
                        echo "[MISSING] $eval_json"
                        missing=$((missing + 1))
                        continue
                    fi

                    echo "════════════════════════════════════════"
                    echo "[JUDGE-MUSE] $MODEL / $split / $method / s$seed"
                    echo "════════════════════════════════════════"

                    judge_one_muse "$eval_json" "$MODEL" "$split" "$method" "$seed" && done_count=$((done_count + 1))
                    echo ""
                done
            done
        fi
    done
else
    # ── TOFU mode (original) ──────────────────────────────
    for split in $SPLITS; do
        if $OURS; then
            # Judge our adaptive runs
            for seed in $SEEDS; do
                total=$((total + 1))
                method="BLADE"

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

                judge_one_tofu "$eval_json" "$MODEL" "$split" "$method" "$seed" && done_count=$((done_count + 1))
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

                    judge_one_tofu "$eval_json" "$MODEL" "$split" "$method" "$seed" && done_count=$((done_count + 1))
                    echo ""
                done
            done
        fi
    done
fi

echo ""
echo "════════════════════════════════════════"
echo "ALL DONE: total=$total skip=$skip done=$done_count missing=$missing"
echo "Results: $OUTPUT_CSV"
echo "════════════════════════════════════════"
