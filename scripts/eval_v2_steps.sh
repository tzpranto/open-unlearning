#!/bin/bash
# Eval v2 Phase 1 intermediate checkpoints on TOFU
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

BASE="saves/unlearn/tofu_1b_v2_p1"
RETAIN_LOGS="saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json"

# Eval key checkpoints (every 2 steps early, sparser later)
for STEP in 2 4 6 8 10 12 16 20 24 28 32; do
    CKPT="$BASE/step-$STEP"
    EVAL_DIR="$CKPT/evals"

    if [[ -f "$EVAL_DIR/TOFU_SUMMARY.json" ]]; then
        echo "=== step-$STEP === (cached)"
        cat "$EVAL_DIR/TOFU_SUMMARY.json"
        echo
        continue
    fi

    if [[ ! -f "$CKPT/model.safetensors" ]]; then
        echo "=== step-$STEP === MISSING"
        continue
    fi

    echo "[EVAL] step-$STEP"
    python src/eval.py \
        model=Llama-3.2-1B-Instruct \
        model.model_args.pretrained_model_name_or_path="$CKPT" \
        eval=tofu \
        eval.tofu.forget_split=forget01 \
        eval.tofu.holdout_split=holdout01 \
        eval.tofu.retain_logs_path="$RETAIN_LOGS" \
        eval.tofu.overwrite=true \
        task_name="tofu_1b_v2_step${STEP}" \
        2>&1 | grep -E "Result for metric|Evaluated"

    # Move eval results to checkpoint directory
    EVAL_SRC="saves/eval/tofu_1b_v2_step${STEP}/evals"
    if [[ -d "$EVAL_SRC" ]]; then
        mv "$EVAL_SRC" "$EVAL_DIR"
        echo "=== step-$STEP ==="
        cat "$EVAL_DIR/TOFU_SUMMARY.json"
        echo
    fi
done
