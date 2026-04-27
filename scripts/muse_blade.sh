#!/bin/bash
# BLADE on MUSE — Books + News, multi-seed
# Train + eval, skip-if-done (MUSE_SUMMARY.json check)
# Usage: nohup bash scripts/muse_blade.sh > saves/unlearn/muse_blade.log 2>&1 &
# Configure DATA_SPLIT and SEEDS below.
#
# Books: eps_multiplier=3.2 (~60 min, converges ~step 78)
# News:  eps_multiplier=3.2 (TBD — tune after baselines)
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

# ── Config (override via env: DATA_SPLIT=News bash scripts/muse_blade.sh) ──
DATA_SPLIT="${DATA_SPLIT:-Books}"      # "Books" or "News"
MODEL="${MODEL:-Llama-2-7b-hf}"
SEEDS=(${SEEDS:-42})
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
# ────────────────────────────────────────────────────────────

# ── Split-specific experiment config ────────────────────────
if [[ "$DATA_SPLIT" == "Books" ]]; then
    EXP_CONFIG="unlearn/muse/lora_bial_adaptive_books.yaml"
elif [[ "$DATA_SPLIT" == "News" ]]; then
    # TODO: create lora_bial_adaptive_news.yaml after tuning
    EXP_CONFIG="unlearn/muse/lora_bial_adaptive_books.yaml"
    echo "[WARN] Using Books config for News — tune eps_multiplier after baselines"
else
    echo "Invalid DATA_SPLIT: $DATA_SPLIT (must be Books or News)"
    exit 1
fi
# ────────────────────────────────────────────────────────────

total=0; skip=0; fail=0

run_adaptive() {
    local seed=$1
    local task_name="muse_${MODEL}_${DATA_SPLIT}_adaptive_s${seed}"
    local outdir="saves/unlearn/${task_name}"
    total=$((total + 1))

    if [[ -f "$outdir/evals/MUSE_SUMMARY.json" ]]; then
        echo "[SKIP] $task_name"
        skip=$((skip + 1))
        return 0
    fi

    echo "============================================================"
    echo "[TRAIN] $(date) $task_name"
    if ! CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
        experiment=${EXP_CONFIG} \
        data_split=${DATA_SPLIT} \
        task_name=${task_name} \
        retain_logs_path=${RETAIN_LOGS} \
        trainer.args.seed=${seed}; then
        echo "[TRAIN FAILED] $task_name"
        fail=$((fail + 1))
        return 1
    fi

    echo "[EVAL] $(date) $task_name"
    if ! CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${task_name} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${outdir} \
        paths.output_dir=${outdir}/evals \
        retain_logs_path=${RETAIN_LOGS}; then
        echo "[EVAL FAILED] $task_name"
        fail=$((fail + 1))
        return 1
    fi

    echo "[DONE] $(date) $task_name"
    cat "$outdir/evals/MUSE_SUMMARY.json" 2>/dev/null || true

    # Clean model weights — keep only evals/ .hydra/ logs/
    rm -rf "$outdir"/checkpoint-* 2>/dev/null
    rm -f "$outdir"/model*.safetensors "$outdir"/model.safetensors.index.json 2>/dev/null
    rm -f "$outdir"/pytorch_model* "$outdir"/config.json "$outdir"/generation_config* 2>/dev/null
    rm -f "$outdir"/tokenizer* "$outdir"/special_tokens* "$outdir"/added_tokens* 2>/dev/null
    rm -f "$outdir"/optimizer* "$outdir"/scheduler* "$outdir"/training_args* 2>/dev/null
    find "$outdir" -maxdepth 1 -name "*.safetensors" -delete 2>/dev/null
    find "$outdir" -maxdepth 1 -name "*.bin" -delete 2>/dev/null
    echo "[CLEANED] $task_name — kept evals only"
}

# ── Main loop ──────────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "################################################################"
    echo "# SEED=${SEED}  DATA_SPLIT=${DATA_SPLIT}"
    echo "################################################################"
    run_adaptive "$SEED"
done

echo ""
echo "============================================================"
echo "ALL DONE at $(date) — total=${total} skip=${skip} fail=${fail}"
echo "============================================================"
