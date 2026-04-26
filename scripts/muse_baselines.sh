#!/bin/bash
# MUSE baselines — single GPU, train + eval, skip-if-done, multi-seed
# Usage: bash scripts/muse_baselines.sh
# Configure DATA_SPLIT and SEEDS below. Skips any run that already has MUSE_SUMMARY.json.
#
# GA, GradDiff, NPO, SimNPO use default.yaml + trainer override (upstream params).
# RMU, BLURNPO, PDU use dedicated experiment configs with paper-correct overrides.
# All paper references are documented in the experiment YAML files.
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

# ── Config ──────────────────────────────────────────────────
DATA_SPLIT="News"    # "News" or "Books"
MODEL="Llama-2-7b-hf"
SEEDS=(42)
BSZ=4
ACCUM=8
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
# ────────────────────────────────────────────────────────────

# ── Split-specific overrides ────────────────────────────────
# PDU: eps=1.5 for News, eps=0.1 for Books (arXiv:2506.05314, community/methods/PDU/run.sh)
# BLUR: lr=2.5e-5 for News, lr=1e-5 for Books (arXiv:2506.08164, MUSE/baselines/)
if [[ "$DATA_SPLIT" == "News" ]]; then
    PDU_EPS=1.5
    BLUR_LR="2.5e-5"
elif [[ "$DATA_SPLIT" == "Books" ]]; then
    PDU_EPS=0.1
    BLUR_LR="1e-5"
else
    echo "Invalid DATA_SPLIT: $DATA_SPLIT (must be News or Books)"
    exit 1
fi
# ────────────────────────────────────────────────────────────

total=0; skip=0; fail=0

# ── Helper: train + eval a single method ────────────────────
run_method() {
    local task_name=$1
    shift
    local outdir="saves/unlearn/${task_name}"
    total=$((total + 1))

    if [[ -f "$outdir/evals/MUSE_SUMMARY.json" ]]; then
        echo "[SKIP] $task_name"
        skip=$((skip + 1))
        return 0
    fi

    echo "============================================================"
    echo "[TRAIN] $(date) $task_name"
    if ! CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml "$@"; then
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

# ── Main loop over seeds ────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "################################################################"
    echo "# SEED=${SEED}  DATA_SPLIT=${DATA_SPLIT}"
    echo "################################################################"

    # ── Standard methods (default.yaml + trainer override) ──
    COMMON_ARGS=(
        experiment=unlearn/muse/default.yaml
        model=${MODEL}
        data_split=${DATA_SPLIT}
        retain_logs_path=${RETAIN_LOGS}
        trainer.args.per_device_train_batch_size=${BSZ}
        trainer.args.gradient_accumulation_steps=${ACCUM}
        trainer.args.gradient_checkpointing=true
        trainer.args.eval_strategy=no
        trainer.args.do_eval=false
        trainer.args.eval_on_start=false
        trainer.args.seed=${SEED}
    )

    for trainer in GradAscent GradDiff NPO SimNPO; do
        task="muse_${MODEL}_${DATA_SPLIT}_${trainer}_s${SEED}"
        run_method "$task" "${COMMON_ARGS[@]}" trainer=${trainer} task_name=${task}
    done

    # ── RMU (dedicated config: configs/experiment/unlearn/muse/rmu.yaml) ────
    # open-unlearning repro params: steering_coeff=2, alpha=1, lr=5e-5,
    # trainable_params=layers.5-7.mlp.down_proj (not all params).
    TASK="muse_${MODEL}_${DATA_SPLIT}_RMU_s${SEED}"
    run_method "$TASK" \
        experiment=unlearn/muse/rmu.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${TASK} \
        retain_logs_path=${RETAIN_LOGS} \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.seed=${SEED}

    # ── BLUR-NPO (dedicated config: configs/experiment/unlearn/muse/blurnpo.yaml) ──
    # arXiv:2506.08164: beta=0.15 (hardcoded in their code), lr varies by split.
    # NOTE: BLUR needs ref_model deepcopy of 7B — may OOM on single GPU.
    TASK="muse_${MODEL}_${DATA_SPLIT}_BLURNPO_s${SEED}"
    run_method "$TASK" \
        experiment=unlearn/muse/blurnpo.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${TASK} \
        retain_logs_path=${RETAIN_LOGS} \
        trainer.args.learning_rate=${BLUR_LR} \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.seed=${SEED}

    # ── PDU (dedicated config: configs/experiment/unlearn/muse/pdu.yaml) ────
    # arXiv:2506.05314: alpha=50, eps differs by split: News=1.5, Books=0.1 (7B).
    TASK="muse_${MODEL}_${DATA_SPLIT}_PDU_s${SEED}"
    run_method "$TASK" \
        experiment=unlearn/muse/pdu.yaml \
        data_split=${DATA_SPLIT} \
        task_name=${TASK} \
        retain_logs_path=${RETAIN_LOGS} \
        trainer.method_args.retain_loss_eps=${PDU_EPS} \
        trainer.args.per_device_train_batch_size=2 \
        trainer.args.gradient_accumulation_steps=16 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.seed=${SEED}

    echo ""
    echo "[SEED ${SEED} DONE] total=${total} skip=${skip} fail=${fail}"
done

echo ""
echo "============================================================"
echo "ALL DONE at $(date) — total=${total} skip=${skip} fail=${fail}"
echo "============================================================"
