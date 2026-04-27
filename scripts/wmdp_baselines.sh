#!/bin/bash
# WMDP baselines — joint bio+cyber, single GPU, train + eval, skip-if-done
# Model: zephyr-7b-beta
# Data: balanced bio+cyber (1K bio + 1K cyber forget, 4.5K+4.5K retain)
# Eval: wmdp_bio accuracy (↓) + wmdp_cyber accuracy (↓) + mmlu average (↑)
# Usage: nohup bash scripts/wmdp_baselines.sh > saves/unlearn/wmdp_baselines.log 2>&1 &
#
# RMU: paper Zephyr notebook params (centerforaisafety/wmdp/run_rmu_zephyr.ipynb):
#   steering_coeff=6.5, alpha=1200, lr=5e-5, max_steps=150, bs=4
#   layers.5-7.mlp.down_proj, steering=layer.7
#
# GA, GradDiff, NPO, SimNPO: epoch-based (10 epochs), eff_bs=32, lr=1e-5
#
# BLURNPO: paper-correct WMDP params from arXiv:2506.08164 Table 7:
#   lr=2e-6, beta=0.005, gamma=1.0, bs=4, max_steps=150
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

# ── Config ──────────────────────────────────────────────────
DATA_SPLIT="bio_cyber"
MODEL="zephyr-7b-beta"
SEED=42
BSZ=4
ACCUM=8
# ────────────────────────────────────────────────────────────

total=0; skip=0; fail=0

# ── Helper: train + eval a single method ────────────────────
run_method() {
    local task_name=$1
    shift
    local outdir="saves/unlearn/${task_name}"
    total=$((total + 1))

    if [[ -f "$outdir/evals/LMEval_SUMMARY.json" ]]; then
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
        experiment=eval/wmdp/default.yaml \
        task_name=${task_name} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${outdir} \
        paths.output_dir=${outdir}/evals \
        'eval.lm_eval.tasks=[wmdp_bio,wmdp_cyber,mmlu]'; then
        echo "[EVAL FAILED] $task_name"
        fail=$((fail + 1))
        return 1
    fi

    echo "[DONE] $(date) $task_name"
    cat "$outdir/evals/LMEval_SUMMARY.json" 2>/dev/null || true

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

# ── Common args for standard methods (epoch-based) ──────────
# Note: wmdp/default.yaml defaults to trainer=RMU, so we must null out
# RMU-specific method_args when overriding to other trainers.
COMMON_ARGS=(
    experiment=unlearn/wmdp/default.yaml
    data_split=${DATA_SPLIT}
    trainer.args.per_device_train_batch_size=${BSZ}
    trainer.args.gradient_accumulation_steps=${ACCUM}
    trainer.args.gradient_checkpointing=true
    trainer.args.eval_strategy=no
    trainer.args.do_eval=false
    trainer.args.eval_on_start=false
    trainer.args.learning_rate=1e-5
    trainer.args.num_train_epochs=10
    trainer.args.max_steps=-1
    trainer.args.seed=${SEED}
    '~trainer.method_args'
)

echo ""
echo "################################################################"
echo "# WMDP Bio+Cyber Baselines — SEED=${SEED}"
echo "################################################################"

# ── Standard methods (override trainer on WMDP default config) ──
for trainer in GradAscent GradDiff NPO SimNPO; do
    task="wmdp_${MODEL}_${DATA_SPLIT}_${trainer}_s${SEED}"
    run_method "$task" "${COMMON_ARGS[@]}" trainer=${trainer} task_name=${task}
done

# ── RMU (Zephyr notebook params) ──────────────────────────────
# centerforaisafety/wmdp/run_rmu_zephyr.ipynb:
# steering_coeff=6.5, alpha=1200, lr=5e-5, max_steps=150, bs=4
TASK="wmdp_${MODEL}_${DATA_SPLIT}_RMU_s${SEED}"
run_method "$TASK" \
    experiment=unlearn/wmdp/default.yaml \
    data_split=${DATA_SPLIT} \
    task_name=${TASK} \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=2 \
    trainer.args.max_steps=150 \
    trainer.args.eval_strategy=no \
    trainer.args.do_eval=false \
    trainer.args.eval_on_start=false \
    trainer.args.seed=${SEED} \
    trainer.method_args.steering_coeff=6.5 \
    trainer.method_args.alpha=1200

# ── BLUR-NPO (paper-correct WMDP params) ────────────────────
# arXiv:2506.08164, Table 7: lr=2e-6, beta=0.005, bs=4, 150 steps
TASK="wmdp_${MODEL}_${DATA_SPLIT}_BLURNPO_s${SEED}"
run_method "$TASK" \
    experiment=unlearn/wmdp/default.yaml \
    data_split=${DATA_SPLIT} \
    trainer=BLURNPO \
    '~trainer.method_args' \
    task_name=${TASK} \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=2 \
    trainer.args.gradient_checkpointing=true \
    trainer.args.eval_strategy=no \
    trainer.args.do_eval=false \
    trainer.args.eval_on_start=false \
    trainer.args.learning_rate=2e-6 \
    trainer.args.max_steps=150 \
    trainer.args.num_train_epochs=-1 \
    trainer.args.lr_scheduler_type=constant \
    trainer.args.seed=${SEED} \
    trainer.method_args.beta=0.005

echo ""
echo "============================================================"
echo "ALL DONE at $(date) — total=${total} skip=${skip} fail=${fail}"
echo "============================================================"
