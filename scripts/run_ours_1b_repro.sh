#!/bin/bash
# LoRA-BiAL: 1B repro + 3B tuning on TOFU forget01
set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_and_eval() {
    local config="$1"; shift
    local model_name="$1"; shift
    local task="$1"; shift
    local retain_logs="$1"; shift
    local outdir="saves/unlearn/$task"

    if [[ -f "$outdir/evals/TOFU_SUMMARY.json" ]]; then
        log "[SKIP] $task — already done"
        cat "$outdir/evals/TOFU_SUMMARY.json"
        return 0
    fi

    log "[TRAIN] $task"
    python src/train.py --config-name=unlearn.yaml \
        experiment="$config" \
        task_name="$task" \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        retain_logs_path="$retain_logs" \
        trainer.method_args.checkpoint_every_epoch=false \
        "$@"

    log "[EVAL] $task"
    python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=forget01 \
        holdout_split=holdout01 \
        model="$model_name" \
        task_name="$task" \
        model.model_args.pretrained_model_name_or_path="$outdir" \
        paths.output_dir="$outdir/evals" \
        retain_logs_path="$retain_logs"

    log "[RESULT] $task"
    cat "$outdir/evals/TOFU_SUMMARY.json"
}

RETAIN_1B="saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json"
RETAIN_3B="saves/eval/tofu_Llama-3.2-3B-Instruct_retain99/TOFU_EVAL.json"

# ═══════════════════════════════════════════════════════════════
# 1. Eval original exp_03 (already trained, just needs eval)
# ═══════════════════════════════════════════════════════════════
log "═══ 1B: Eval original exp_03 ═══"
mkdir -p saves/unlearn/tofu_1b_01_exp_03/evals
if [[ ! -f saves/unlearn/tofu_1b_01_exp_03/evals/TOFU_SUMMARY.json ]]; then
    python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=forget01 holdout_split=holdout01 \
        model=Llama-3.2-1B-Instruct \
        task_name=tofu_1b_01_exp_03 \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/tofu_1b_01_exp_03 \
        paths.output_dir=saves/unlearn/tofu_1b_01_exp_03/evals \
        retain_logs_path="$RETAIN_1B"
fi
log "[RESULT] exp_03 original:"
cat saves/unlearn/tofu_1b_01_exp_03/evals/TOFU_SUMMARY.json

# ═══════════════════════════════════════════════════════════════
# 2. 1B repro (fresh train + eval)
# ═══════════════════════════════════════════════════════════════
log "═══ 1B: Repro run ═══"
run_and_eval "unlearn/tofu/lora_bial_1b.yaml" "Llama-3.2-1B-Instruct" \
    "tofu_Llama-3.2-1B-Instruct_forget01_LoRABiAL_bs16" "$RETAIN_1B" \
    trainer.method_args.eta_theta=2e-5 \
    trainer.method_args.T=100

# ═══════════════════════════════════════════════════════════════
# 3. 3B tuning: sweep eta_theta with same recipe
# ═══════════════════════════════════════════════════════════════
log "═══ 3B: Tuning sweep ═══"

# Start with 1B champion LR (2e-5), T=100
run_and_eval "unlearn/tofu/lora_bial_3b.yaml" "Llama-3.2-3B-Instruct" \
    "tofu_Llama-3.2-3B-Instruct_forget01_LoRABiAL_lr2e5" "$RETAIN_3B" \
    trainer.method_args.eta_theta=2e-5 \
    trainer.method_args.T=100

# Slightly higher LR
run_and_eval "unlearn/tofu/lora_bial_3b.yaml" "Llama-3.2-3B-Instruct" \
    "tofu_Llama-3.2-3B-Instruct_forget01_LoRABiAL_lr3e5" "$RETAIN_3B" \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.T=100

# Slightly lower LR
run_and_eval "unlearn/tofu/lora_bial_3b.yaml" "Llama-3.2-3B-Instruct" \
    "tofu_Llama-3.2-3B-Instruct_forget01_LoRABiAL_lr1e5" "$RETAIN_3B" \
    trainer.method_args.eta_theta=1e-5 \
    trainer.method_args.T=100

log "═══ ALL DONE ═══"
