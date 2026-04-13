#!/bin/bash
# Ablation Series C: ALM hyperparameter tuning for MUSE News / Llama-2-7b
# Problem: A1a shows L_fgt plateaus at ~17.5 because retain constraint (ε=0.1) is too tight.
# The inner loop (K=10 steps) pulls model back too strongly after each outer forget step.
# Solution: Relax ε, reduce K, increase η_θ, then recover retain with post-inner steps.
#
# Experiments:
#   C1: ε=0.8, K=5, η_θ=2e-4, ρ=0.3  (relaxed ALM, test forgetting improvement)
#   C2: C1 + post_inner_steps=50  (relaxed forget + retain recovery)
#   C3: C1 + mask(th=1.1) + retain_protect(layer31)  (relaxed + surgical)
#   C4: ε=0.5, K=3, η_θ=3e-4, ρ=0.2 + post_inner=100  (very aggressive forget + long recovery)
#   C5: C1 + no implicit correction  (test if Neumann hurts forgetting)
#
# Date: 2026-04-08
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
PROGRESS="/tmp/ablation_C_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting Ablation Series C" | tee "$PROGRESS"
echo "Gold: fk<=0.328 rk>=0.560 fv<=0.202 ex<=0.024 | A1a baseline: fk=0.524 rk=0.500" | tee -a "$PROGRESS"

print_metrics() {
    local EVAL_FILE=$1
    "${PYTHON_BIN}" -c "
import json
with open('$EVAL_FILE') as f: d = json.load(f)
def agg(d, k):
    v = d.get(k, {})
    return v.get('agg_value', v) if isinstance(v, dict) else v
fk=agg(d,'forget_knowmem_ROUGE'); rk=agg(d,'retain_knowmem_ROUGE')
fv=agg(d,'forget_verbmem_ROUGE'); ex=agg(d,'extraction_strength')
pl=agg(d,'privleak')
print(f'  fk={fk:.4f} rk={rk:.4f} fv={fv:.4f} ex={ex:.4f} pl={pl:.2f}')
" 2>/dev/null || echo "  [metrics parse failed]"
}

run_eval() {
    local TASK=$1
    echo "[$(date '+%H:%M:%S')] Evaluating ${TASK}" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/eval.py \
        experiment=eval/muse/default.yaml data_split=${DATA_SPLIT} \
        task_name=${TASK} model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK} \
        "${ATTN_ARGS[@]}" \
        paths.output_dir=saves/unlearn/${TASK}/evals \
        retain_logs_path=${RETAIN_LOGS} \
        2>&1 | tee /tmp/eval_${TASK}.log
    local EVAL_FILE="saves/unlearn/${TASK}/evals/MUSE_EVAL.json"
    [ -f "$EVAL_FILE" ] && print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"
}

run_sibl() {
    local TASK=$1 EXPERIMENT=$2
    local TASK_DIR="saves/unlearn/${TASK}"

    if compgen -G "${TASK_DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] ${TASK}: weights exist" | tee -a "$PROGRESS"
        local EVAL_FILE="${TASK_DIR}/evals/MUSE_EVAL.json"
        if [ ! -f "$EVAL_FILE" ]; then
            run_eval ${TASK}
        else
            echo "[CACHED]" | tee -a "$PROGRESS"
            print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"
        fi
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] [TRAIN] ${TASK}" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=${EXPERIMENT} model=${MODEL} data_split=${DATA_SPLIT} \
        trainer=SIBL task_name=${TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_accumulation_steps=32 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.num_train_epochs=3 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        "${ATTN_ARGS[@]}" \
        2>&1 | tee /tmp/train_${TASK}.log

    local STATUS=$?
    [ $STATUS -ne 0 ] && echo "[WARN] ${TASK} failed (exit $STATUS)" | tee -a "$PROGRESS" && return $STATUS
    run_eval ${TASK}
}

# ─── C1: Relaxed ALM (ε=0.8, K=5, η_θ=2e-4, ρ=0.3) ─────────────────────────
echo "=== [C1] relaxed ALM: eps=0.8 K=5 eta=2e-4 rho=0.3 ===" | tee -a "$PROGRESS"
run_sibl ablation_C1_relaxed unlearn/muse/ablation_C1_relaxed \
    || echo "[WARN] C1 failed" | tee -a "$PROGRESS"

# ─── C2: C1 + post_inner=50 (forget + retain recovery) ──────────────────────
echo "=== [C2] C1 + post_inner_steps=50 ===" | tee -a "$PROGRESS"
run_sibl ablation_C2_relaxed_postinner unlearn/muse/ablation_C2_relaxed_postinner \
    || echo "[WARN] C2 failed" | tee -a "$PROGRESS"

# ─── C3: SKIPPED — logit_margin ceiling confirmed; same result expected ────────
echo "=== [C3] SKIPPED (logit_margin ceiling ~fk=0.52 regardless of mask/retain config) ===" | tee -a "$PROGRESS"

# ─── C4: Very aggressive (ε=0.5, K=3, η_θ=3e-4, ρ=0.2) + post_inner=100 ─────
echo "=== [C4] very aggressive: eps=0.5 K=3 eta=3e-4 rho=0.2 + post_inner=100 ===" | tee -a "$PROGRESS"
run_sibl ablation_C4_steep_decay unlearn/muse/ablation_C4_steep_decay \
    || echo "[WARN] C4 failed" | tee -a "$PROGRESS"

# ─── C5: SKIPPED — no implicit correction trivially degrades stability ─────────
echo "=== [C5] SKIPPED (logit_margin ceiling confirmed; no implicit adds noise) ===" | tee -a "$PROGRESS"

echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] Ablation C DONE ===" | tee -a "$PROGRESS"

# Auto-chain: run D series after C
bash /datadrive/forked/open-unlearning/scripts/run_ablation_D.sh
