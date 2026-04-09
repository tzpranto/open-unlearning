#!/bin/bash
# I series: DGA soft-masked post-inner recovery
# I0: Compute DGA selectivity scores on G1 checkpoint
# I1: G1 + soft-masked post_inner (β=5.0, 25 steps) — main experiment
# I2: β sweep {1.0, 3.0, 10.0, 20.0} — find sharpness sweet spot
#
# Background: G1 fk=0.274 ✅ rk=0.327 ❌ | G3 fk=0.657 ❌ rk=0.572 ✅
# Expert diagnosis: binary bitmap has 68% contested neurons → G3 fails.
# Fix: α = σ(-β * s_n) continuous mask, contested neurons get partial updates.
# Gold targets: fk<=0.328, rk>=0.560
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/I_progress.log"
DGA_SCORES="trace_analysis/figures/traces/analysis/dga_selectivity_G1.pt"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting I series (DGA soft-masked recovery)" | tee "$PROGRESS"
echo "G1 anchor: fk=0.274 rk=0.327 | G3 rk-ceiling: rk=0.572 | Gold: fk<=0.328 rk>=0.560" | tee -a "$PROGRESS"

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
fk_ok = 'GOLD' if fk<=0.328 else ('NEAR' if fk<=0.360 else 'HURT')
rk_ok = 'GOLD' if rk>=0.560 else ('NEAR' if rk>=0.450 else f'{rk:.3f}')
fk_delta = fk - 0.274; rk_delta = rk - 0.327
print(f'  fk={fk:.4f}[{fk_ok}]({fk_delta:+.3f} vs G1) rk={rk:.4f}[{rk_ok}]({rk_delta:+.3f} vs G1) fv={fv:.4f} ex={ex:.4f}')
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
        2>&1 | tee "${LOG_DIR}/eval_${TASK}.log"
    local EVAL_FILE="saves/unlearn/${TASK}/evals/MUSE_EVAL.json"
    [ -f "$EVAL_FILE" ] && print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"
}

run_sibl() {
    local TASK=$1 EXPERIMENT=$2
    shift 2
    local EXTRA_ARGS=("$@")
    local TASK_DIR="saves/unlearn/${TASK}"
    if compgen -G "${TASK_DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] ${TASK}: weights exist" | tee -a "$PROGRESS"
        local EVAL_FILE="${TASK_DIR}/evals/MUSE_EVAL.json"
        if [ ! -f "$EVAL_FILE" ]; then run_eval ${TASK}
        else echo "[CACHED]" | tee -a "$PROGRESS"; print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"; fi
        return 0
    fi
    echo "[$(date '+%H:%M:%S')] [TRAIN] ${TASK}" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/train.py --config-name=unlearn.yaml \
        experiment=${EXPERIMENT} model=${MODEL} data_split=${DATA_SPLIT} \
        trainer=SIBL task_name=${TASK} retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_accumulation_steps=32 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        "${ATTN_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/train_${TASK}.log"
    local STATUS=$?
    [ $STATUS -ne 0 ] && echo "[WARN] ${TASK} failed (exit $STATUS)" | tee -a "$PROGRESS" && return $STATUS
    run_eval ${TASK}
}

# ── I0: DGA scoring on G1 checkpoint ─────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [I0] DGA gradient attribution on G1 checkpoint ===" | tee -a "$PROGRESS"
if [ -f "${DGA_SCORES}" ]; then
    echo "[SKIP] DGA scores already exist at ${DGA_SCORES}" | tee -a "$PROGRESS"
else
    echo "[$(date '+%H:%M:%S')] Running DGA scorer..." | tee -a "$PROGRESS"
    "${PYTHON_BIN}" scripts/score_dga.py \
        --model_path saves/unlearn/ablation_G1_npo_weak_steering \
        --ref_model muse-bench/MUSE-News_target \
        --tokenizer_path meta-llama/Llama-2-7b-hf \
        --forget_dataset muse-bench/MUSE-News --forget_config raw --forget_split forget \
        --retain_dataset muse-bench/MUSE-News --retain_config raw --retain_split retain1 \
        --output_path "${DGA_SCORES}" \
        --max_forget 0 --max_retain 200 --max_length 512 --npo_beta 2.0 --attn_impl sdpa \
        2>&1 | tee "${LOG_DIR}/I0_dga_scoring.log"
    if [ $? -ne 0 ] || [ ! -f "${DGA_SCORES}" ]; then
        echo "[ERROR] DGA scoring failed — check ${LOG_DIR}/I0_dga_scoring.log" | tee -a "$PROGRESS"
        exit 1
    fi
    echo "[$(date '+%H:%M:%S')] DGA scoring complete → ${DGA_SCORES}" | tee -a "$PROGRESS"
fi

# ── I1: G1 + soft-masked post_inner (β=5.0) ──────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [I1] G1 + DGA soft recovery β=5.0 ===" | tee -a "$PROGRESS"
run_sibl ablation_I1_dga_b5 unlearn/muse/ablation_I1_dga_soft_recovery \
    || echo "[WARN] I1 failed" | tee -a "$PROGRESS"

# ── I2: β sweep ───────────────────────────────────────────────────────────────
# Run only if I1 shows promise (rk improvement over G1 without fk regression)
echo "" | tee -a "$PROGRESS"
echo "=== [I2] β sweep: 1.0, 3.0, 10.0, 20.0 ===" | tee -a "$PROGRESS"

for BETA in 1.0 3.0 10.0 20.0; do
    BETA_TAG=$(echo "$BETA" | tr '.' '_')
    TASK="ablation_I2_dga_b${BETA_TAG}"
    echo "" | tee -a "$PROGRESS"
    echo "--- [I2] β=${BETA} → ${TASK} ---" | tee -a "$PROGRESS"
    run_sibl "${TASK}" unlearn/muse/ablation_I1_dga_soft_recovery \
        trainer.method_args.post_inner_soft_mask_beta="${BETA}" \
        || echo "[WARN] I2 β=${BETA} failed" | tee -a "$PROGRESS"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] I series DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "  Gold targets: fk<=0.328, rk>=0.560" | tee -a "$PROGRESS"
echo "  G1 anchor:    fk=0.274, rk=0.327" | tee -a "$PROGRESS"
for task in \
    ablation_G1_npo_weak_steering \
    ablation_I1_dga_b5 \
    ablation_I2_dga_b1_0 \
    ablation_I2_dga_b3_0 \
    ablation_I2_dga_b10_0 \
    ablation_I2_dga_b20_0; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done
