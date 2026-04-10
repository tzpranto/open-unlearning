#!/bin/bash
# P series: Fix ALM retain penalty — lambda_init + stronger rho
# KEY INSIGHT: With lambda_init=0 and rho=0.1, the retain gradient is <1% of NPO at step 1.
# The dual variable λ takes ~20 steps to build up enough retain protection, by which point
# retain is already destroyed. Fix: initialize λ > 0 and/or increase ρ.
#
# All experiments use N4 base (full outer NPO + inverted inner mask on retain neurons).
#
# Experiments:
#   P0: λ_init=5.0, ρ=1.0 (strongest ALM — retain protected from step 1)
#   P1: λ_init=2.0, ρ=1.0 (moderate ALM — less retain-heavy)
#   P2: λ_init=5.0, ρ=1.0, K=2, eta_in=3e-4, eta_theta=1e-4 (strong ALM + strong inner + gentle outer)
#   P3: λ_init=0.0, ρ=5.0 (control: fast ρ growth, no initial λ)
#
# Anchor: N4 fk=0.344 rk=0.343 | G1 fk=0.274 rk=0.327 | Gold: fk<=0.328, rk>=0.560
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/P_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting P series (ALM retain penalty fix)" | tee "$PROGRESS"
echo "Anchors: N4 fk=0.344 rk=0.343 | G1 fk=0.274 rk=0.327 | Gold: fk<=0.328 rk>=0.560" | tee -a "$PROGRESS"

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
fk_gold = 'GOLD' if fk<=0.328 else ('OK' if fk<=0.340 else 'HURT')
rk_gold = 'GOLD' if rk>=0.560 else f'{rk:.3f}'
fk_delta = fk - 0.274
rk_delta = rk - 0.327
print(f'  fk={fk:.4f}[{fk_gold}]({fk_delta:+.3f} vs G1) rk={rk:.4f}[{rk_gold}]({rk_delta:+.3f} vs G1) fv={fv:.4f} ex={ex:.4f}')
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
        2>&1 | tee "${LOG_DIR}/train_${TASK}.log"
    local STATUS=$?
    [ $STATUS -ne 0 ] && echo "[WARN] ${TASK} failed (exit $STATUS)" | tee -a "$PROGRESS" && return $STATUS
    run_eval ${TASK}
}

# ── P0: Strongest ALM (most promising — run first) ────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [P0] λ_init=5.0, ρ=1.0 + full outer + inverted inner ===" | tee -a "$PROGRESS"
echo "  Tests: strong retain protection from step 1 + N4 architecture" | tee -a "$PROGRESS"
run_sibl ablation_P0_alm_strong unlearn/muse/ablation_P0_alm_strong \
    || echo "[WARN] P0 failed" | tee -a "$PROGRESS"

# ── P1: Moderate ALM ──────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [P1] λ_init=2.0, ρ=1.0 + full outer + inverted inner ===" | tee -a "$PROGRESS"
echo "  Tests: moderate retain protection — less likely to starve forget" | tee -a "$PROGRESS"
run_sibl ablation_P1_alm_moderate unlearn/muse/ablation_P1_alm_moderate \
    || echo "[WARN] P1 failed" | tee -a "$PROGRESS"

# ── P2: Strong ALM + K=2 + higher inner LR + gentler outer ───────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [P2] λ_init=5.0, ρ=1.0, K=2, eta_in=3e-4, eta_theta=1e-4 ===" | tee -a "$PROGRESS"
echo "  Tests: maximum retain recovery — strong ALM + 2x inner steps + 3x inner LR + half outer LR" | tee -a "$PROGRESS"
run_sibl ablation_P2_alm_strong_K2 unlearn/muse/ablation_P2_alm_strong_K2 \
    || echo "[WARN] P2 failed" | tee -a "$PROGRESS"

# ── P3: Control — high rho only ──────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [P3] λ_init=0.0, ρ=5.0 (control: fast dual growth, no initial λ) ===" | tee -a "$PROGRESS"
echo "  Tests: whether fast ρ growth alone fixes retain (vs needing λ_init)" | tee -a "$PROGRESS"
run_sibl ablation_P3_rho_only unlearn/muse/ablation_P3_rho_only \
    || echo "[WARN] P3 failed" | tee -a "$PROGRESS"

# ── Summary ──────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] P series DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "  Gold targets: fk<=0.328, rk>=0.560" | tee -a "$PROGRESS"
echo "  Anchor G1: fk=0.274 rk=0.327 | N4: fk=0.344 rk=0.343" | tee -a "$PROGRESS"
for task in ablation_P0_alm_strong ablation_P1_alm_moderate ablation_P2_alm_strong_K2 ablation_P3_rho_only; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done
