#!/bin/bash
# V series: Push T8f's fk lower while preserving frontier break
#
# T8f (K=3, log+contrastive): fk=0.428, rk=0.461, ABOVE +0.056
# U4  (K=5, log+contrastive): fk=0.328, rk=0.330, ON frontier
#
# The frontier break lives between K=3 and K=5. This series explores:
#   V_interp_a7/a5/a3: Weight interpolation T8f/U4 (no training, free)
#   V0: K=4 (midpoint between T8f and U4)
#   V1: K=3, npo_beta=1.5 (moderate NPO boost, less aggressive than U2's 1.0)
#   V2: K=3, gamma=0.3 (weaker contrastive → more NPO headroom)
#   V3: K=3, gamma=0.7 (stronger contrastive → between T8f and U5)
#   V4: K=3, eta_theta=3e-4 (higher outer LR → more NPO per step)
#   V5: K=4, gamma=0.3 (combine K=4 with weaker contrastive)
#   V6: K=3, 80 steps (stop before ALM λ grows too large)
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
LOG_DIR="/datadrive/forked/open-unlearning/logs/ablation"
mkdir -p "$LOG_DIR"
PROGRESS="${LOG_DIR}/V_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting V series (push T8f fk lower)" | tee "$PROGRESS"

print_metrics() {
    local EVAL_FILE=$1
    local REF_FK=${2:-0.428}  # default reference: T8f
    local REF_RK=${3:-0.461}
    "${PYTHON_BIN}" -c "
import json
with open('$EVAL_FILE') as f: d = json.load(f)
def agg(d, k):
    v = d.get(k, {})
    return v.get('agg_value', v) if isinstance(v, dict) else v
fk=float(agg(d,'forget_knowmem_ROUGE')); rk=float(agg(d,'retain_knowmem_ROUGE'))
fv=float(agg(d,'forget_verbmem_ROUGE')); ex=float(agg(d,'extraction_strength'))
pred_rk = 0.55*fk + 0.17
delta = rk - pred_rk
frontier = 'ABOVE' if delta > 0.02 else ('ON' if delta > -0.02 else 'BELOW')
fk_gold = '[GOLD]' if fk <= 0.328 else ''
rk_gold = '[GOLD]' if rk >= 0.560 else f'[{rk:.3f}]'
print(f'  fk={fk:.4f}{fk_gold} rk={rk:.4f}{rk_gold} fv={fv:.4f} ex={ex:.4f} | {frontier} ({delta:+.3f}) | vs T8f: fk{fk-$REF_FK:+.3f} rk{rk-$REF_RK:+.3f}')
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

# ── Phase 1: Eval weight interpolations (FREE, no training) ─────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [Phase 1] Weight interpolation evals ===" | tee -a "$PROGRESS"
for alpha in 7 5 3; do
    TASK="ablation_V_interp_a${alpha}"
    EVAL_FILE="saves/unlearn/${TASK}/evals/MUSE_EVAL.json"
    if [ -f "$EVAL_FILE" ]; then
        echo "[CACHED] ${TASK}" | tee -a "$PROGRESS"
        print_metrics "$EVAL_FILE" | tee -a "$PROGRESS"
    else
        run_eval "${TASK}"
    fi
done

# ── Phase 2: Training experiments ────────────────────────────────────────────
# All use T8f base config (S5 + log weighting) with contrastive inner enabled

# V0: K=4 (midpoint between T8f K=3 and U4 K=5)
echo "" | tee -a "$PROGRESS"
echo "=== [V0] K=4 + log + contrastive ===" | tee -a "$PROGRESS"
run_sibl ablation_V0_k4 unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.K=4 \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=0.5 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] V0 failed" | tee -a "$PROGRESS"

# V1: npo_beta=1.5 (moderate boost, between T8f's 2.0 and U2's 1.0)
echo "" | tee -a "$PROGRESS"
echo "=== [V1] npo_beta=1.5 + log + contrastive ===" | tee -a "$PROGRESS"
run_sibl ablation_V1_npo15 unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.npo_beta=1.5 \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=0.5 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] V1 failed" | tee -a "$PROGRESS"

# V2: gamma=0.3 (weaker contrastive → more NPO headroom for fk)
echo "" | tee -a "$PROGRESS"
echo "=== [V2] gamma=0.3 (weaker contrastive push) ===" | tee -a "$PROGRESS"
run_sibl ablation_V2_gamma03 unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=0.3 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] V2 failed" | tee -a "$PROGRESS"

# V3: gamma=0.7 (stronger push, between T8f's 0.5 and U5's 1.0)
echo "" | tee -a "$PROGRESS"
echo "=== [V3] gamma=0.7 (moderate-strong push) ===" | tee -a "$PROGRESS"
run_sibl ablation_V3_gamma07 unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=0.7 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] V3 failed" | tee -a "$PROGRESS"

# V4: eta_theta=3e-4 (50% higher outer LR → stronger NPO per step)
echo "" | tee -a "$PROGRESS"
echo "=== [V4] eta_theta=3e-4 (higher outer LR) ===" | tee -a "$PROGRESS"
run_sibl ablation_V4_highLR unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.eta_theta=3e-4 \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=0.5 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] V4 failed" | tee -a "$PROGRESS"

# V5: K=4 + gamma=0.3 (combine two fk-pushing changes)
echo "" | tee -a "$PROGRESS"
echo "=== [V5] K=4 + gamma=0.3 (combined push) ===" | tee -a "$PROGRESS"
run_sibl ablation_V5_k4_gamma03 unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=99 \
    trainer.method_args.K=4 \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=0.3 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] V5 failed" | tee -a "$PROGRESS"

# V6: 80 steps (stop before ALM λ grows too large)
echo "" | tee -a "$PROGRESS"
echo "=== [V6] 80 steps (shorter) ===" | tee -a "$PROGRESS"
run_sibl ablation_V6_80steps unlearn/muse/ablation_T8_weighted_s5 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.method_args.debug_stop_after_outer=79 \
    trainer.method_args.inner_contrastive=true \
    trainer.method_args.inner_contrastive_beta=1.0 \
    trainer.method_args.inner_contrastive_gamma=0.5 \
    "trainer.method_args.inner_contrastive_layers=[5,6,7]" \
    || echo "[WARN] V6 failed" | tee -a "$PROGRESS"

# ── Summary ──────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] V series DONE ===" | tee -a "$PROGRESS"
echo "" | tee -a "$PROGRESS"
echo "=== SUMMARY ===" | tee -a "$PROGRESS"
echo "  Gold targets: fk<=0.328, rk>=0.560" | tee -a "$PROGRESS"
echo "  CE frontier: rk ≈ 0.55*fk + 0.17" | tee -a "$PROGRESS"
echo "  T8f reference (log+contrastive K=3): fk=0.428 rk=0.461 | ABOVE (+0.056)" | tee -a "$PROGRESS"
echo "  U4  reference (log+contrastive K=5): fk=0.328 rk=0.330 | ON (-0.021)" | tee -a "$PROGRESS"
for task in ablation_V_interp_a7 ablation_V_interp_a5 ablation_V_interp_a3 \
            ablation_V0_k4 ablation_V1_npo15 ablation_V2_gamma03 ablation_V3_gamma07 \
            ablation_V4_highLR ablation_V5_k4_gamma03 ablation_V6_80steps; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done
