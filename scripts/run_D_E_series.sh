#!/bin/bash
# Run D series (NPO-based) then E series (retain recovery strategies)
# D1: NPO relaxed, D2: NPO+steering (Exp8r repro), D3: logit+steering
# E1: D2 + neuron mask, E2: D2 + post_inner=200, E3: D2 + tighter eps, E4: D2 + layer31 protection
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/datadrive/conda/envs/unlearning/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
ATTN_ARGS=(model.model_args.attn_implementation=sdpa)
PROGRESS="/tmp/ablation_DE_progress.log"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting D+E series" | tee "$PROGRESS"
echo "Gold: fk<=0.328 rk>=0.560 fv<=0.202 ex<=0.024" | tee -a "$PROGRESS"
echo "D2 old result (broken data): fk=0.286 rk=0.234 | Goal: fk<0.328 AND rk>0.45" | tee -a "$PROGRESS"

print_metrics() {
    local EVAL_FILE=$1
    "${PYTHON_BIN}" -c "
import json
with open('$EVAL_FILE') as f: d = json.load(f)
def agg(d, k):
    v = d.get(k, {})
    return v.get('agg_value', v) if isinstance(v, dict) else v
fk=agg(d,'forget_knowmem_ROUGE'); rk=agg(d,'retain_knowmem_ROUGE')
fv=agg(d,'forget_verbmem_ROUGE'); ex=agg(d,'extraction_strength'); pl=agg(d,'privleak')
beat_fk = '✓' if fk<=0.328 else '✗'
beat_rk = '✓' if rk>=0.560 else '✗'
print(f'  fk={fk:.4f}{beat_fk} rk={rk:.4f}{beat_rk} fv={fv:.4f} ex={ex:.4f} pl={pl:.2f}')
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
        trainer.args.num_train_epochs=3 \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        "${ATTN_ARGS[@]}" \
        2>&1 | tee /tmp/train_${TASK}.log
    local STATUS=$?
    [ $STATUS -ne 0 ] && echo "[WARN] ${TASK} failed (exit $STATUS)" | tee -a "$PROGRESS" && return $STATUS
    run_eval ${TASK}
}

# ── Eval C4 first (weights ready, just need eval) ─────────────────────────────
echo "=== [C4 eval] Evaluating C4 weights ===" | tee -a "$PROGRESS"
run_eval ablation_C4_steep_decay || echo "[WARN] C4 eval failed" | tee -a "$PROGRESS"

# ── D1: NPO β=2.0 + relaxed ε=0.8 ───────────────────────────────────────────
echo "=== [D1] NPO beta=2.0 + eps=0.8 K=5 rho=0.3 ===" | tee -a "$PROGRESS"
run_sibl ablation_D1_npo_relaxed unlearn/muse/ablation_D1_npo_relaxed \
    || echo "[WARN] D1 failed" | tee -a "$PROGRESS"

# ── D2: NPO + steering[5,6,7] retain_match + ε=0.8 (Exp8r repro) ─────────────
echo "=== [D2] NPO + steering[5,6,7] retain_match + eps=0.8 ===" | tee -a "$PROGRESS"
run_sibl ablation_D2_npo_steering unlearn/muse/ablation_D2_npo_steering \
    || echo "[WARN] D2 failed" | tee -a "$PROGRESS"

# ── D3: logit_margin + steering[5,6,7] + ε=0.8 ───────────────────────────────
echo "=== [D3] logit_margin + steering[5,6,7] + eps=0.8 ===" | tee -a "$PROGRESS"
run_sibl ablation_D3_logit_steering_relaxed unlearn/muse/ablation_D3_logit_steering_relaxed \
    || echo "[WARN] D3 failed" | tee -a "$PROGRESS"

echo "" | tee -a "$PROGRESS"
echo "=== D series done — checking if NPO worked ===" | tee -a "$PROGRESS"

# Quick summary
for task in ablation_D1_npo_relaxed ablation_D2_npo_steering ablation_D3_logit_steering_relaxed; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done

# ── E1: D2 + forget-dominant neuron mask ─────────────────────────────────────
echo "=== [E1] NPO + steering + neuron mask (forget-dominant only) ===" | tee -a "$PROGRESS"
run_sibl ablation_E1_npo_masked unlearn/muse/ablation_E1_npo_masked \
    || echo "[WARN] E1 failed" | tee -a "$PROGRESS"

# ── E2: D2 + post_inner_steps=200 (retain recovery after NPO forget) ─────────
echo "=== [E2] NPO + steering + post_inner=200 (retain recovery after real forget) ===" | tee -a "$PROGRESS"
run_sibl ablation_E2_npo_postinner unlearn/muse/ablation_E2_npo_postinner \
    || echo "[WARN] E2 failed" | tee -a "$PROGRESS"

# ── E3: NPO + steering + tighter ε=0.5 + K=10 ───────────────────────────────
echo "=== [E3] NPO + steering + eps=0.5 K=10 (tighter retain with NPO) ===" | tee -a "$PROGRESS"
run_sibl ablation_E3_npo_tight unlearn/muse/ablation_E3_npo_tight \
    || echo "[WARN] E3 failed" | tee -a "$PROGRESS"

# ── E4: D2 + retain_protection_layers=[31] ───────────────────────────────────
echo "=== [E4] NPO + steering + retain_protection layer31 ===" | tee -a "$PROGRESS"
run_sibl ablation_E4_npo_layer31 unlearn/muse/ablation_E4_npo_layer31 \
    || echo "[WARN] E4 failed" | tee -a "$PROGRESS"

# ── E5: NPO β=0.5 (weaker) + steering — prevents early NPO saturation ────────
echo "=== [E5] NPO beta=0.5 + steering (prevent saturation) ===" | tee -a "$PROGRESS"
run_sibl ablation_E5_npo_weak unlearn/muse/ablation_E5_npo_weak \
    || echo "[WARN] E5 failed" | tee -a "$PROGRESS"

# ── E6: SimNPO + steering — length-normalized, reference-free ────────────────
echo "=== [E6] SimNPO beta=2.0 + steering (stable reference-free NPO) ===" | tee -a "$PROGRESS"
run_sibl ablation_E6_simnpo_steering unlearn/muse/ablation_E6_simnpo_steering \
    || echo "[WARN] E6 failed" | tee -a "$PROGRESS"

echo "" | tee -a "$PROGRESS"
echo "=== [$(date '+%H:%M:%S')] D+E series DONE ===" | tee -a "$PROGRESS"

# Final summary
echo "" | tee -a "$PROGRESS"
echo "=== FULL RESULTS SUMMARY ===" | tee -a "$PROGRESS"
for task in ablation_C4_steep_decay ablation_D1_npo_relaxed ablation_D2_npo_steering ablation_D3_logit_steering_relaxed ablation_E1_npo_masked ablation_E2_npo_postinner ablation_E3_npo_tight ablation_E4_npo_layer31 ablation_E5_npo_weak ablation_E6_simnpo_steering; do
    ef="saves/unlearn/${task}/evals/MUSE_EVAL.json"
    echo -n "  $task: " | tee -a "$PROGRESS"
    [ -f "$ef" ] && print_metrics "$ef" | tee -a "$PROGRESS" || echo "no eval" | tee -a "$PROGRESS"
done
