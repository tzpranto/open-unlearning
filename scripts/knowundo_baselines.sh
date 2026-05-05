#!/bin/bash
# KnowUnDo baselines — single GPU, train + inline eval, skip-if-done
# Usage: bash scripts/knowundo_baselines.sh
# Requires finetuned target model at saves/unlearn/knowundo_Llama-2-7b-chat_copyright_ft
# (run scripts/knowundo_finetune.sh first)
#
# All methods use MUSE News params (lr, scheduler, epochs) as unified config.
# Eval runs inline after training (ROUGE-L + Probability on forget/retain val sets).
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

# ── Config ──────────────────────────────────────────────────────
DOMAIN="${DOMAIN:-copyright}"
SEEDS=(${SEEDS:-42})
TARGET_MODEL="saves/unlearn/knowundo_Llama-2-7b-chat_${DOMAIN}_ft"
# ────────────────────────────────────────────────────────────────

# Verify target model exists
if [[ ! -d "$TARGET_MODEL" ]]; then
    echo "[ERROR] Target model not found: $TARGET_MODEL"
    echo "Run: bash scripts/knowundo_finetune.sh $DOMAIN"
    exit 1
fi

# ── Gold standard eval on finetuned target model ──────────────
TARGET_EVAL_DIR="${TARGET_MODEL}/evals"
if [[ -f "${TARGET_EVAL_DIR}/KnowUnDo_SUMMARY.json" ]] && [[ -f "${TARGET_EVAL_DIR}/LMEval_SUMMARY.json" ]]; then
    echo "[GOLD SKIP] Target model already evaluated"
else
    echo "[GOLD] Evaluating finetuned target model..."
    mkdir -p "${TARGET_EVAL_DIR}"
    CUDA_VISIBLE_DEVICES=0 python src/eval.py --config-name=eval.yaml \
        model=Llama-2-7b-chat-hf \
        model.model_args.pretrained_model_name_or_path=${TARGET_MODEL} \
        eval=knowundo \
        eval.knowundo.domain=${DOMAIN} \
        eval.knowundo.output_dir=${TARGET_EVAL_DIR} \
        eval.knowundo.overwrite=true \
        task_name=knowundo_target_${DOMAIN} \
        seed=42
    echo "[GOLD] KnowUnDo eval done:"
    cat "${TARGET_EVAL_DIR}/KnowUnDo_SUMMARY.json" 2>/dev/null || true

    CUDA_VISIBLE_DEVICES=0 python src/eval.py --config-name=eval.yaml \
        model=Llama-2-7b-chat-hf \
        model.model_args.pretrained_model_name_or_path=${TARGET_MODEL} \
        eval=lm_eval \
        "eval.lm_eval.tasks=[mmlu]" \
        eval.lm_eval.output_dir=${TARGET_EVAL_DIR} \
        eval.lm_eval.overwrite=true \
        task_name=knowundo_target_${DOMAIN}_lm_eval \
        seed=42
    echo "[GOLD] LMEval done:"
    cat "${TARGET_EVAL_DIR}/LMEval_SUMMARY.json" 2>/dev/null || true
fi

# ── MemFlex localization (using original KnowUnDo code on LoRA adapters) ──
MEMFLEX_PARAMS="saves/unlearn/knowundo_memflex_localize_${DOMAIN}/located_params.json"
if [[ ! -f "$MEMFLEX_PARAMS" ]]; then
    echo "[LOCALIZE] Running MemFlex parameter localization (original code)..."
    mkdir -p "saves/unlearn/knowundo_memflex_localize_${DOMAIN}"

    # Get final checkpoint (LoRA adapter) — need absolute path
    FINAL_CKPT=$(ls -d /datadrive/forked/open-unlearning/${TARGET_MODEL}/checkpoint-* | sort -t- -k2 -n | tail -1)
    echo "[LOCALIZE] Using adapter: $FINAL_CKPT"

    # Run original localization from KnowUnDo repo
    cd /datadrive/forked/KnowUnDo/pretrain
    CUDA_VISIBLE_DEVICES=0 python localization.py \
        --model_name_or_path ${FINAL_CKPT} \
        --model_id meta-llama/Llama-2-7b-chat-hf \
        --data_type ${DOMAIN} \
        --num_copies 3 \
        --sim_thresh 0.92 \
        --grad_thresh 6e-4
    cd /datadrive/forked/open-unlearning

    # Map LoRA param names → full model param names
    LORA_REGION="/datadrive/forked/KnowUnDo/pretrain/outputs/Llama-2-7b-chat-hf/located_region_${DOMAIN}.json"
    python -c "
import json
with open('${LORA_REGION}') as f:
    lora_names = json.load(f)
# Map: base_model.model.model.layers.X.self_attn.q_proj.lora_A.weight -> model.layers.X.self_attn.q_proj.weight
full_names = set()
for name in lora_names:
    # Strip 'base_model.model.' prefix and '.lora_[AB].weight' suffix
    core = name.replace('base_model.model.', '')
    # Remove .lora_A.weight or .lora_B.weight
    for suffix in ['.lora_A.weight', '.lora_B.weight', '.lora_A.default.weight', '.lora_B.default.weight']:
        if core.endswith(suffix):
            core = core[:-len(suffix)] + '.weight'
            break
    full_names.add(core)
import os
os.makedirs(os.path.dirname('${MEMFLEX_PARAMS}'), exist_ok=True)
with open('${MEMFLEX_PARAMS}', 'w') as f:
    json.dump(sorted(full_names), f, indent=2)
print(f'Mapped {len(lora_names)} LoRA params -> {len(full_names)} full model params')
"
    echo "[LOCALIZE DONE] $MEMFLEX_PARAMS"
else
    echo "[LOCALIZE SKIP] $MEMFLEX_PARAMS already exists"
fi

total=0; skip=0; fail=0

# ── Helper: train + eval a single method ────────────────────────
run_method() {
    local task_name=$1
    local experiment=$2
    shift 2
    local outdir="saves/unlearn/${task_name}"
    total=$((total + 1))

    # Skip if already fully evaluated (KnowUnDo + lm_eval)
    if [[ -f "$outdir/evals/KnowUnDo_SUMMARY.json" ]] && [[ -f "$outdir/evals/LMEval_SUMMARY.json" ]]; then
        echo "[SKIP] $task_name"
        skip=$((skip + 1))
        return 0
    fi

    echo "============================================================"
    echo "[TRAIN] $(date) $task_name"
    if ! CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/knowundo/${experiment} \
        domain=${DOMAIN} \
        task_name=${task_name} \
        trainer.args.eval_strategy=no \
        trainer.args.eval_on_start=false \
        trainer.args.seed=${SEED} \
        "$@"; then
        echo "[TRAIN FAILED] $task_name"
        fail=$((fail + 1))
        return 1
    fi

    # Run KnowUnDo eval separately (not inline) to ensure correct output path
    echo "[EVAL] $(date) $task_name"
    CUDA_VISIBLE_DEVICES=0 python src/eval.py --config-name=eval.yaml \
        model=Llama-2-7b-chat-hf \
        model.model_args.pretrained_model_name_or_path=${outdir} \
        eval=knowundo \
        eval.knowundo.domain=${DOMAIN} \
        eval.knowundo.output_dir=${outdir}/evals \
        eval.knowundo.overwrite=true \
        task_name=${task_name} \
        seed=${SEED} || echo "[EVAL WARN] $task_name — KnowUnDo eval failed"
    echo "[EVAL DONE] $(date) $task_name"
    cat "$outdir/evals/KnowUnDo_SUMMARY.json" 2>/dev/null || true

    # Run MMLU
    echo "[LM_EVAL] $task_name ..."
    CUDA_VISIBLE_DEVICES=0 python src/eval.py --config-name=eval.yaml \
        model=Llama-2-7b-chat-hf \
        model.model_args.pretrained_model_name_or_path=${outdir} \
        eval=lm_eval \
        "eval.lm_eval.tasks=[mmlu]" \
        eval.lm_eval.output_dir=${outdir}/evals \
        eval.lm_eval.overwrite=false \
        task_name=${task_name}_lm_eval \
        seed=${SEED} 2>&1 | tail -5 || echo "[LM_EVAL WARN] $task_name — lm_eval failed, continuing"

    # Clean checkpoints only — keep model weights for LLM Judge eval later
    rm -rf "$outdir"/checkpoint-* 2>/dev/null
    rm -f "$outdir"/optimizer* "$outdir"/scheduler* "$outdir"/training_args* 2>/dev/null
    echo "[CLEANED] $task_name — kept model + evals"
}

# ── Main loop over seeds ────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "################################################################"
    echo "# SEED=${SEED}  DOMAIN=${DOMAIN}"
    echo "################################################################"

    # GA (GradAscent) — default.yaml already uses GradAscent trainer
    run_method "knowundo_GA_${DOMAIN}_s${SEED}" "default"

    # GradDiff
    run_method "knowundo_GradDiff_${DOMAIN}_s${SEED}" "grad_diff"

    # NPO
    run_method "knowundo_NPO_${DOMAIN}_s${SEED}" "npo"

    # SimNPO
    run_method "knowundo_SimNPO_${DOMAIN}_s${SEED}" "simnpo"

    # RMU
    run_method "knowundo_RMU_${DOMAIN}_s${SEED}" "rmu"

    # BLUR-NPO
    run_method "knowundo_BLURNPO_${DOMAIN}_s${SEED}" "blurnpo"

    # PDU
    run_method "knowundo_PDU_${DOMAIN}_s${SEED}" "pdu"

    # BLADE (ours)
    run_method "knowundo_BLADE_${DOMAIN}_s${SEED}" "blade"

    # MemFlex (KnowUnDo authors' method)
    run_method "knowundo_MemFlex_${DOMAIN}_s${SEED}" "memflex" \
        trainer.method_args.located_params_path=${MEMFLEX_PARAMS}

    echo ""
    echo "[SEED ${SEED} DONE] total=${total} skip=${skip} fail=${fail}"
done

# ── Summarize results ──────────────────────────────────────────
echo ""
echo "============================================================"
python scripts/knowundo_summarize.py ${DOMAIN}
echo "============================================================"
echo "ALL DONE at $(date) — total=${total} skip=${skip} fail=${fail}"
echo "============================================================"
