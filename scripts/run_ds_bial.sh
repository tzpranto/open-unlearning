#!/bin/bash
# DS-BiAL (Dual-Space Bilevel Augmented Lagrangian) unlearning pipeline
#
# Pipeline:
#   1. [Optional] Collect mechanistic traces (gradient + activation differentials)
#   2. Compute ψ(ℓ) scores and select top-k steering layers
#   3. Run DS-BiAL unlearning with the selected layers
#   4. Evaluate the unlearned model
#
# Usage:
#   bash scripts/run_ds_bial.sh                              # MUSE News, default settings
#   bash scripts/run_ds_bial.sh --data_split Books           # MUSE Books
#   bash scripts/run_ds_bial.sh --skip_trace                 # Skip trace collection (use existing)
#   bash scripts/run_ds_bial.sh --task_name my_exp           # Custom experiment name
#   DEBUG_SIBL=1 bash scripts/run_ds_bial.sh                 # Quick debug run (T=3, K=1)
#
# Reproduces Exp8r (best validated DS-BiAL result on MUSE News):
#   forget_knowmem=0.371, verbmem=0.211, retain=0.417, extract=0.053

set -euo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ─── Defaults ────────────────────────────────────────────────────────────────
DATA_SPLIT="News"
MODEL="Llama-2-7b-hf"
TASK_NAME=""
SKIP_TRACE=0
DEBUG_MODE="${DEBUG_SIBL:-0}"
TOP_K=3
LAYER_RANGE_MIN=0
LAYER_RANGE_MAX=20

# ─── Parse CLI args ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_split)   DATA_SPLIT="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --task_name)    TASK_NAME="$2"; shift 2 ;;
        --skip_trace)   SKIP_TRACE=1; shift ;;
        --top_k)        TOP_K="$2"; shift 2 ;;
        --layer_range)  LAYER_RANGE_MIN="$2"; LAYER_RANGE_MAX="$3"; shift 3 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$TASK_NAME" ]]; then
    TASK_NAME="muse_${MODEL}_${DATA_SPLIT}_DSBiAL"
fi

# ─── Python binary ───────────────────────────────────────────────────────────
PYTHON_BIN=python
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
echo "[INFO] Using python: ${PYTHON_BIN} ($(${PYTHON_BIN} -V 2>&1))"

# ─── Attention implementation ─────────────────────────────────────────────────
ATTN_IMPL_OVERRIDE=()
if "${PYTHON_BIN}" -c "import importlib.util as u; raise SystemExit(0 if u.find_spec('flash_attn') else 1)" 2>/dev/null; then
    echo "[INFO] flash_attn detected: using model default attention."
else
    echo "[INFO] flash_attn not found: overriding to sdpa."
    ATTN_IMPL_OVERRIDE+=(model.model_args.attn_implementation=sdpa)
fi

# ─── Trace paths ─────────────────────────────────────────────────────────────
TRACE_DIR="trace_analysis/figures/traces/${DATA_SPLIT,,}_${MODEL,,}"
TRACE_RESULTS="${TRACE_DIR}/trace_results.pt"
# Fall back to MUSE News traces if no dataset-specific traces exist
MUSE_NEWS_TRACES="trace_analysis/figures/traces/muse_news_full/trace_results.pt"

# ─── Step 1: Trace collection ─────────────────────────────────────────────────
if [[ "$SKIP_TRACE" -eq 1 ]]; then
    echo "[INFO] --skip_trace: using existing traces."
elif [[ -f "$TRACE_RESULTS" ]]; then
    echo "[INFO] Trace results already exist at ${TRACE_RESULTS}, skipping collection."
    SKIP_TRACE=1
else
    echo "[INFO] Running trace collection for ${DATA_SPLIT} / ${MODEL}..."
    mkdir -p "$TRACE_DIR"
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" trace_analysis/scripts/trace_activations.py \
        --skip_causal \
        --output_dir "$TRACE_DIR" \
        --model_name "muse-bench/MUSE-${DATA_SPLIT}_target" \
        --dataset_name "muse-bench/MUSE-${DATA_SPLIT}"
    SKIP_TRACE=1
fi

# ─── Step 2: Compute ψ and select steering layers ─────────────────────────────
TRACE_TO_USE=""
if [[ -f "$TRACE_RESULTS" ]]; then
    TRACE_TO_USE="$TRACE_RESULTS"
elif [[ -f "$MUSE_NEWS_TRACES" ]]; then
    echo "[WARN] No dataset-specific traces found, falling back to MUSE News traces."
    TRACE_TO_USE="$MUSE_NEWS_TRACES"
fi

STEERING_LAYERS_OVERRIDE=()
if [[ -n "$TRACE_TO_USE" ]]; then
    echo "[INFO] Computing ψ-scores from ${TRACE_TO_USE} ..."
    PSI_JSON=$("${PYTHON_BIN}" scripts/compute_psi_layers.py \
        --trace_path "$TRACE_TO_USE" \
        --top_k "$TOP_K" \
        --layer_range "$LAYER_RANGE_MIN" "$LAYER_RANGE_MAX" \
        --json 2>/dev/null)
    STEERING_LAYERS=$(echo "$PSI_JSON" | "${PYTHON_BIN}" -c "import json,sys; d=json.load(sys.stdin); print(str(d['steering_layers']).replace(' ',''))")
    echo "[INFO] ψ-selected steering layers: ${STEERING_LAYERS}"
    STEERING_LAYERS_OVERRIDE+=("trainer.method_args.steering_layers=${STEERING_LAYERS}")
else
    echo "[WARN] No traces available; using default steering layers [5,6,7] from ds_bial.yaml config."
fi

# ─── Step 3: Run DS-BiAL unlearning ───────────────────────────────────────────
DEBUG_OVERRIDES=()
if [[ "$DEBUG_MODE" == "1" ]]; then
    echo "[INFO] DEBUG mode: T=3, K=1, stop_after_outer=2"
    DEBUG_OVERRIDES+=(
        trainer.method_args.debug_implicit=true
        trainer.method_args.debug_stop_after_outer=2
        trainer.method_args.T=3
        trainer.method_args.K=1
    )
fi

echo "[INFO] Starting DS-BiAL training: task_name=${TASK_NAME}"
if CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/muse/ds_bial \
    model="${MODEL}" \
    data_split="${DATA_SPLIT}" \
    trainer=SIBL \
    task_name="${TASK_NAME}" \
    retain_logs_path="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json" \
    trainer.args.per_device_train_batch_size=1 \
    trainer.args.gradient_accumulation_steps=2 \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true \
    trainer.args.eval_strategy=no \
    trainer.args.do_eval=false \
    trainer.args.eval_on_start=false \
    "${ATTN_IMPL_OVERRIDE[@]}" \
    "${STEERING_LAYERS_OVERRIDE[@]}" \
    "${DEBUG_OVERRIDES[@]}"; then

    # ─── Step 4: Evaluate ─────────────────────────────────────────────────────
    echo "[INFO] Evaluating: task_name=${TASK_NAME}"
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split="${DATA_SPLIT}" \
        task_name="${TASK_NAME}" \
        model="${MODEL}" \
        model.model_args.pretrained_model_name_or_path="saves/unlearn/${TASK_NAME}" \
        "${ATTN_IMPL_OVERRIDE[@]}" \
        paths.output_dir="saves/unlearn/${TASK_NAME}/evals" \
        retain_logs_path="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"

    echo "[INFO] Done. Results at: saves/unlearn/${TASK_NAME}/evals/MUSE_EVAL.json"
else
    echo "[WARN] Training failed for ${TASK_NAME}. Skipping eval."
    exit 1
fi
