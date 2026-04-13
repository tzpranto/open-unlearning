#!/bin/bash
# Surgical SIBL ablation study — step-by-step kitchen-sink experiment
# A1: full model, no mask — logit_margin vs NPO (choose best loss)
# A2: winner + neuron mask (th=1.0/1.5) — sub-exps at 1.5/1.75/2.0 to tune threshold
# A3: A2-best + proportional LR
# A4: A3 + implicit correction (layer 31 only)
# A5: A4 + gradient projection (rescale + linear decay)
# A6: A5 + NPO + retain-matching steering (dual-space)
# Results tracked in trace_analysis/research/ABLATION_REPORT.md
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
MODEL="Llama-2-7b-hf"
DATA_SPLIT="News"
RETAIN_LOGS="saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json"
PROGRESS="/tmp/ablation_progress.log"

# Neuron traces (per-row format, saved by trace_activations.py)
TRACE_PATH="trace_analysis/figures/traces/muse_news/neuron_traces.pt"

cd /datadrive/forked/open-unlearning
echo "[$(date '+%H:%M:%S')] Starting surgical SIBL ablation" | tee "$PROGRESS"

# Shared base args for all experiments
BASE_TRAIN=(
    src/train.py --config-name=unlearn.yaml
    experiment=unlearn/muse/ds_bial
    model=${MODEL} data_split=${DATA_SPLIT}
    retain_logs_path=${RETAIN_LOGS}
    trainer=SIBL
    trainer.args.per_device_train_batch_size=1
    trainer.args.gradient_accumulation_steps=32
    trainer.args.num_train_epochs=3
    trainer.args.gradient_checkpointing=true
    trainer.args.eval_strategy=no
    trainer.args.do_eval=false
    trainer.args.eval_on_start=false
    model.model_args.attn_implementation=sdpa
    trainer.method_args.forget_loss_type=logit_margin
    trainer.method_args.use_steering=false
    trainer.method_args.use_implicit=false
    trainer.method_args.gradient_projection=false
    trainer.method_args.rho=0.5
    trainer.method_args.regularization_type=none
)

run_exp() {
    local NAME=$1; shift
    local DIR="saves/unlearn/ablation_${NAME}"
    if compgen -G "${DIR}/model-*.safetensors" > /dev/null 2>&1; then
        echo "[SKIP] ${NAME}: already done" | tee -a "$PROGRESS"; return 0
    fi
    echo "[RUN] ${NAME}" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" "${BASE_TRAIN[@]}" task_name="ablation_${NAME}" "$@" \
        2>&1 | tee "/tmp/train_ablation_${NAME}.log"
}

run_eval() {
    local NAME=$1
    local DIR="saves/unlearn/ablation_${NAME}"
    echo "[EVAL] ${NAME}" | tee -a "$PROGRESS"
    "${PYTHON_BIN}" src/eval.py \
        experiment=eval/muse/default.yaml data_split=${DATA_SPLIT} \
        task_name="ablation_${NAME}" model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${DIR} \
        model.model_args.attn_implementation=sdpa \
        paths.output_dir=${DIR}/evals \
        retain_logs_path=${RETAIN_LOGS} \
        2>&1 | tee "/tmp/eval_ablation_${NAME}.log"
}

# ─── A1: Full model, no mask — establish stable loss baseline ───
echo "=== [A1a] Full model, logit_margin ===" | tee -a "$PROGRESS"
run_exp "A1a_logit_margin"
run_eval "A1a_logit_margin"
echo "[A1a done]" | tee -a "$PROGRESS"

echo "=== [A1b] Full model, NPO beta=2.0 ===" | tee -a "$PROGRESS"
run_exp "A1b_npo" \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0
run_eval "A1b_npo"
echo "[A1b done]" | tee -a "$PROGRESS"

# ─── A2: Winner loss + neuron mask threshold sweep ───
# Ratio distribution in new neuron_traces.pt: th>1.0=71%, th>1.1=23%, th>1.2=3%
# Sub-experiments: th=1.0, 1.05, 1.1 — pick best for A3+

echo "=== [A2a] + mask th=1.0 ===" | tee -a "$PROGRESS"
run_exp "A2a_mask10" \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.neuron_traces_path=${TRACE_PATH} \
    trainer.method_args.mask_th_low=1.0 \
    trainer.method_args.mask_th_high=1.0
run_eval "A2a_mask10"
echo "[A2a done]" | tee -a "$PROGRESS"

echo "=== [A2b] + mask th=1.05 ===" | tee -a "$PROGRESS"
run_exp "A2b_mask105" \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.neuron_traces_path=${TRACE_PATH} \
    trainer.method_args.mask_th_low=1.05 \
    trainer.method_args.mask_th_high=1.05
run_eval "A2b_mask105"
echo "[A2b done]" | tee -a "$PROGRESS"

echo "=== [A2c] + mask th=1.1 ===" | tee -a "$PROGRESS"
run_exp "A2c_mask11" \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.neuron_traces_path=${TRACE_PATH} \
    trainer.method_args.mask_th_low=1.1 \
    trainer.method_args.mask_th_high=1.1
run_eval "A2c_mask11"
echo "[A2c done]" | tee -a "$PROGRESS"

# ─── A3: A2-best + proportional LR (th_low=1.0, th_high=best from A2) ───
echo "=== [A3] + proportional LR (th_low=1.0, th_high=best) ===" | tee -a "$PROGRESS"
# UPDATE th_high below with best threshold from A2 before running
BEST_TH=1.5
run_exp "A3_mask_propLR" \
    trainer.method_args.neuron_traces_path=${TRACE_PATH} \
    trainer.method_args.mask_th_low=1.0 \
    trainer.method_args.mask_th_high=${BEST_TH} \
    trainer.method_args.proportional_outer_lr=true
run_eval "A3_mask_propLR"
echo "[A3 done]" | tee -a "$PROGRESS"

# ─── A4: A3 + implicit correction (layer 31 only — only causally retain-critical) ───
echo "=== [A4] + implicit (layer 31 only) ===" | tee -a "$PROGRESS"
run_exp "A4_mask_propLR_implicit" \
    trainer.method_args.neuron_traces_path=${TRACE_PATH} \
    trainer.method_args.mask_th_low=1.0 \
    trainer.method_args.mask_th_high=${BEST_TH} \
    trainer.method_args.proportional_outer_lr=true \
    trainer.method_args.use_implicit=true \
    trainer.method_args.implicit_blockwise=true \
    trainer.method_args.implicit_block_last_n_layers=1
run_eval "A4_mask_propLR_implicit"
echo "[A4 done]" | tee -a "$PROGRESS"

# ─── A5: A4 + gradient projection (rescale + decay) ───
echo "=== [A5] + gradient projection ===" | tee -a "$PROGRESS"
run_exp "A5_mask_propLR_implicit_proj" \
    trainer.method_args.neuron_traces_path=${TRACE_PATH} \
    trainer.method_args.mask_th_low=1.0 \
    trainer.method_args.mask_th_high=${BEST_TH} \
    trainer.method_args.proportional_outer_lr=true \
    trainer.method_args.use_implicit=true \
    trainer.method_args.implicit_blockwise=true \
    trainer.method_args.implicit_block_last_n_layers=1 \
    trainer.method_args.gradient_projection=true \
    trainer.method_args.gradient_projection_scope=aggressive \
    trainer.method_args.projection_rescale=true \
    trainer.method_args.projection_schedule=linear_decay
run_eval "A5_mask_propLR_implicit_proj"
echo "[A5 done]" | tee -a "$PROGRESS"

# ─── A6: Full kitchen sink — A5 + NPO + retain-matching steering on layers 27-31 ───
echo "=== [A6] Full kitchen sink ===" | tee -a "$PROGRESS"
run_exp "A6_full_kitchen_sink" \
    trainer.method_args.neuron_traces_path=${TRACE_PATH} \
    trainer.method_args.mask_th_low=1.0 \
    trainer.method_args.mask_th_high=${BEST_TH} \
    trainer.method_args.proportional_outer_lr=true \
    trainer.method_args.use_implicit=true \
    trainer.method_args.implicit_blockwise=true \
    trainer.method_args.implicit_block_last_n_layers=1 \
    trainer.method_args.gradient_projection=true \
    trainer.method_args.gradient_projection_scope=aggressive \
    trainer.method_args.projection_rescale=true \
    trainer.method_args.projection_schedule=linear_decay \
    trainer.method_args.forget_loss_type=npo \
    trainer.method_args.npo_beta=2.0 \
    trainer.method_args.use_steering=true \
    trainer.method_args.steering_layers=[27,28,29,30,31] \
    trainer.method_args.steering_coeff=20.0 \
    trainer.method_args.steering_alpha=1.0 \
    trainer.method_args.steering_retain_match=true \
    trainer.method_args.steering_only=false
run_eval "A6_full_kitchen_sink"
echo "[A6 done]" | tee -a "$PROGRESS"

echo "=== ABLATION STUDY DONE ===" | tee -a "$PROGRESS"
