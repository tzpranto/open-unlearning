#!/usr/bin/env bash
set -euo pipefail

# Minimal debug experiment runner.
# Usage:
#   bash debug/run_experiments.sh toy
#   bash debug/run_experiments.sh train-neumann
#   bash debug/run_experiments.sh train-cg

mode="${1:-toy}"

case "${mode}" in
  toy)
    python debug/toy_neumann_vs_cg.py
    ;;

  train-neumann)
    DEBUG_SIBL=1 bash scripts/muse_baselines.sh
    ;;

  train-cg)
    DEBUG_SIBL=1 CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
      experiment=unlearn/muse/sibl.yaml \
      model=Llama-2-7b-hf \
      data_split=News \
      trainer=SIBL \
      task_name=muse_Llama-2-7b-hf_News_SIBL_debug_cg \
      retain_logs_path=saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json \
      trainer.args.per_device_train_batch_size=1 \
      trainer.args.gradient_accumulation_steps=2 \
      trainer.args.ddp_find_unused_parameters=true \
      trainer.args.gradient_checkpointing=true \
      trainer.method_args.use_implicit=true \
      trainer.method_args.implicit_solver=cg \
      trainer.method_args.debug_implicit=true \
      trainer.method_args.debug_save_arrays=true \
      trainer.method_args.debug_stop_after_outer=2 \
      trainer.method_args.T=3 \
      trainer.method_args.K=1
    ;;

  *)
    echo "Unknown mode: ${mode}"
    echo "Valid modes: toy | train-neumann | train-cg"
    exit 1
    ;;
esac
