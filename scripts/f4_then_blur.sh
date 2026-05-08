#!/bin/bash
# After scal f4 finishes: eval + LLM judge, then run all BLUR experiments
set -uo pipefail
cd /datadrive/forked/open-unlearning
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

F4_DIR="saves/unlearn/muse_Llama-2-7b-hf_News_BLADE_scal_f4_s42"
RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json"

# Wait for scal f4
echo "[$(date)] Waiting for scal f4..."
while [ ! -f "${F4_DIR}/model.safetensors.index.json" ] && [ ! -f "${F4_DIR}/model.safetensors" ]; do
    sleep 30
done
echo "[$(date)] Scal f4 done."

# Eval
echo "[$(date)] Running eval on f4..."
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    experiment=eval/muse/default.yaml \
    data_split=News \
    task_name=muse_Llama-2-7b-hf_News_BLADE_scal_f4_s42 \
    model=Llama-2-7b-hf \
    model.model_args.pretrained_model_name_or_path=${F4_DIR} \
    paths.output_dir=${F4_DIR}/checkpoint-0/evals \
    retain_logs_path=${RETAIN_LOGS}
echo "[$(date)] Eval done."

# LLM judge
echo "[$(date)] Running LLM judge on f4..."
python scripts/llm_judge.py --eval-dir ${F4_DIR}/checkpoint-0/evals --benchmark muse
echo "[$(date)] LLM judge done."

# Now run all BLUR experiments
echo "[$(date)] Starting BLUR runs..."
bash scripts/run_blur_all.sh
