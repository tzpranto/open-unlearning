#!/bin/bash
# Wait for scal f4, then eval + LLM judge + update report
set -e
cd /datadrive/forked/open-unlearning
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

BASE="saves/unlearn/muse_Llama-2-7b-hf_News_BLADE_scal_f4_s42"
RETAIN_LOGS="saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json"

# Wait for training to finish
echo "[$(date)] Waiting for scal f4 to complete..."
while [ ! -f "${BASE}/model.safetensors.index.json" ] && [ ! -f "${BASE}/model.safetensors" ]; do
    sleep 30
done
echo "[$(date)] Scal f4 done. Running eval..."

# Eval
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    experiment=eval/muse/default.yaml \
    data_split=News \
    task_name=muse_Llama-2-7b-hf_News_BLADE_scal_f4_s42 \
    model=Llama-2-7b-hf \
    model.model_args.pretrained_model_name_or_path=${BASE} \
    paths.output_dir=${BASE}/checkpoint-0/evals \
    retain_logs_path=${RETAIN_LOGS}

echo "[$(date)] Eval done. Running LLM judge..."

# LLM judge
python scripts/llm_judge.py --eval-dir ${BASE}/checkpoint-0/evals --benchmark muse

echo "[$(date)] LLM judge done."
