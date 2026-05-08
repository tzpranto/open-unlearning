#!/bin/bash
# Wait for Books s1024 to finish, then run News BLADE 4 seeds
set -e
cd /datadrive/forked/open-unlearning

SEEDS=(123 456 789 1024)
BASE_DIR="./saves/unlearn"

# Wait for s1024 to finish (check for model weights)
echo "[$(date)] Waiting for Books s1024 to complete..."
while [ ! -f "${BASE_DIR}/muse_Llama-2-7b-hf_Books_adaptive_T250_s1024/model.safetensors.index.json" ] && \
      [ ! -f "${BASE_DIR}/muse_Llama-2-7b-hf_Books_adaptive_T250_s1024/model.safetensors" ]; do
    sleep 60
done
echo "[$(date)] Books s1024 done!"

# Run News BLADE 4 seeds
for SEED in "${SEEDS[@]}"; do
    OUT="${BASE_DIR}/muse_Llama-2-7b-hf_News_adaptive_s${SEED}"

    if [ -f "${OUT}/model.safetensors.index.json" ] || [ -f "${OUT}/model.safetensors" ]; then
        echo "[SKIP] ${OUT##*/} already completed"
        continue
    fi

    echo "============================================================"
    echo "[$(date)] Starting News BLADE seed=${SEED}..."

    CUDA_VISIBLE_DEVICES=0 python src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/muse/lora_bial_adaptive_books \
        data_split=News \
        model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-News_target \
        trainer.method_args.T=300 \
        trainer.method_args.conv_patience=20 \
        trainer.args.seed=${SEED} \
        task_name=muse_Llama-2-7b-hf_News_adaptive_s${SEED}

    echo "[$(date)] Done News BLADE seed=${SEED}"
done

echo "============================================================"
echo "[$(date)] ALL NEWS SEEDS DONE"
