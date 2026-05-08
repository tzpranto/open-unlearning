#!/bin/bash
# BLUR-NPO: 4 seeds each for News, Books, KnowUnDo copyright, KnowUnDo privacy
# Run AFTER News BLADE finishes
set -e
cd /datadrive/forked/open-unlearning

SEEDS=(123 456 789 1024)
BASE_DIR="./saves/unlearn"

# Wait for BLADE scal f4 to finish
echo "[$(date)] Waiting for BLADE scal f4 to complete..."
while [ ! -f "${BASE_DIR}/muse_Llama-2-7b-hf_News_BLADE_scal_f4_s42/model.safetensors.index.json" ] && \
      [ ! -f "${BASE_DIR}/muse_Llama-2-7b-hf_News_BLADE_scal_f4_s42/model.safetensors" ]; do
    sleep 60
done
echo "[$(date)] Scal f4 done! Starting BLUR runs..."

# ============================================================
# MUSE News BLUR (4 seeds)
# ============================================================
for SEED in "${SEEDS[@]}"; do
    OUT="${BASE_DIR}/muse_Llama-2-7b-hf_News_BLURNPO_s${SEED}"
    if [ -f "${OUT}/model.safetensors.index.json" ] || [ -f "${OUT}/model.safetensors" ]; then
        echo "[SKIP] ${OUT##*/} already completed"
        continue
    fi
    echo "============================================================"
    echo "[$(date)] BLUR News seed=${SEED}..."
    CUDA_VISIBLE_DEVICES=0 python src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/muse/blurnpo_muse.yaml \
        data_split=News \
        task_name=muse_Llama-2-7b-hf_News_BLURNPO_s${SEED} \
        retain_logs_path=saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json \
        trainer.method_args.beta=0.05 \
        trainer.args.learning_rate=2.5e-5 \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.seed=${SEED}
    echo "[$(date)] Done BLUR News seed=${SEED}"
done

# ============================================================
# MUSE Books BLUR (4 seeds)
# ============================================================
for SEED in "${SEEDS[@]}"; do
    OUT="${BASE_DIR}/muse_Llama-2-7b-hf_Books_BLURNPO_s${SEED}"
    if [ -f "${OUT}/model.safetensors.index.json" ] || [ -f "${OUT}/model.safetensors" ]; then
        echo "[SKIP] ${OUT##*/} already completed"
        continue
    fi
    echo "============================================================"
    echo "[$(date)] BLUR Books seed=${SEED}..."
    CUDA_VISIBLE_DEVICES=0 python src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/muse/blurnpo_muse.yaml \
        data_split=Books \
        task_name=muse_Llama-2-7b-hf_Books_BLURNPO_s${SEED} \
        retain_logs_path=saves/eval/muse_Llama-2-7b-hf_Books_retrain/MUSE_EVAL.json \
        trainer.method_args.beta=0.4 \
        trainer.args.learning_rate=1e-5 \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.seed=${SEED}
    echo "[$(date)] Done BLUR Books seed=${SEED}"
done

# ============================================================
# KnowUnDo Copyright BLUR (4 seeds)
# ============================================================
for SEED in "${SEEDS[@]}"; do
    OUT="${BASE_DIR}/knowundo_BLURNPO_copyright_s${SEED}"
    if [ -f "${OUT}/model.safetensors.index.json" ] || [ -f "${OUT}/model.safetensors" ]; then
        echo "[SKIP] ${OUT##*/} already completed"
        continue
    fi
    echo "============================================================"
    echo "[$(date)] BLUR KnowUnDo copyright seed=${SEED}..."
    CUDA_VISIBLE_DEVICES=0 python src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/knowundo/blurnpo \
        domain=copyright \
        task_name=knowundo_BLURNPO_copyright_s${SEED} \
        trainer.args.eval_strategy=no \
        trainer.args.eval_on_start=false \
        trainer.args.seed=${SEED}
    echo "[$(date)] Done BLUR KnowUnDo copyright seed=${SEED}"
done

# ============================================================
# KnowUnDo Privacy BLUR (4 seeds)
# ============================================================
for SEED in "${SEEDS[@]}"; do
    OUT="${BASE_DIR}/knowundo_BLURNPO_privacy_s${SEED}"
    if [ -f "${OUT}/model.safetensors.index.json" ] || [ -f "${OUT}/model.safetensors" ]; then
        echo "[SKIP] ${OUT##*/} already completed"
        continue
    fi
    echo "============================================================"
    echo "[$(date)] BLUR KnowUnDo privacy seed=${SEED}..."
    CUDA_VISIBLE_DEVICES=0 python src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/knowundo/blurnpo \
        domain=privacy \
        task_name=knowundo_BLURNPO_privacy_s${SEED} \
        trainer.args.eval_strategy=no \
        trainer.args.eval_on_start=false \
        trainer.args.seed=${SEED}
    echo "[$(date)] Done BLUR KnowUnDo privacy seed=${SEED}"
done

echo "============================================================"
echo "[$(date)] ALL BLUR RUNS DONE"
