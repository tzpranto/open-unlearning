#!/bin/bash
set -e
cd /datadrive/forked/open-unlearning

echo "[$(date -u)] === BLURNPO 5-fold: all benchmarks ==="

SEEDS="42 123 456 789 1024"

# --- KnowUnDo ---
for domain in copyright privacy; do
    for seed in $SEEDS; do
        TASK="knowundo_${domain}_blurnpo_s${seed}"
        if [ -d "saves/unlearn/${TASK}" ] && ls saves/unlearn/${TASK}/model-*.safetensors 1>/dev/null 2>&1; then
            echo "SKIP $TASK (exists)"
            continue
        fi
        echo "[$(date -u)] Training $TASK"
        rm -rf "saves/unlearn/${TASK}"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py \
            --config-name=unlearn.yaml \
            experiment=unlearn/knowundo/blurnpo.yaml \
            domain=${domain} \
            trainer.args.seed=${seed} \
            trainer.args.gradient_checkpointing=true \
            trainer.args.per_device_train_batch_size=1 \
            trainer.args.gradient_accumulation_steps=32 \
            trainer.args.eval_strategy=no \
            trainer.args.do_eval=false \
            trainer.args.eval_on_start=false \
            task_name=${TASK}
        echo "[$(date -u)] Done $TASK"
    done
done

# --- MUSE News ---
for seed in $SEEDS; do
    TASK="muse_Llama-2-7b-hf_News_BLURNPO_s${seed}"
    if [ -d "saves/unlearn/${TASK}" ] && ls saves/unlearn/${TASK}/model-*.safetensors 1>/dev/null 2>&1; then
        echo "SKIP $TASK (exists)"
        continue
    fi
    echo "[$(date -u)] Training $TASK"
    rm -rf "saves/unlearn/${TASK}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/muse/blurnpo_muse.yaml \
        data_split=raw \
        trainer.args.seed=${seed} \
        trainer.args.per_device_train_batch_size=2 \
        trainer.args.gradient_accumulation_steps=16 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        task_name=${TASK}
    echo "[$(date -u)] Done $TASK"
done

# --- MUSE Books ---
for seed in $SEEDS; do
    TASK="muse_Llama-2-7b-hf_Books_BLURNPO_s${seed}"
    if [ -d "saves/unlearn/${TASK}" ] && ls saves/unlearn/${TASK}/model-*.safetensors 1>/dev/null 2>&1; then
        echo "SKIP $TASK (exists)"
        continue
    fi
    echo "[$(date -u)] Training $TASK"
    rm -rf "saves/unlearn/${TASK}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/muse/blurnpo_muse.yaml \
        data_split=Books \
        model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-Books_target \
        trainer.args.seed=${seed} \
        trainer.args.per_device_train_batch_size=2 \
        trainer.args.gradient_accumulation_steps=16 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        task_name=${TASK}
    echo "[$(date -u)] Done $TASK"
done

echo "[$(date -u)] === ALL BLURNPO DONE ==="
