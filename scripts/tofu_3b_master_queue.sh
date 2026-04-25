#!/bin/bash
# Master queue: (1) Ours 5-seed 3B, (2) Baselines forget01 3B retrain
# LLM judge launched in background after each GPU batch completes
# Usage: nohup bash scripts/tofu_3b_master_queue.sh > saves/unlearn/master_queue.log 2>&1 &
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

echo "================================================================"
echo " MASTER QUEUE — started $(date)"
echo "================================================================"

# ══════════════════════════════════════════════════════════════
# PHASE 1: Ours 5-seed (forget01, forget05, forget10)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: LoRA-BiAL-Adaptive 3B — 5 seeds × 3 splits      ║"
echo "╚══════════════════════════════════════════════════════════════╝"

bash scripts/tofu_adaptive_3b_seeds.sh

echo ""
echo "[PHASE 1 DONE] $(date) — launching LLM judge on all adaptive results"

# LLM judge on ours (all 3 splits, background — no GPU needed)
nohup bash scripts/llm_judge_run.sh \
    --model 3B --splits "forget01 forget05 forget10" --ours --workers 32 \
    > saves/unlearn/llm_judge_ours_3b.log 2>&1 &
JUDGE_OURS_PID=$!
echo "[LLM JUDGE] ours PID=$JUDGE_OURS_PID"

# ══════════════════════════════════════════════════════════════
# PHASE 2: Baselines forget01 retrain (7 methods × 5 seeds)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: Baselines 3B forget01 — 7 methods × 5 seeds     ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# The baselines script processes all QUEUE entries, but skip logic
# means only forget01 3B will actually run (others already in CSV).
# To be safe, create a dedicated forget01-only run:

SEEDS=(42 123 456 789 1337)
TRAINERS="GradAscent GradDiff NPO SimNPO RMU BLURNPO PDU"
MODEL="Llama-3.2-3B-Instruct"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
SPLIT="forget01"
RETAIN="retain99"
HOLDOUT="holdout01"
RETAIN_LOGS="saves/eval/tofu_${MODEL}_${RETAIN}/TOFU_EVAL.json"
BSZ=4
ACCUM=8

PDU_ARGS="trainer.method_args.alpha=100 trainer.method_args.retain_loss_eps=0.3 trainer.method_args.dual_step_size=5 trainer.method_args.dual_warmup_epochs=5"

total=0; skip=0; fail=0; done_count=0

for trainer in $TRAINERS; do
    for seed in "${SEEDS[@]}"; do
        total=$((total + 1))

        task_name="bs32_${MODEL}_${SPLIT}_${trainer}_s${seed}"
        outdir="saves/unlearn/${task_name}"
        eval_json="${outdir}/evals/TOFU_EVAL.json"

        # Skip if eval JSON already exists
        if [[ -f "$eval_json" ]]; then
            echo "[SKIP] $task_name — eval exists"
            skip=$((skip + 1))
            continue
        fi

        echo "────────────────────────────────────────"
        echo "[RUN] $task_name"
        echo "────────────────────────────────────────"

        extra_args=""
        [[ "$trainer" == "PDU" ]] && extra_args="$PDU_ARGS"

        mkdir -p "$outdir"

        # TRAIN
        echo "[TRAIN] $(date '+%H:%M:%S') $task_name"
        if ! python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default.yaml \
            trainer=${trainer} \
            task_name=${task_name} \
            model=${MODEL} \
            forget_split=${SPLIT} \
            retain_split=${RETAIN} \
            model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
            retain_logs_path=${RETAIN_LOGS} \
            trainer.args.per_device_train_batch_size=${BSZ} \
            trainer.args.gradient_accumulation_steps=${ACCUM} \
            trainer.args.gradient_checkpointing=true \
            trainer.args.eval_strategy=no \
            trainer.args.do_eval=false \
            trainer.args.eval_on_start=false \
            trainer.args.seed=${seed} \
            ${extra_args} ; then
            echo "[TRAIN FAILED] $task_name"
            fail=$((fail + 1))
            rm -rf "$outdir" 2>/dev/null
            continue
        fi

        # EVAL
        echo "[EVAL] $(date '+%H:%M:%S') $task_name"
        if ! python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${SPLIT} \
            holdout_split=${HOLDOUT} \
            model=${MODEL} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=${outdir} \
            paths.output_dir=${outdir}/evals \
            retain_logs_path=${RETAIN_LOGS} ; then
            echo "[EVAL FAILED] $task_name"
            fail=$((fail + 1))
            continue
        fi

        if [[ -f "$eval_json" ]]; then
            done_count=$((done_count + 1))
            echo "[OK] $task_name"
        fi

        # Clean model weights — keep evals for LLM judge
        find "$outdir" -maxdepth 1 -name "*.safetensors" -delete 2>/dev/null
        find "$outdir" -maxdepth 1 -name "*.bin" -delete 2>/dev/null
        rm -f "$outdir"/config.json "$outdir"/generation_config* \
              "$outdir"/tokenizer* "$outdir"/special_tokens* "$outdir"/added_tokens* 2>/dev/null

        echo "[CLEANED] $task_name"
        echo ""
    done
done

echo ""
echo "[PHASE 2 DONE] $(date) — total=$total skip=$skip done=$done_count fail=$fail"

# LLM judge on forget01 baselines (background)
echo "[LLM JUDGE] Launching forget01 baselines judge..."
nohup bash scripts/llm_judge_run.sh \
    --model 3B --splits "forget01" --workers 32 \
    > saves/unlearn/llm_judge_baselines_f01.log 2>&1 &
JUDGE_BL_PID=$!
echo "[LLM JUDGE] baselines PID=$JUDGE_BL_PID"

# ══════════════════════════════════════════════════════════════
# Wait for all LLM judges
# ══════════════════════════════════════════════════════════════
echo ""
echo "Waiting for LLM judge processes..."
wait $JUDGE_OURS_PID 2>/dev/null && echo "[LLM JUDGE] ours finished" || echo "[LLM JUDGE] ours exited with error"
wait $JUDGE_BL_PID 2>/dev/null && echo "[LLM JUDGE] baselines finished" || echo "[LLM JUDGE] baselines exited with error"

echo ""
echo "================================================================"
echo " ALL DONE — $(date)"
echo "================================================================"
