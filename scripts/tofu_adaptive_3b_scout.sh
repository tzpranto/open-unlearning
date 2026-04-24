#!/bin/bash
# LoRA-BiAL-Adaptive 3B scout — seed 42 only, all 3 splits
# Usage: nohup bash scripts/tofu_adaptive_3b_scout.sh > saves/unlearn/adaptive_3b_scout.log 2>&1 &
set -uo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /datadrive/forked/open-unlearning

SEED=42
MODEL="Llama-3.2-3B-Instruct"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"

SPLITS=(
    "forget01,retain99,holdout01,250"
    "forget05,retain95,holdout05,250"
    "forget10,retain90,holdout10,500"
)

for split_cfg in "${SPLITS[@]}"; do
    IFS=',' read -r SPLIT RETAIN HOLDOUT T <<< "$split_cfg"
    RETAIN_LOGS="saves/eval/tofu_${MODEL}_${RETAIN}/TOFU_EVAL.json"

    task_name="adaptive_${MODEL}_${SPLIT}_s${SEED}"
    outdir="saves/unlearn/${task_name}"

    echo "════════════════════════════════════════"
    echo "[RUN] $task_name (T=$T)"
    echo "════════════════════════════════════════"

    mkdir -p "$outdir"

    # TRAIN
    echo "[TRAIN] $(date '+%H:%M:%S') $task_name"
    if ! python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/lora_bial_3b.yaml \
        task_name="$task_name" \
        trainer=LoRABiALAdaptive \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
        forget_split=${SPLIT} retain_split=${RETAIN} holdout_split=${HOLDOUT} \
        trainer.args.eval_strategy=no trainer.args.do_eval=false trainer.args.eval_on_start=false \
        trainer.method_args.checkpoint_every_epoch=false \
        trainer.method_args.eta_theta=5e-5 trainer.method_args.T=${T} \
        trainer.method_args.epsilon=99.0 \
        trainer.method_args.epsilon_multiplier=0.85 \
        trainer.args.seed=${SEED} \
        retain_logs_path="$RETAIN_LOGS" 2>&1 | tee "${outdir}/train.log" ; then
        echo "[TRAIN FAILED] $task_name"
        continue
    fi

    # EVAL
    echo "[EVAL] $(date '+%H:%M:%S') $task_name"
    if ! python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=${SPLIT} holdout_split=${HOLDOUT} \
        model=${MODEL} \
        task_name="${task_name}" \
        model.model_args.pretrained_model_name_or_path="${outdir}" \
        paths.output_dir="${outdir}/evals" \
        retain_logs_path="$RETAIN_LOGS" ; then
        echo "[EVAL FAILED] $task_name"
        continue
    fi

    # Extract metrics
    eval_json="${outdir}/evals/TOFU_EVAL.json"
    python3 -c "
import json
from statistics import harmonic_mean
d = json.load(open('$eval_json'))
def val(x): return x['agg_value'] if isinstance(x, dict) else x
mu = val(d.get('model_utility', 0))
fq = val(d.get('forget_quality', d.get('mia_min_k', 0)))
es = val(d.get('extraction_strength', 0))
fp = val(d.get('fgt_Q_A_Prob', d.get('forget_Q_A_Prob', 0)))
fr = val(d.get('fgt_Q_A_ROUGE', d.get('forget_Q_A_ROUGE', 0)))
vals = [mu, 1-fp, 1-fr]
hm = harmonic_mean(vals) if all(v > 0 for v in vals) else 0.0
print(f'MU={mu:.4f} FQ={fq:.4f} ES={es:.4f} fgt_Prob={fp:.4f} fgt_ROUGE={fr:.4f} HM={hm:.4f}')
"
    echo "[DONE] $task_name"
    echo ""
done

echo "ALL DONE"
