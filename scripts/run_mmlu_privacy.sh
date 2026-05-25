#!/bin/bash
# Run MMLU eval for all 40 KnowUnDo privacy sweep models, 8 GPUs parallel
set -uo pipefail
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
cd /data/open-unlearning

SAVES="saves/unlearn"
LOGFILE="logs/mmlu_privacy.log"
mkdir -p logs

exec > >(tee -a "$LOGFILE") 2>&1
echo "Started MMLU eval: $(date '+%Y-%m-%d %H:%M:%S')"

# Collect all model dirs that need MMLU
declare -a TASKS=()
for dir in ${SAVES}/privacy_k3_sweep_*; do
    if [[ -d "$dir" ]] && [[ ! -f "${dir}/evals/LMEval_SUMMARY.json" ]]; then
        TASKS+=("$dir")
    fi
done

echo "Models needing MMLU: ${#TASKS[@]}"

run_mmlu_one() {
    local gpu=$1
    local model_dir=$2
    local task_name=$(basename "$model_dir")
    local out_dir="${model_dir}/evals"
    local summary="${out_dir}/LMEval_SUMMARY.json"

    mkdir -p "$out_dir"

    echo "[GPU $gpu] EVAL $task_name"
    CUDA_VISIBLE_DEVICES=$gpu python3 << PYEOF
import sys, types, json, torch, logging
logging.basicConfig(level=logging.WARNING)

# Patch vllm
fake_vllm = types.ModuleType('vllm')
fake_vllm.LLM = None
fake_vllm.SamplingParams = None
sys.modules['vllm'] = fake_vllm
sys.modules['vllm.lora'] = types.ModuleType('vllm.lora')
sys.modules['vllm.lora.request'] = types.ModuleType('vllm.lora.request')
sys.modules['vllm.lora.request'].LoRARequest = None
sys.modules['vllm.sampling_params'] = types.ModuleType('vllm.sampling_params')
sys.modules['vllm.sampling_params'].GuidedDecodingParams = None

from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

model_path = "${model_dir}"
output_path = "${summary}"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

lm = HFLM(pretrained=model, tokenizer=tokenizer)
results = simple_evaluate(model=lm, tasks=["mmlu"], task_manager=TaskManager(), batch_size=16)

summary = {}
group_metrics = results.get("groups", {}).get("mmlu", {})
for k, v in group_metrics.items():
    if k == "alias":
        continue
    base = k.split(",", 1)[0].strip()
    try:
        summary[f"mmlu/{base}"] = float(v)
    except (TypeError, ValueError):
        summary[f"mmlu/{base}"] = v

with open(output_path, 'w') as f:
    json.dump(summary, f, indent=4)
print(f"mmlu/acc={summary.get('mmlu/acc', 'N/A')}")
PYEOF

    if [[ $? -eq 0 ]]; then
        echo "[GPU $gpu] DONE $task_name"
    else
        echo "[GPU $gpu] FAILED $task_name"
    fi
}

# Run in batches of 8
total=${#TASKS[@]}
batch=0
while [[ $batch -lt $total ]]; do
    pids=()
    for gpu in 0 1 2 3 4 5 6 7; do
        idx=$((batch + gpu))
        if [[ $idx -ge $total ]]; then
            break
        fi
        run_mmlu_one $gpu "${TASKS[$idx]}" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait $pid || true
    done

    batch=$((batch + 8))
    echo "Batch done. Progress: $batch/$total at $(date '+%H:%M:%S')"
done

echo ""
echo "ALL MMLU EVALS COMPLETE at $(date '+%Y-%m-%d %H:%M:%S')"
