"""Run MMLU eval for all 40 KnowUnDo privacy sweep models.
Patches vllm import issue and runs on available GPUs.
"""
import sys
import types
import os
import json
import logging
from pathlib import Path

# Patch vllm before importing lm_eval
fake_vllm = types.ModuleType('vllm')
fake_vllm.LLM = None
fake_vllm.SamplingParams = None
sys.modules['vllm'] = fake_vllm
sys.modules['vllm.lora'] = types.ModuleType('vllm.lora')
sys.modules['vllm.lora.request'] = types.ModuleType('vllm.lora.request')
sys.modules['vllm.lora.request'].LoRARequest = None
sys.modules['vllm.sampling_params'] = types.ModuleType('vllm.sampling_params')
sys.modules['vllm.sampling_params'].GuidedDecodingParams = None

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAVES = Path("/data/open-unlearning/saves/unlearn")
GPU = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

SWEEPS = {
    'eps_mul': ['0.75', '1.0', '1.25', '1.5', '2.0', '2.5', '3.0', '3.2'],
    'tau': ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.9', '1.0'],
    'alpha_dual': ['0.01', '0.05', '0.1', '0.2', '0.3', '0.5', '0.7', '1.0'],
    'rho': ['0.01', '0.03', '0.05', '0.1', '0.2', '0.5', '1.0', '2.0'],
    'eta_in': ['1e-5', '2.5e-5', '5e-5', '1e-4', '2e-4', '5e-4', '1e-3', '2e-3'],
}


def run_mmlu(model_path: Path, output_path: Path):
    """Run MMLU evaluation on a single model."""
    if output_path.exists():
        logger.info(f"SKIP {model_path.name} (MMLU already exists)")
        return json.load(open(output_path))

    logger.info(f"EVAL MMLU: {model_path.name}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    task_manager = TaskManager()

    results = simple_evaluate(
        model=lm,
        tasks=["mmlu"],
        task_manager=task_manager,
        batch_size=16,
    )

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)

    logger.info(f"DONE {model_path.name}: mmlu/acc={summary.get('mmlu/acc', 'N/A')}")

    del model, lm
    torch.cuda.empty_cache()

    return summary


def main():
    tasks = []
    for sweep_name, values in SWEEPS.items():
        for val in values:
            task_name = f"privacy_k3_sweep_{sweep_name}_{val}"
            model_dir = SAVES / task_name
            mmlu_path = model_dir / "evals" / "LMEval_SUMMARY.json"
            if model_dir.exists():
                tasks.append((model_dir, mmlu_path))

    logger.info(f"Total models to eval: {len(tasks)}")

    done = 0
    for model_dir, mmlu_path in tasks:
        try:
            run_mmlu(model_dir, mmlu_path)
            done += 1
        except Exception as e:
            logger.error(f"FAILED {model_dir.name}: {e}")

    logger.info(f"Complete: {done}/{len(tasks)}")


if __name__ == "__main__":
    main()
