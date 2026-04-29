"""MemFlex localization pre-computation for KnowUnDo.

Identifies which model parameters are specifically responsible for forget
knowledge vs retain knowledge using gradient-based analysis.
"""

import argparse
import sys
sys.path.insert(0, "src")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.knowundo import KnowUnDoDataset
from trainer.unlearn.memflex import MemFlexLocalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--domain", type=str, default="copyright")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--mu", type=float, default=0.92)
    parser.add_argument("--sigma", type=float, default=6e-4)
    parser.add_argument("--num_copies", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template_args = {
        "apply_chat_template": False,
        "user_start_tag": "[INST] ",
        "user_end_tag": " [/INST]",
        "asst_start_tag": "",
        "asst_end_tag": " ",
    }

    print(f"Loading forget dataset (domain={args.domain})...")
    forget_ds = KnowUnDoDataset(
        hf_args={"path": "zjunlp/KnowUnDo", "name": args.domain, "split": "unlearn", "trust_remote_code": True},
        template_args=template_args,
        tokenizer=tokenizer,
        subset="train",
        max_length=256,
    )

    print(f"Loading retain dataset (domain={args.domain})...")
    retain_ds = KnowUnDoDataset(
        hf_args={"path": "zjunlp/KnowUnDo", "name": args.domain, "split": "retention", "trust_remote_code": True},
        template_args=template_args,
        tokenizer=tokenizer,
        subset="train",
        max_length=256,
    )

    print(f"Running localization (mu={args.mu}, sigma={args.sigma}, copies={args.num_copies})...")
    localizer = MemFlexLocalize(
        model=model,
        tokenizer=tokenizer,
        forget_dataset=forget_ds,
        retain_dataset=retain_ds,
        device="cuda",
    )
    localizer.localize(mu=args.mu, sigma=args.sigma, output_path=args.output_path)
    print("Done!")


if __name__ == "__main__":
    main()
