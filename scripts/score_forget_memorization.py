"""
Memorization scoring for MUSE News forget sequences.

Proxy mechanism:
  memorization_score(seq) = loss_base(seq) / loss_finetuned(seq)
  = how much MORE the fine-tuned model memorized this sequence vs the base model.

High score → sequence is specifically memorized by fine-tuning → good forget target
             (like "Hogwarts" — specific, memorized, should be forgotten)
Low score  → generic text shared with retain → skip
             (like "boarding school" — generic, won't help forgetting, may hurt retain)

Also computes NPO-proxy score:
  npo_score(seq) = log(p_base / p_finetuned) = log_ratio
  This is exactly what NPO maximizes during training. High npo_score = high-value forget target.

Outputs:
  - trace_analysis/figures/traces/analysis/forget_memorization_scores.json
  - Ranked list of forget sequence indices by memorization score
  - "hard_forget" JSONL: top-K sequences saved locally for targeted training
"""

import argparse
import json
import math
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_sequence_loss(model, tokenizer, text, max_length=1024, device="cuda"):
    """Compute average per-token CE loss for a sequence."""
    inputs = tokenizer(
        text, return_tensors="pt", max_length=max_length,
        truncation=True, padding=False
    ).to(device)
    if inputs["input_ids"].shape[1] < 2:
        return None
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()  # average CE per token

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned_model", default="muse-bench/MUSE-News_target",
                        help="Fine-tuned model (what we want to unlearn FROM)")
    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf",
                        help="Base model (what the model was before fine-tuning)")
    parser.add_argument("--tokenizer_path", default=None,
                        help="Override tokenizer path (use when finetuned_model has no tokenizer files)")
    parser.add_argument("--dataset", default="muse-bench/MUSE-News")
    parser.add_argument("--config", default="raw",
                        help="Dataset config name (e.g. 'raw')")
    parser.add_argument("--split", default="forget",
                        help="Forget split name within config")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of top memorized sequences to save as hard_forget")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", default="trace_analysis/figures/traces/analysis")
    parser.add_argument("--hard_forget_path", default="data/hard_forget_news.jsonl",
                        help="Path to save top-K memorized sequences")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading fine-tuned model: {args.finetuned_model}")
    # Use explicit tokenizer_path if provided (muse-bench models have no tokenizer files).
    # Fall back to base_model tokenizer since finetuned model shares the same vocab.
    tok_src = args.tokenizer_path or args.base_model
    print(f"Loading tokenizer from: {tok_src}")
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    finetuned_model = AutoModelForCausalLM.from_pretrained(
        args.finetuned_model, torch_dtype=torch.bfloat16
    ).to(device)
    finetuned_model.eval()

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16
    ).to(device)
    base_model.eval()

    print(f"Loading forget dataset: {args.dataset} config={args.config} split={args.split}")
    dataset = load_dataset(args.dataset, args.config, split=args.split)
    text_col = "text" if "text" in dataset.column_names else dataset.column_names[0]
    sequences = [row[text_col] for row in dataset]
    print(f"Total forget sequences: {len(sequences)}")

    scores = []
    for i, text in enumerate(sequences):
        if i % 50 == 0:
            print(f"  Scoring {i}/{len(sequences)}...")

        loss_ft = compute_sequence_loss(finetuned_model, tokenizer, text, args.max_length, device)
        loss_base = compute_sequence_loss(base_model, tokenizer, text, args.max_length, device)

        if loss_ft is None or loss_base is None:
            scores.append({"idx": i, "loss_finetuned": None, "loss_base": None,
                           "memorization_score": 0.0, "npo_log_ratio": 0.0,
                           "text_preview": text[:100]})
            continue

        # memorization_score: how much more the fine-tuned model memorized this vs base
        # Higher = fine-tuned model is more confident = more memorized
        memorization_score = loss_base / loss_ft  # > 1 means fine-tuned is better (memorized)

        # NPO log ratio: log(p_ft / p_base) = -(loss_ft - loss_base) per token
        # Higher = fine-tuned assigns much higher probability than base = definitely memorized
        npo_log_ratio = loss_base - loss_ft  # positive = fine-tuned lower loss = memorized

        scores.append({
            "idx": i,
            "loss_finetuned": round(loss_ft, 4),
            "loss_base": round(loss_base, 4),
            "memorization_score": round(memorization_score, 4),
            "npo_log_ratio": round(npo_log_ratio, 4),
            "text_preview": text[:120]
        })

    # Sort by NPO log ratio (most memorized first)
    scores_sorted = sorted(scores, key=lambda x: x["npo_log_ratio"], reverse=True)

    # Save full results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "forget_memorization_scores.json")
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "finetuned_model": args.finetuned_model,
                "base_model": args.base_model,
                "dataset": args.dataset,
                "split": args.split,
                "total_sequences": len(sequences),
                "top_k": args.top_k,
            },
            "top_k_indices": [s["idx"] for s in scores_sorted[:args.top_k]],
            "scores": scores_sorted
        }, f, indent=2)
    print(f"\nScores saved to {out_path}")

    # Print top-20 summary
    print(f"\nTop-20 most memorized sequences (by NPO log ratio):")
    print(f"{'idx':>5} {'loss_ft':>8} {'loss_base':>9} {'npo_ratio':>10}  preview")
    for s in scores_sorted[:20]:
        print(f"{s['idx']:>5} {s['loss_finetuned']:>8.4f} {s['loss_base']:>9.4f} "
              f"{s['npo_log_ratio']:>10.4f}  {s['text_preview'][:60]}")

    print(f"\nBottom-10 (least memorized / generic):")
    for s in scores_sorted[-10:]:
        print(f"{s['idx']:>5} {s['loss_finetuned']:>8.4f} {s['loss_base']:>9.4f} "
              f"{s['npo_log_ratio']:>10.4f}  {s['text_preview'][:60]}")

    # Save hard_forget subset (top-K sequences)
    os.makedirs(os.path.dirname(args.hard_forget_path) if os.path.dirname(args.hard_forget_path) else ".", exist_ok=True)
    top_indices = set(s["idx"] for s in scores_sorted[:args.top_k])
    hard_forget_seqs = [{"text": sequences[s["idx"]], "memorization_score": s["memorization_score"],
                         "npo_log_ratio": s["npo_log_ratio"], "original_idx": s["idx"]}
                        for s in scores_sorted[:args.top_k]]
    with open(args.hard_forget_path, "w") as f:
        for seq in hard_forget_seqs:
            f.write(json.dumps(seq) + "\n")
    print(f"\nTop-{args.top_k} hard forget sequences saved to {args.hard_forget_path}")

    # Statistics
    mem_scores = [s["memorization_score"] for s in scores if s["memorization_score"] > 0]
    npo_scores = [s["npo_log_ratio"] for s in scores]
    print(f"\nMemorization score stats:")
    print(f"  Mean: {sum(mem_scores)/len(mem_scores):.3f}")
    print(f"  Min: {min(mem_scores):.3f} | Max: {max(mem_scores):.3f}")
    print(f"  Sequences with score > 1.5 (highly memorized): {sum(1 for s in mem_scores if s > 1.5)}")
    print(f"  Sequences with score < 0.8 (generic/not memorized): {sum(1 for s in mem_scores if s < 0.8)}")
    print(f"\nNPO log ratio stats:")
    print(f"  Mean: {sum(npo_scores)/len(npo_scores):.3f}")
    print(f"  Sequences with ratio > 0.5 (well memorized): {sum(1 for s in npo_scores if s > 0.5)}")


if __name__ == "__main__":
    main()
