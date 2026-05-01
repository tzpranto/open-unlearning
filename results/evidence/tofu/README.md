# TOFU Evaluation Evidence

Raw evaluation outputs backing the tables in `results/tofu.md`.

## Structure

```
tofu/
├── eval_outputs/
│   ├── eval_results.json          # 181 baseline entries (7 methods × 3 splits × 2 models × ~5 seeds)
│   ├── blade_eval_results.json    # 30 BLADE/adaptive entries (3 splits × 2 models × 5 seeds)
│   ├── retrain_baselines.json     # 6 gold (retrain) baselines
│   ├── tofu_baselines.csv         # Aggregated baselines table (210 rows)
│   └── tofu_adaptive.csv          # BLADE results table (30 rows)
├── llm_judge/
│   └── tofu_llm_judge.csv         # LLM judge scores (229 rows)
└── reproduce_eval.sh              # Script to reproduce any eval from checkpoint
```

## Configuration

- Models: Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct
- Splits: forget01 (1%), forget05 (5%), forget10 (10%)
- Seeds: 42, 123, 456, 789, 1337
- Methods: GradAscent, GradDiff, NPO, SimNPO, RMU, PDU, BLURNPO, BLADE

## Eval Metrics

- `Model Utility` (MU): Harmonic mean of retain quality metrics (higher = better)
- `Forget Quality` (FQ): Aggregated forgetting quality (higher = better)
- `extraction_strength` (ES): Membership inference attack success (lower = better)
- `forget_ROUGE_L` / `forget_Probability`: Per-metric forget scores
- `retain_ROUGE_L` / `retain_Probability`: Per-metric retain scores
- `HM`: Harmonic mean of MU and FQ

## LLM Judge Metrics (CSV)

- `forget_leakage`: Knowledge leakage on forget set (lower = better)
- `retain_accuracy`: Factual accuracy on retain set (higher = better)
- `response_quality`: Overall response quality (higher = better)
- `forget_rq` / `retain_rq`: Response quality by split

## Reproducing

```bash
# Run eval for a specific baseline checkpoint
bash results/evidence/tofu/reproduce_eval.sh Llama-3.2-1B-Instruct forget01 GradAscent 42

# Requires: model checkpoint + retrain eval in saves/
```
