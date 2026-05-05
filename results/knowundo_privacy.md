# KnowUnDo Privacy (Llama-2-7b-chat-hf)

Updated: 2026-05-05

Gold target: forget should decrease (Acc↓, ROUGE↓), retain should stay high (Acc↑, ROUGE↑), general knowledge preserved (MMLU↑).

## Results (seed=42)

HM = harmonic_mean(1 - forget_ROUGE, retain_ROUGE, mmlu). All metrics in 0-1 range.

| Method           | forget_Acc↓ | forget_ROUGE↓ | retain_Acc↑ | retain_ROUGE↑ | MMLU↑  | HM↑   |
| ---------------- | ----------- | ------------- | ----------- | ------------- | ------ | ----- |
| FT (target)      | 0.9395      | 0.6652        | 0.9488      | 0.6975        | 0.4448 | 0.450 |
| GradAscent       | 0.0413      | 0.0000        | 0.0292      | 0.0004        | 0.3256 | -     |
| GradDiff         | 0.0701      | 0.0561        | 0.7525      | 0.4267        | 0.4341 | 0.526 |
| NPO              | 0.6338      | 0.3131        | 0.7857      | 0.4313        | 0.4494 | 0.500 |
| SimNPO           | 0.4858      | 0.2893        | 0.8886      | 0.5402        | 0.4407 | 0.543 |
| RMU              | 0.5781      | 0.0720        | 0.6688      | 0.0835        | 0.3607 | 0.190 |
| BLURNPO          | 0.1026      | 0.0384        | 0.7037      | 0.3622        | 0.4504 | 0.498 |
| PDU              | 0.0589      | 0.0315        | 0.8086      | 0.4681        | 0.4413 | 0.552 |
| **BLADE**        | **0.2094**  | **0.0747**    | **0.9199**  | **0.6207**    | 0.4353 | **0.601** |
| MemFlex          | 0.1318      | 0.0747        | 0.6949      | 0.3038        | 0.3431 | 0.412 |

## LLM Judge (Claude Sonnet 4.6, 0-1 scale)

FL = Forget Leakage (lower = better forgetting), RA = Retain Accuracy (higher = better), RQ = Response Quality (higher = better).

| Method           | FL↓   | RA↑   | RQ↑   | forget_RQ↑ | retain_RQ↑ |
| ---------------- | ----- | ----- | ----- | ---------- | ---------- |
| FT (target)      | 0.882 | 0.847 | 0.975 | 0.959      | 0.991      |
| GradAscent       | 0.000 | 0.000 | 0.000 | 0.000      | 0.000      |
| GradDiff         | 0.028 | 0.495 | 0.379 | 0.036      | 0.727      |
| NPO              | 0.282 | 0.555 | 0.897 | 0.800      | 0.996      |
| SimNPO           | 0.177 | 0.657 | 0.548 | 0.186      | 0.916      |
| RMU              | 0.009 | 0.014 | 0.081 | 0.054      | 0.106      |
| BLURNPO          | 0.036 | 0.421 | 0.457 | 0.082      | 0.838      |
| PDU              | 0.018 | 0.574 | 0.379 | 0.022      | 0.741      |
| **BLADE**        | **0.068** | **0.732** | **0.477** | 0.072 | **0.889** |
| MemFlex          | 0.036 | 0.343 | 0.346 | 0.082      | 0.616      |

## Results (5-fold, seeds=42,123,456,789,1024)

| Method           | forget_Acc↓     | forget_ROUGE↓   | retain_Acc↑     | retain_ROUGE↑   |
| ---------------- | --------------- | --------------- | --------------- | --------------- |
| GradAscent       | 0.017 ± 0.019   | 0.004 ± 0.006   | 0.012 ± 0.013   | 0.006 ± 0.009   |
| GradDiff         | 0.049 ± 0.015   | 0.039 ± 0.015   | 0.708 ± 0.030   | 0.406 ± 0.023   |
| NPO              | 0.646 ± 0.019   | 0.322 ± 0.024   | 0.790 ± 0.002   | 0.426 ± 0.006   |
| SimNPO           | 0.516 ± 0.037   | 0.302 ± 0.032   | 0.889 ± 0.005   | 0.557 ± 0.014   |
| RMU              | 0.568 ± 0.007   | 0.072 ± 0.004   | 0.661 ± 0.005   | 0.079 ± 0.007   |
| PDU              | 0.055 ± 0.015   | 0.022 ± 0.007   | 0.813 ± 0.019   | 0.452 ± 0.023   |

## LLM Judge (5-fold, seeds=42,123,456,789,1024)

FL = Forget Leakage (lower = better), RA = Retain Accuracy (higher = better). Haiku 3.5 judge.

| Method           | FL↓           | RA↑           | RQ↑           | forget_RQ↑    | retain_RQ↑    |
| ---------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| GradAscent       | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   |
| GradDiff         | 0.04 ± 0.02   | 1.08 ± 0.07   | 0.66 ± 0.08   | 0.04 ± 0.02   | 1.28 ± 0.15   |
| NPO              | 0.99 ± 0.23   | 1.40 ± 0.14   | 1.90 ± 0.06   | 1.84 ± 0.13   | 1.97 ± 0.02   |
| SimNPO           | 0.79 ± 0.23   | 1.55 ± 0.12   | 1.35 ± 0.20   | 0.81 ± 0.37   | 1.89 ± 0.04   |
| RMU              | 0.13 ± 0.06   | 0.17 ± 0.07   | 0.59 ± 0.21   | 0.66 ± 0.27   | 0.52 ± 0.15   |
| PDU              | 0.05 ± 0.03   | 1.25 ± 0.06   | 0.79 ± 0.03   | 0.06 ± 0.02   | 1.53 ± 0.04   |

## Notes

- Finetuned using original KnowUnDo code (LoRA r=8, alpha=16, 10 epochs, lr=1e-4)
- MemFlex localization uses original code on LoRA adapters, params mapped to full model
- All baselines use MUSE News unlearning params (lr=1e-5, 10 epochs, constant scheduler)
- MMLU evaluated via lm-evaluation-harness
- LLM Judge (seed=42): Claude Sonnet 4.6 via Bedrock, 218 items per method (110 forget + 108 retain)
- LLM Judge (5-fold): Claude Haiku 3.5 via Bedrock (Opus 4.7 refuses on copyrighted content)
- Privacy domain has much higher memorization than copyright (FT forget_ROUGE=0.665 vs 0.244)
- GA collapsed (all metrics near 0); RMU collapsed (retain_RQ=0.106)
- BLADE achieves best HM (0.601) and best LLM Judge RA (0.732): strong retain preservation + reasonable forgetting + MMLU intact
- BLADE's retain_ROUGE (0.621) nearly matches FT target (0.698), suggesting minimal collateral damage
- 5-fold tables: ALL COMPLETE (GA, GradDiff, NPO, SimNPO, RMU, PDU)
- BLURNPO OOM on 40GB A100 (model + ref_model deepcopy exceeds memory); only s42 available (ran on 96GB local)
- NPO privacy: FL=0.99 (near full leakage), barely forgets despite high retain
- SimNPO privacy: high forget_ROUGE (0.302) + high FL (0.79) = poor forgetting; retains well though
