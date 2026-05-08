# KnowUnDo Privacy (Llama-2-7b-chat-hf)

Updated: 2026-05-06

Gold target: forget should decrease (Acc↓, ROUGE↓), retain should stay high (Acc↑, ROUGE↑), general knowledge preserved (MMLU↑).

## Results (5-fold, seeds=42,123,456,789,1024)

HM = harmonic_mean(1 - forget_ROUGE, retain_ROUGE, MMLU).

| Method           | forget_Acc↓     | forget_ROUGE↓   | retain_Acc↑     | retain_ROUGE↑   | MMLU↑           | HM↑             |
| ---------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| GradAscent       | 0.017 ± 0.019   | 0.004 ± 0.006   | 0.012 ± 0.013   | 0.006 ± 0.009   | 0.318 ± 0.016   | 0.017 ± 0.024   |
| GradDiff         | 0.049 ± 0.015   | 0.039 ± 0.015   | 0.708 ± 0.030   | 0.406 ± 0.023   | 0.428 ± 0.005   | 0.513 ± 0.011   |
| NPO              | 0.646 ± 0.019   | 0.322 ± 0.024   | 0.790 ± 0.002   | 0.426 ± 0.006   | 0.450 ± 0.004   | 0.496 ± 0.005   |
| SimNPO           | 0.516 ± 0.037   | 0.302 ± 0.032   | 0.889 ± 0.005   | 0.557 ± 0.014   | 0.438 ± 0.002   | 0.544 ± 0.008   |
| RMU              | 0.568 ± 0.007   | 0.072 ± 0.004   | 0.661 ± 0.005   | 0.079 ± 0.007   | 0.361 ± 0.002   | 0.182 ± 0.012   |
| PDU              | 0.055 ± 0.015   | 0.022 ± 0.007   | 0.813 ± 0.019   | 0.452 ± 0.023   | 0.444 ± 0.001   | 0.546 ± 0.011   |
| **BLADE**        | 0.194 ± 0.030   | 0.075 ± 0.027   | 0.930 ± 0.022   | 0.623 ± 0.031   | 0.439 ± 0.003   | **0.604 ± 0.010** |

## LLM Judge (5-fold, seeds=42,123,456,789,1024)

FL = Forget Leakage (lower = better), RA = Retain Accuracy (higher = better). Haiku 3.5 judge.
HM = harmonic_mean(1 - FL/2, RA/2, RQ/2). All normalized to 0-1.

| Method           | FL↓           | RA↑           | RQ↑           | forget_RQ↑    | retain_RQ↑    | HM↑           |
| ---------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| GradAscent       | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.000 ± 0.000 |
| GradDiff         | 0.04 ± 0.02   | 1.08 ± 0.07   | 0.66 ± 0.08   | 0.04 ± 0.02   | 1.28 ± 0.15   | 0.50 ± 0.04   |
| NPO              | 0.99 ± 0.23   | 1.40 ± 0.14   | 1.90 ± 0.06   | 1.84 ± 0.13   | 1.97 ± 0.02   | 0.66 ± 0.04   |
| SimNPO           | 0.79 ± 0.23   | 1.55 ± 0.12   | 1.35 ± 0.20   | 0.81 ± 0.37   | 1.89 ± 0.04   | 0.66 ± 0.02   |
| RMU              | 0.13 ± 0.06   | 0.17 ± 0.07   | 0.59 ± 0.21   | 0.66 ± 0.27   | 0.52 ± 0.15   | 0.18 ± 0.07   |
| PDU              | 0.05 ± 0.03   | 1.25 ± 0.06   | 0.79 ± 0.03   | 0.06 ± 0.02   | 1.53 ± 0.04   | 0.58 ± 0.02   |
| **BLADE**        | 0.20 ± 0.12   | 1.50 ± 0.07   | 0.95 ± 0.10   | 0.18 ± 0.12   | 1.74 ± 0.09   | **0.66 ± 0.03** |

## Single-seed only (seed=42)

Methods that could not be run 5-fold due to memory constraints.

| Method           | forget_Acc↓ | forget_ROUGE↓ | retain_Acc↑ | retain_ROUGE↑ | MMLU↑  | HM↑   |
| ---------------- | ----------- | ------------- | ----------- | ------------- | ------ | ----- |
| FT (target)      | 0.9395      | 0.6652        | 0.9488      | 0.6975        | 0.4448 | 0.450 |
| BLURNPO          | 0.1026      | 0.0384        | 0.7037      | 0.3622        | 0.4504 | 0.498 |
| MemFlex          | 0.1318      | 0.0747        | 0.6949      | 0.3038        | 0.3431 | 0.412 |

LLM Judge (Claude Sonnet 4.6, seed=42):

| Method           | FL↓   | RA↑   | RQ↑   | forget_RQ↑ | retain_RQ↑ | HM↑   |
| ---------------- | ----- | ----- | ----- | ---------- | ---------- | ----- |
| FT (target)      | 1.718 | 1.769 | 1.940 | 1.900      | 1.981      | 0.325 |
| BLURNPO          | 0.036 | 0.421 | 0.457 | 0.082      | 0.838      | 0.379 |
| MemFlex          | 0.036 | 0.343 | 0.346 | 0.082      | 0.616      | 0.305 |

## Notes

- Finetuned using original KnowUnDo code (LoRA r=8, alpha=16, 10 epochs, lr=1e-4)
- MemFlex localization uses original code on LoRA adapters, params mapped to full model
- All baselines use MUSE News unlearning params (lr=1e-5, 10 epochs, constant scheduler)
- MMLU evaluated via lm-evaluation-harness (per-seed, all 5-fold models)
- LLM Judge (5-fold): Claude Haiku 3.5 via Bedrock (Opus 4.7 refuses on copyrighted content)
- Privacy domain has much higher memorization than copyright (FT forget_ROUGE=0.665 vs 0.244)
- GA collapsed (all metrics near 0); RMU collapsed (retain_RQ=0.106)
- BLURNPO OOM on 40GB A100 (model + ref_model deepcopy exceeds memory); only s42 available (ran on 96GB local)
- NPO privacy: FL=0.99 (near full leakage), barely forgets despite high retain
- SimNPO privacy: high forget_ROUGE (0.302) + high FL (0.79) = poor forgetting; retains well though
- BLADE 5-fold: best retain_ROUGE (0.623 ± 0.031, nearly matches FT=0.698); RA=1.50 competitive with SimNPO (1.55) but much lower FL (0.20 vs 0.79)
