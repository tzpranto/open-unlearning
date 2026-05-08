# KnowUnDo Copyright (Llama-2-7b-chat-hf)

Updated: 2026-05-08

Gold target: forget should decrease (Acc↓, ROUGE↓), retain should stay high (Acc↑, ROUGE↑), general knowledge preserved (MMLU↑).

## Results (5-fold, seeds=42,123,456,789,1024)

HM = harmonic_mean(1 - forget_ROUGE, retain_ROUGE, MMLU).

| Method           | forget_Acc↓     | forget_ROUGE↓   | retain_Acc↑     | retain_ROUGE↑   | MMLU↑           | HM↑             |
| ---------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| GradAscent       | 0.002 ± 0.002   | 0.000 ± 0.000   | 0.002 ± 0.003   | 0.000 ± 0.000   | 0.250 ± 0.018   | 0.000 ± 0.000   |
| GradDiff         | 0.002 ± 0.003   | 0.000 ± 0.001   | 0.642 ± 0.041   | 0.199 ± 0.008   | 0.386 ± 0.007   | 0.348 ± 0.007   |
| NPO              | 0.678 ± 0.024   | 0.189 ± 0.005   | 0.763 ± 0.016   | 0.289 ± 0.002   | 0.445 ± 0.004   | 0.433 ± 0.002   |
| SimNPO           | 0.303 ± 0.046   | 0.048 ± 0.036   | 0.788 ± 0.013   | 0.317 ± 0.004   | 0.435 ± 0.001   | 0.461 ± 0.005   |
| RMU              | 0.435 ± 0.008   | 0.076 ± 0.003   | 0.763 ± 0.003   | 0.201 ± 0.006   | 0.388 ± 0.002   | 0.347 ± 0.007   |
| PDU              | 0.045 ± 0.004   | 0.011 ± 0.003   | 0.836 ± 0.004   | 0.285 ± 0.008   | 0.442 ± 0.001   | 0.442 ± 0.006   |
| **BLADE**        | 0.332 ± 0.044   | 0.040 ± 0.006   | 0.856 ± 0.012   | 0.332 ± 0.006   | 0.449 ± 0.005   | **0.477 ± 0.004** |

## LLM Judge (5-fold, seeds=42,123,456,789,1024)

FL = Forget Leakage (lower = better), RA = Retain Accuracy (higher = better). Sonnet 4.6 judge (Opus 4.7 refuses on copyrighted content).
HM = harmonic_mean(1 - FL/2, RA/2, RQ/2). All normalized to 0-1.

| Method           | FL↓           | RA↑           | RQ↑           | forget_RQ↑    | retain_RQ↑    | HM↑           |
| ---------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| GradAscent       | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.000 ± 0.000 |
| GradDiff         | 0.06 ± 0.07   | 0.96 ± 0.16   | 0.63 ± 0.10   | 0.00 ± 0.01   | 0.84 ± 0.14   | 0.47 ± 0.06   |
| NPO              | 1.37 ± 0.45   | 1.77 ± 0.10   | 1.85 ± 0.19   | 1.82 ± 0.28   | 1.86 ± 0.16   | 0.49 ± 0.15   |
| SimNPO           | 0.29 ± 0.22   | 1.76 ± 0.13   | 1.41 ± 0.15   | 0.15 ± 0.12   | 1.85 ± 0.18   | 0.80 ± 0.04   |
| RMU              | 0.42 ± 0.22   | 1.12 ± 0.12   | 1.16 ± 0.10   | 0.44 ± 0.13   | 1.42 ± 0.10   | 0.62 ± 0.03   |
| PDU              | 0.08 ± 0.04   | 1.61 ± 0.07   | 1.18 ± 0.09   | 0.01 ± 0.01   | 1.58 ± 0.13   | 0.75 ± 0.03   |
| **BLADE**        | 0.24 ± 0.19   | 1.81 ± 0.03   | 1.42 ± 0.10   | 0.07 ± 0.03   | 1.89 ± 0.13   | **0.82 ± 0.03** |

## Single-seed only (seed=42)

Methods that could not be run 5-fold due to memory constraints.

| Method           | forget_Acc↓ | forget_ROUGE↓ | retain_Acc↑ | retain_ROUGE↑ | MMLU↑  | HM↑   |
| ---------------- | ----------- | ------------- | ----------- | ------------- | ------ | ----- |
| FT (target)      | 0.8543      | 0.2444        | 0.8941      | 0.3355        | 0.4390 | 0.456 |
| BLURNPO          | 0.4935      | 0.0229        | 0.6886      | 0.1401        | 0.4351 | 0.287 |

LLM Judge (Claude Sonnet 4.6, seed=42):

| Method           | FL↓   | RA↑   | RQ↑   | forget_RQ↑ | retain_RQ↑ | HM↑   |
| ---------------- | ----- | ----- | ----- | ---------- | ---------- | ----- |
| FT (target)      | 1.203 | 1.660 | 1.591 | 1.554      | 1.604      | 0.605 |
| BLURNPO          | 0.007 | 0.238 | 0.254 | 0.068      | 0.319      | 0.209 |

## Notes

- Finetuned using original KnowUnDo code (LoRA r=8, alpha=16, 10 epochs, lr=1e-4)
- All baselines use MUSE News unlearning params (lr=1e-5, 10 epochs, constant scheduler)
- BLADE unified: same hyperparams as MUSE News (eps_mul=3.2, eta_theta=3e-5, conv_patience=20, T=150)
- MMLU evaluated via lm-evaluation-harness (all 5-fold models)
- LLM Judge (all seeds): Claude Opus 4.6 via Bedrock (Opus 4.7 refuses on copyrighted content)
- BLURNPO OOM on 40GB A100 (model + ref_model deepcopy exceeds memory); only s42 available (ran on 96GB local)
- GA/GradDiff over-forget (retain_ROUGE=0, model collapsed)
- NPO has highest FL in judge (1.37) — barely forgets; high retain but useless for unlearning
