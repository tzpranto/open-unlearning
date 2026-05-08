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

FL = Forget Leakage (lower = better), RA = Retain Accuracy (higher = better). Opus 4.6 judge (Opus 4.7 refuses on copyrighted content).
HM = harmonic_mean(1 - FL/2, RA/2, retain_RQ/2). All normalized to 0-1.

| Method           | FL↓           | RA↑           | RQ↑           | forget_RQ↑    | retain_RQ↑    | HM↑           |
| ---------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| GradAscent       | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   |
| GradDiff         | 0.00 ± 0.00   | 0.81 ± 0.10   | 0.39 ± 0.05   | 0.00 ± 0.00   | 0.53 ± 0.06   | 0.41 ± 0.04   |
| NPO              | 0.77 ± 0.10   | 1.74 ± 0.02   | 1.19 ± 0.02   | 1.31 ± 0.05   | 1.15 ± 0.03   | 0.66 ± 0.02   |
| SimNPO           | 0.05 ± 0.03   | 1.73 ± 0.03   | 0.84 ± 0.02   | 0.05 ± 0.01   | 1.12 ± 0.03   | 0.75 ± 0.01   |
| RMU              | 0.04 ± 0.02   | 1.01 ± 0.03   | 0.77 ± 0.04   | 0.14 ± 0.03   | 0.98 ± 0.06   | 0.60 ± 0.02   |
| PDU              | 0.01 ± 0.01   | 1.59 ± 0.05   | 0.69 ± 0.04   | 0.01 ± 0.01   | 0.93 ± 0.06   | 0.68 ± 0.03   |
| **BLADE**        | 0.04 ± 0.02   | 1.79 ± 0.02   | 0.88 ± 0.02   | 0.05 ± 0.01   | 1.17 ± 0.02   | **0.78 ± 0.01** |

## Single-seed only (seed=42)

Methods that could not be run 5-fold due to memory constraints.

| Method           | forget_Acc↓ | forget_ROUGE↓ | retain_Acc↑ | retain_ROUGE↑ | MMLU↑  | HM↑   |
| ---------------- | ----------- | ------------- | ----------- | ------------- | ------ | ----- |
| FT (target)      | 0.8543      | 0.2444        | 0.8941      | 0.3355        | 0.4390 | 0.456 |
| BLURNPO          | 0.4935      | 0.0229        | 0.6886      | 0.1401        | 0.4351 | 0.287 |

LLM Judge (Claude Opus 4.6, seed=42):

| Method           | FL↓   | RA↑   | RQ↑   | forget_RQ↑ | retain_RQ↑ | HM↑   |
| ---------------- | ----- | ----- | ----- | ---------- | ---------- | ----- |
| FT (target)      | 1.500 | 1.802 | 1.185 | 1.338      | 1.132      | 0.44  |

## Notes

- Finetuned using original KnowUnDo code (LoRA r=8, alpha=16, 10 epochs, lr=1e-4)
- All baselines use MUSE News unlearning params (lr=1e-5, 10 epochs, constant scheduler)
- BLADE unified: same hyperparams as MUSE News (eps_mul=3.2, eta_theta=3e-5, conv_patience=20, T=150)
- MMLU evaluated via lm-evaluation-harness (all 5-fold models)
- LLM Judge (all seeds): Claude Opus 4.6 (Opus 4.7 safety guardrails refuse to evaluate copyrighted content, so we use Opus 4.6)
- BLURNPO OOM on 40GB A100 (model + ref_model deepcopy exceeds memory); only s42 available (ran on 96GB local)
- GA/GradDiff over-forget (retain_ROUGE=0, model collapsed)
- NPO has highest FL in judge (1.37) — barely forgets; high retain but useless for unlearning
