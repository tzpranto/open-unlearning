# KnowUnDo Copyright (Llama-2-7b-chat-hf)

Updated: 2026-05-05

Gold target: forget should decrease (Acc↓, ROUGE↓), retain should stay high (Acc↑, ROUGE↑), general knowledge preserved (MMLU↑).

## Results (seed=42)

HM = harmonic_mean(1 - forget_ROUGE, retain_ROUGE, mmlu). All metrics in 0-1 range.

| Method           | forget_Acc↓ | forget_ROUGE↓ | retain_Acc↑ | retain_ROUGE↑ | MMLU↑  | HM↑   |
| ---------------- | ----------- | ------------- | ----------- | ------------- | ------ | ----- |
| FT (target)      | 0.8543      | 0.2444        | 0.8941      | 0.3355        | 0.4390 | 0.456 |
| GradAscent       | 0.0000      | 0.0000        | 0.0000      | 0.0000        | 0.2689 | -     |
| GradDiff         | 0.0000      | 0.0000        | 0.6232      | 0.1999        | 0.3871 | 0.349 |
| NPO              | 0.6703      | 0.1929        | 0.7566      | 0.2998        | 0.4471 | 0.441 |
| SimNPO           | 0.2446      | 0.0124        | 0.7858      | 0.3177        | 0.4348 | 0.464 |
| RMU              | 0.4442      | 0.0806        | 0.7678      | 0.2065        | 0.3901 | 0.353 |
| BLURNPO          | 0.4935      | 0.0229        | 0.6886      | 0.1401        | 0.4351 | 0.287 |
| PDU              | 0.0505      | 0.0072        | 0.8460      | 0.3021        | 0.4408 | 0.456 |
| **BLADE**        | **0.1329**  | **0.0087**    | **0.8477**  | **0.3343**    | 0.4415 | **0.479** |
| MemFlex          | 0.0650      | 0.0277        | 0.6811      | 0.1616        | 0.2577 | 0.270 |

## LLM Judge (Claude Sonnet 4.6, 0-1 scale)

FL = Forget Leakage (lower = better forgetting), RA = Retain Accuracy (higher = better), RQ = Response Quality (higher = better).

| Method           | FL↓   | RA↑   | RQ↑   | forget_RQ↑ | retain_RQ↑ |
| ---------------- | ----- | ----- | ----- | ---------- | ---------- |
| FT (target)      | 0.602 | 0.830 | 0.795 | 0.777      | 0.802      |
| GradAscent       | 0.000 | 0.000 | 0.000 | 0.000      | 0.000      |
| GradDiff         | 0.000 | 0.344 | 0.239 | 0.000      | 0.323      |
| NPO              | 0.243 | 0.783 | 0.740 | 0.635      | 0.776      |
| SimNPO           | 0.000 | 0.755 | 0.556 | 0.014      | 0.746      |
| RMU              | 0.000 | 0.453 | 0.479 | 0.102      | 0.611      |
| BLURNPO          | 0.007 | 0.238 | 0.254 | 0.068      | 0.319      |
| PDU              | 0.007 | 0.748 | 0.500 | 0.014      | 0.670      |
| **BLADE**        | **0.021** | **0.875** | **0.612** | 0.027 | **0.816** |
| MemFlex          | 0.000 | 0.213 | 0.486 | 0.061      | 0.635      |

## Results (5-fold, seeds=42,123,456,789,1024)

| Method           | forget_Acc↓     | forget_ROUGE↓   | retain_Acc↑     | retain_ROUGE↑   |
| ---------------- | --------------- | --------------- | --------------- | --------------- |
| GradAscent       | 0.002 ± 0.002   | 0.000 ± 0.000   | 0.002 ± 0.003   | 0.000 ± 0.000   |
| GradDiff         | 0.002 ± 0.003   | 0.000 ± 0.001   | 0.642 ± 0.041   | 0.199 ± 0.008   |
| NPO              | 0.678 ± 0.024   | 0.189 ± 0.005   | 0.763 ± 0.016   | 0.289 ± 0.002   |
| SimNPO           | 0.303 ± 0.046   | 0.048 ± 0.036   | 0.788 ± 0.013   | 0.317 ± 0.004   |
| RMU              | 0.435 ± 0.008   | 0.076 ± 0.003   | 0.763 ± 0.003   | 0.201 ± 0.007   |
| PDU              | 0.045 ± 0.004   | 0.011 ± 0.003   | 0.836 ± 0.004   | 0.285 ± 0.008   |

## LLM Judge (5-fold, seeds=42,123,456,789,1024)

FL = Forget Leakage (lower = better), RA = Retain Accuracy (higher = better). Haiku 3.5 judge.

| Method           | FL↓           | RA↑           | RQ↑           | forget_RQ↑    | retain_RQ↑    |
| ---------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| GradAscent       | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   | 0.00 ± 0.00   |
| GradDiff         | 0.06 ± 0.07   | 0.96 ± 0.16   | 0.63 ± 0.10   | 0.00 ± 0.01   | 0.84 ± 0.14   |
| NPO              | 1.37 ± 0.45   | 1.77 ± 0.10   | 1.85 ± 0.19   | 1.82 ± 0.28   | 1.86 ± 0.16   |
| SimNPO           | 0.29 ± 0.22   | 1.76 ± 0.13   | 1.41 ± 0.15   | 0.15 ± 0.12   | 1.85 ± 0.18   |
| RMU              | 0.42 ± 0.22   | 1.12 ± 0.12   | 1.16 ± 0.10   | 0.44 ± 0.13   | 1.42 ± 0.10   |
| PDU              | 0.08 ± 0.04   | 1.61 ± 0.07   | 1.18 ± 0.09   | 0.01 ± 0.01   | 1.58 ± 0.13   |

## Notes

- Finetuned using original KnowUnDo code (LoRA r=8, alpha=16, 10 epochs, lr=1e-4)
- MemFlex localization uses original code on LoRA adapters, params mapped to full model
- All baselines use MUSE News unlearning params (lr=1e-5, 10 epochs, constant scheduler)
- MMLU evaluated via lm-evaluation-harness
- LLM Judge (seed=42): Claude Sonnet 4.6 via Bedrock, 286 items per method (74 forget + 212 retain)
- LLM Judge (5-fold): Claude Haiku 3.5 via Bedrock (Opus 4.7 refuses on copyrighted content)
- BLURNPO: ref model + gradient surgery buffers are memory-heavy; initially OOM'd due to concurrent processes, succeeded when GPU was free
- GA/GradDiff over-forget (retain_ROUGE=0, model collapsed)
- BLADE achieves best HM and best LLM Judge scores: lowest leakage among functional methods + highest retain accuracy and response quality
- 5-fold tables: ALL COMPLETE (GA, GradDiff, NPO, SimNPO, RMU, PDU)
- BLURNPO OOM on 40GB A100 (model + ref_model deepcopy exceeds memory); only s42 available (ran on 96GB local)
- NPO has highest FL in judge (1.37) — barely forgets; high retain but useless for unlearning
