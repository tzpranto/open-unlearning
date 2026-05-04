# MUSE Sustainability & Scalability (Llama-2-7b-hf, News)

Updated: 2026-05-04

## Scalability

Single-shot unlearning on progressively larger forget sets (1x, 2x, 3x, 4x the standard 889 samples).

HM = harmonic mean of (1-forget_knowmem, 1-verbmem, retain).

| Scale | forget_size | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | HM↑ |
| --- | --- | --- | --- | --- | --- | --- |
| PDU 5-fold ref | 889 | 0.525 | 0.092 | 0.508 | 0.024 | 0.577 |
| scal/forget_1 | 889 | 0.533 | 0.071 | 0.538 | 0.018 | 0.591 |
| scal/forget_2 | 1778 | 0.495 | 0.050 | 0.467 | 0.016 | 0.579 |
| scal/forget_3 | 2667 | 0.538 | 0.024 | 0.488 | 0.015 | 0.573 |
| scal/forget_4 | 3554 | 0.321 | 0.021 | 0.002 | 0.014 | 0.005 |

## Sustainability

Sequential unlearning: each step applies PDU on a new 889-sample forget set, starting from the previous step's checkpoint.

HM = harmonic mean of (1-forget_knowmem, 1-verbmem, retain).

| Step | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | HM↑ |
| --- | --- | --- | --- | --- | --- |
| sust/forget_1 | 0.574 | 0.249 | 0.543 | 0.056 | 0.543 |
| sust/forget_2 | 0.523 | 0.036 | 0.504 | 0.017 | 0.586 |
| sust/forget_3 | 0.552 | 0.014 | 0.115 | 0.008 | 0.251 |
| sust/forget_4 | 0.481 | 0.010 | 0.045 | 0.008 | 0.120 |

## LLM Judge — Scalability

FL = forget leakage (lower = better). RA = retain accuracy, RQ = response quality (higher = better). HM = hmean(1-FL/2, RA/2, ret_RQ/2).

| Scale | FL↓ | RA↑ | ret_RQ↑ | HM↑ |
| --- | --- | --- | --- | --- |
| scal/forget_1 | 0.88 | 1.20 | 1.70 | 0.648 |
| scal/forget_2 | 0.75 | 1.08 | 1.62 | 0.640 |
| scal/forget_3 | 0.74 | 1.22 | 1.72 | 0.683 |
| scal/forget_4 | 0.53 | 0.02 | 0.02 | 0.015 |

## LLM Judge — Sustainability

| Step | FL↓ | RA↑ | ret_RQ↑ | HM↑ |
| --- | --- | --- | --- | --- |
| sust/forget_1 | 1.20 | 1.24 | 1.69 | 0.566 |
| sust/forget_2 | 0.74 | 1.19 | 1.67 | 0.673 |
| sust/forget_3 | 0.73 | 0.58 | 0.63 | 0.366 |
| sust/forget_4 | 0.65 | 0.12 | 0.13 | 0.089 |

## Notes

- All runs use seed=42, bsz=2, accum=16, gradient_checkpointing=true
- Model: Llama-2-7b-hf, Data: MUSE News
- Gold (retrain): forget_knowmem=0.324, retain=0.552, HM=0.660
- Scalability tests if method handles larger forget sets in one shot
- Sustainability tests if method can be applied repeatedly without destroying the model
- scal/forget_1 = sust/forget_1 = standard News forget split (889 samples)
- Scalability: retain collapses at 4x scale (3554 samples) but holds at 1-3x
- Sustainability: retain degrades progressively after step 2, model largely destroyed by step 4
